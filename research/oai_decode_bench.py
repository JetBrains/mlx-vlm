#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Run standardized inference benchmarks through an OpenAI-compatible endpoint.

The script has no third-party dependencies and is intended to be run with uv:

    uv run benchmarks/oai_decode_bench.py \
      --base-url http://127.0.0.1:8000/v1 \
      --model my-model

To compare a server without and with a speculative drafter, expose each server
configuration at a different URL and name both arms:

    uv run benchmarks/oai_decode_bench.py \
      --arm off=http://127.0.0.1:8000/v1 \
      --arm mtp=http://127.0.0.1:8001/v1 \
      --model my-model

The endpoints must implement streamed ``/chat/completions`` responses and
return ``usage.prompt_tokens`` and ``usage.completion_tokens``. Prompt lengths
are calibrated against the first arm because tokenization is not part of the
OpenAI API. Every arm then receives exactly the same prompt text.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

MARKER_PLACEHOLDER = "000000000000000000"
SYSTEM_PROMPT = (
    f"Benchmark run marker: {MARKER_PLACEHOLDER}. "
    "You are taking part in a deterministic inference performance benchmark. "
    "Read the supplied technical record and write a detailed engineering "
    "analysis. Continue until the response limit; do not stop early."
)
QUESTION = (
    "\n\nUsing the complete record above, produce a detailed technical review. "
    "Discuss interactions, failure modes, and concrete improvements. Continue "
    "until the response limit and do not end with a short summary."
)

NOUNS = (
    "scheduler",
    "decoder",
    "cache",
    "kernel",
    "request",
    "worker",
    "router",
    "tensor",
    "checkpoint",
    "stream",
    "allocator",
    "batch",
    "profile",
)
VERBS = (
    "measured",
    "validated",
    "replayed",
    "compared",
    "traced",
    "rejected",
    "accepted",
    "prefetched",
    "synchronized",
    "recorded",
    "partitioned",
)
QUALIFIERS = (
    "cold",
    "sustained",
    "interleaved",
    "deterministic",
    "uncached",
    "sampled",
    "quantized",
    "concurrent",
    "bounded",
    "reproducible",
    "asynchronous",
)


@dataclass(frozen=True)
class BenchmarkCase:
    key: str
    label: str
    input_tokens: int
    output_tokens: int
    cached: bool = False
    ceiling: bool = False


@dataclass(frozen=True)
class Arm:
    name: str
    base_url: str

    @property
    def chat_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"


@dataclass
class Trial:
    arm: str
    requested_context_tokens: int
    prompt_tokens: int
    cached_prompt_tokens: int | None
    uncached_prompt_tokens: int
    completion_tokens: int
    ttft_seconds: float
    prefill_tokens_per_second: float
    decode_seconds: float | None
    decode_tokens_per_second: float | None
    total_seconds: float
    end_to_end_tokens_per_second: float
    finish_reason: str | None
    repeat: int


@dataclass
class CaseRun:
    arm: str
    case: str
    repeat: int
    measurement: Trial
    cache_fill: Trial | None = None


class ApiError(RuntimeError):
    pass


def parse_positive_csv(value: str, option: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{option} must contain integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(f"{option} values must be positive")
    return values


def parse_arm(value: str) -> Arm:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--arm must be NAME=BASE_URL")
    name, base_url = value.split("=", 1)
    name = name.strip()
    base_url = base_url.strip()
    if not name or not base_url:
        raise argparse.ArgumentTypeError("--arm must be NAME=BASE_URL")
    if not base_url.startswith(("http://", "https://")):
        raise argparse.ArgumentTypeError("arm URL must start with http:// or https://")
    return Arm(name=name, base_url=base_url)


def parse_header(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--header must be NAME=VALUE")
    name, header_value = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("header name cannot be empty")
    return name.strip(), header_value.strip()


def technical_corpus(minimum_characters: int, nonce: int = 0) -> str:
    """Return deterministic, varied text without reading repository files."""

    parts: list[str] = []
    length = 0
    index = 0
    while length < minimum_characters:
        numeric = index + nonce
        noun = NOUNS[index % len(NOUNS)]
        other = NOUNS[(index * 7 + 3) % len(NOUNS)]
        verb = VERBS[(index * 5 + 1) % len(VERBS)]
        qualifier = QUALIFIERS[(index * 11 + 2) % len(QUALIFIERS)]
        line = (
            f"Record {numeric % 1_000_000:06d}: the {qualifier} {noun} {verb} "
            f"{17 + numeric % 983:03d} events while the {other} processed shard "
            f"{numeric % 97:02d}. Latency bucket {numeric % 31:02d} changed by "
            f"{(numeric * 13) % 211 - 105:+04d} microseconds; checksum "
            f"{(numeric * 2654435761) % 10_000_000_000:010d}.\n"
        )
        parts.append(line)
        length += len(line)
        index += 1
    return "".join(parts)


def messages_for(source: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "TECHNICAL RECORD:\n" + source + QUESTION},
    ]


def ceiling_messages_for(source: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"Benchmark run marker: {MARKER_PLACEHOLDER}. "
                "Continue the sequence supplied by the user exactly. Emit only "
                "sequence words and continue until the response limit."
            ),
        },
        {
            "role": "user",
            "content": "Continue this sequence:\n" + source + "\nContinuation:\n",
        },
    ]


def ceiling_corpus(minimum_characters: int) -> str:
    pattern = " alpha beta gamma delta epsilon zeta eta theta"
    repeats = minimum_characters // len(pattern) + 2
    return (pattern * repeats)[:minimum_characters]


class Client:
    def __init__(
        self,
        *,
        api_key: str | None,
        headers: dict[str, str],
        timeout: float,
        insecure: bool,
        max_tokens_field: str,
        extra_body: dict[str, Any],
    ) -> None:
        self.headers = {"Content-Type": "application/json", **headers}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.timeout = timeout
        self.context = ssl._create_unverified_context() if insecure else None
        self.max_tokens_field = max_tokens_field
        self.extra_body = extra_body

    def _payload(
        self,
        model: str,
        messages: list[dict[str, str]],
        output_tokens: int,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            self.max_tokens_field: output_tokens,
            "temperature": 0,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        payload.update(self.extra_body)
        return payload

    def _open(self, url: str, payload: dict[str, Any]):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers=self.headers,
            method="POST",
        )
        try:
            return urllib.request.urlopen(
                request, timeout=self.timeout, context=self.context
            )
        except urllib.error.HTTPError as error:
            body = error.read(2000).decode(errors="replace")
            raise ApiError(f"HTTP {error.code} from {url}: {body}") from error
        except urllib.error.URLError as error:
            raise ApiError(f"request to {url} failed: {error.reason}") from error

    def prompt_tokens(
        self, arm: Arm, model: str, messages: list[dict[str, str]]
    ) -> int:
        payload = self._payload(model, messages, 1, stream=False)
        with self._open(arm.chat_url, payload) as response:
            parsed = json.loads(response.read().decode())
        usage = parsed.get("usage") or {}
        value = usage.get("prompt_tokens")
        if value is None:
            raise ApiError(
                f"{arm.name} returned no usage.prompt_tokens during calibration"
            )
        return int(value)

    def stream_trial(
        self,
        arm: Arm,
        model: str,
        messages: list[dict[str, str]],
        output_tokens: int,
        requested_context: int,
        repeat: int,
    ) -> Trial:
        payload = self._payload(model, messages, output_tokens, stream=True)
        started = time.perf_counter()
        first_generated: float | None = None
        last_generated: float | None = None
        usage: dict[str, Any] = {}
        finish_reason: str | None = None

        with self._open(arm.chat_url, payload) as response:
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as error:
                    raise ApiError(
                        f"invalid SSE JSON from {arm.name}: {data[:500]}"
                    ) from error
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
                    if _choice_has_generated_data(choice):
                        now = time.perf_counter()
                        if first_generated is None:
                            first_generated = now
                        last_generated = now
        ended = time.perf_counter()

        if first_generated is None:
            raise ApiError(f"{arm.name} streamed no generated content")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is None or completion_tokens is None:
            raise ApiError(
                f"{arm.name} returned no final streamed token usage; the endpoint "
                "must support stream_options.include_usage"
            )
        completion_tokens = int(completion_tokens)
        prompt_tokens = int(prompt_tokens)
        prompt_details = usage.get("prompt_tokens_details") or {}
        cached_value = prompt_details.get("cached_tokens")
        cached_tokens = int(cached_value) if cached_value is not None else None
        uncached_tokens = (
            max(0, prompt_tokens - cached_tokens)
            if cached_tokens is not None
            else prompt_tokens
        )
        total_seconds = ended - started
        ttft_seconds = first_generated - started
        decode_seconds: float | None = None
        decode_tps: float | None = None
        if completion_tokens > 1 and last_generated is not None:
            elapsed = last_generated - first_generated
            if elapsed > 0:
                decode_seconds = elapsed
                decode_tps = (completion_tokens - 1) / elapsed
        return Trial(
            arm=arm.name,
            requested_context_tokens=requested_context,
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached_tokens,
            uncached_prompt_tokens=uncached_tokens,
            completion_tokens=completion_tokens,
            ttft_seconds=ttft_seconds,
            prefill_tokens_per_second=uncached_tokens / max(ttft_seconds, 1e-9),
            decode_seconds=decode_seconds,
            decode_tokens_per_second=decode_tps,
            total_seconds=total_seconds,
            end_to_end_tokens_per_second=completion_tokens / max(total_seconds, 1e-9),
            finish_reason=finish_reason,
            repeat=repeat,
        )


def _choice_has_generated_data(choice: dict[str, Any]) -> bool:
    if choice.get("text"):
        return True
    delta = choice.get("delta") or {}
    for key in ("content", "reasoning_content", "refusal"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            return True
        if isinstance(value, list) and value:
            return True
    return bool(delta.get("tool_calls") or delta.get("function_call"))


def calibrate_prompt(
    client: Client,
    arm: Arm,
    model: str,
    target_tokens: int,
    corpus: str,
    message_builder: Callable[[str], list[dict[str, str]]],
    attempts: int,
    tolerance: int,
) -> tuple[list[dict[str, str]], int, int]:
    """Fit a corpus prefix using prompt-token usage reported by the endpoint."""

    overhead = client.prompt_tokens(arm, model, message_builder(""))
    if overhead >= target_tokens:
        raise ApiError(
            f"fixed chat template uses {overhead} tokens, which does not fit in "
            f"the requested {target_tokens}-token context"
        )

    source_chars = min(len(corpus), max(64, (target_tokens - overhead) * 4))
    best: tuple[int, int] | None = None
    seen: set[int] = set()
    for _ in range(attempts):
        source_chars = max(1, min(len(corpus), source_chars))
        if source_chars in seen:
            break
        seen.add(source_chars)
        actual = client.prompt_tokens(
            arm, model, message_builder(corpus[:source_chars])
        )
        if best is None or abs(actual - target_tokens) < abs(best[1] - target_tokens):
            best = (source_chars, actual)
        error = target_tokens - actual
        if abs(error) <= tolerance:
            break
        contributed = max(1, actual - overhead)
        desired = target_tokens - overhead
        source_chars = round(source_chars * desired / contributed)

    if best is None:
        raise AssertionError("calibration made no request")
    source_chars, actual = best
    if abs(actual - target_tokens) > tolerance:
        raise ApiError(
            f"could not calibrate {target_tokens} tokens within +/-{tolerance}; "
            f"closest result was {actual}. Increase --calibration-attempts or "
            "--context-tolerance."
        )
    return message_builder(corpus[:source_chars]), actual, source_chars


def messages_for_trial(
    calibrated: list[dict[str, str]], *, marker: str | None
) -> list[dict[str, str]]:
    """Clone a calibrated prompt and optionally replace its leading cache key."""

    result = [dict(message) for message in calibrated]
    if marker is None:
        return result
    replaced = result[0]["content"].replace(MARKER_PLACEHOLDER, marker, 1)
    if replaced == result[0]["content"]:
        raise AssertionError("benchmark marker placeholder is missing")
    result[0]["content"] = replaced
    return result


def median_optional(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def aggregate_rows(
    arms: list[Arm], cases: list[BenchmarkCase], runs: list[CaseRun]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline: dict[str, float | None] = {}
    first_case_speed: dict[str, float | None] = {}
    for arm_index, arm in enumerate(arms):
        for case_index, case in enumerate(cases):
            bucket = [
                run for run in runs if run.arm == arm.name and run.case == case.key
            ]
            measured = [run.measurement for run in bucket]
            prefill = [run.cache_fill or run.measurement for run in bucket]
            decode_tps = median_optional(
                trial.decode_tokens_per_second for trial in measured
            )
            if arm_index == 0:
                baseline[case.key] = decode_tps
            if case_index == 0:
                first_case_speed[arm.name] = decode_tps
            base = baseline.get(case.key)
            case_base = first_case_speed.get(arm.name)
            cached_values = [
                trial.cached_prompt_tokens
                for trial in measured
                if trial.cached_prompt_tokens is not None
            ]
            rows.append(
                {
                    "case": case.label,
                    "arm": arm.name,
                    "requested": case.input_tokens,
                    "actual": round(
                        statistics.median(t.prompt_tokens for t in measured)
                    ),
                    "cached": (
                        round(statistics.median(cached_values))
                        if cached_values
                        else None
                    ),
                    "output": round(
                        statistics.median(t.completion_tokens for t in measured)
                    ),
                    "ttft": statistics.median(t.ttft_seconds for t in measured),
                    "prefill_tps": statistics.median(
                        t.prefill_tokens_per_second for t in prefill
                    ),
                    "prefill_seconds": statistics.median(
                        t.ttft_seconds for t in prefill
                    ),
                    "decode_tps": decode_tps,
                    "e2e_tps": statistics.median(
                        t.end_to_end_tokens_per_second for t in measured
                    ),
                    "request_total": statistics.median(
                        t.total_seconds for t in measured
                    ),
                    "session_total": statistics.median(
                        run.measurement.total_seconds
                        + (run.cache_fill.total_seconds if run.cache_fill else 0.0)
                        for run in bucket
                    ),
                    "vs_baseline": (
                        decode_tps / base
                        if decode_tps is not None and base not in (None, 0)
                        else None
                    ),
                    "vs_first_case": (
                        decode_tps / case_base
                        if decode_tps is not None and case_base not in (None, 0)
                        else None
                    ),
                }
            )
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    headers = (
        "Case",
        "Arm",
        "Input",
        "Cached",
        "Output",
        "Prefill tok/s",
        "Prefill s",
        "TTFT s",
        "Decode tok/s",
        "vs case 1",
        "vs arm 1",
        "Request s",
        "Session s",
    )
    body = []
    for row in rows:
        body.append(
            (
                str(row["case"]),
                str(row["arm"]),
                str(row["actual"]),
                str(row["cached"]) if row["cached"] is not None else "n/a",
                str(row["output"]),
                f"{row['prefill_tps']:.2f}",
                f"{row['prefill_seconds']:.3f}",
                f"{row['ttft']:.3f}",
                _number(row["decode_tps"]),
                (
                    f"{row['vs_first_case']:.3f}x"
                    if row["vs_first_case"] is not None
                    else "n/a"
                ),
                (
                    f"{row['vs_baseline']:.3f}x"
                    if row["vs_baseline"] is not None
                    else "n/a"
                ),
                f"{row['request_total']:.3f}",
                f"{row['session_total']:.3f}",
            )
        )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in body))
        for index in range(len(headers))
    ]
    numeric = set(range(2, len(headers)))

    def render(row: tuple[str, ...]) -> str:
        cells = [
            (
                value.rjust(widths[index])
                if index in numeric
                else value.ljust(widths[index])
            )
            for index, value in enumerate(row)
        ]
        return "| " + " | ".join(cells) + " |"

    separator = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    return "\n".join([render(headers), separator, *(render(row) for row in body)])


def _number(value: float | None) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.2f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run standardized prefill and decode benchmarks through one or more "
            "OpenAI-compatible chat-completions endpoints."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Speculative decoding is configured on the server, not by the "
            "OpenAI API. Use repeated --arm NAME=URL arguments to compare "
            "servers with the drafter off and on."
        ),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--arm",
        action="append",
        type=parse_arm,
        help="named endpoint arm, NAME=BASE_URL; repeat to compare configurations",
    )
    parser.add_argument("--model", required=True, help="model name sent to every arm")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument(
        "--suite",
        choices=("standard",),
        default="standard",
        help="standard runs cold 1K->1K and cache-primed 40K->500",
    )
    parser.add_argument(
        "--include-ceiling",
        action="store_true",
        help="add an optional low-entropy 256->1K decode-ceiling case",
    )
    parser.add_argument(
        "--contexts",
        help="custom comma-separated input lengths; replaces the standard cases",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        help="output length for custom --contexts (default: 1024)",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument(
        "--cache-mode",
        choices=("cold", "warm"),
        default="cold",
        help=(
            "cache behavior for custom --contexts; warm explicitly primes and "
            "verifies the prompt cache"
        ),
    )
    parser.add_argument(
        "--minimum-cache-ratio",
        type=float,
        default=0.95,
        help="minimum cached/input ratio required by cached cases",
    )
    parser.add_argument(
        "--allow-short-output",
        action="store_true",
        help="accept EOS before a case's requested output length",
    )
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument(
        "--calibration-attempts",
        type=int,
        default=4,
        help="maximum server token-count probes per context after fixed overhead",
    )
    parser.add_argument(
        "--context-tolerance",
        type=int,
        default=8,
        help="acceptable difference between requested and calibrated prompt tokens",
    )
    parser.add_argument(
        "--max-tokens-field",
        choices=("max_tokens", "max_completion_tokens"),
        default="max_tokens",
    )
    parser.add_argument(
        "--extra-body",
        default="{}",
        help="JSON merged into every request, e.g. '{\"ignore_eos\":true}'",
    )
    parser.add_argument(
        "--header", action="append", type=parse_header, default=[], help="NAME=VALUE"
    )
    parser.add_argument(
        "--insecure", action="store_true", help="disable TLS verification"
    )
    parser.add_argument("--json", type=Path, help="optionally save raw trials as JSON")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.contexts is not None:
        try:
            args.contexts = parse_positive_csv(args.contexts, "--contexts")
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
    if args.output_tokens is not None and args.output_tokens <= 1:
        parser.error("--output-tokens must be greater than 1")
    if args.contexts is None and args.output_tokens is not None:
        parser.error("--output-tokens is only valid with custom --contexts")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup_tokens < 0:
        parser.error("--warmup-tokens cannot be negative")
    if args.calibration_attempts <= 0:
        parser.error("--calibration-attempts must be positive")
    if args.context_tolerance < 0:
        parser.error("--context-tolerance cannot be negative")
    if not 0.0 <= args.minimum_cache_ratio <= 1.0:
        parser.error("--minimum-cache-ratio must be between 0 and 1")
    try:
        args.extra_body = json.loads(args.extra_body)
    except json.JSONDecodeError as error:
        parser.error(f"--extra-body is not valid JSON: {error}")
    if not isinstance(args.extra_body, dict):
        parser.error("--extra-body must decode to a JSON object")


def benchmark_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    if args.contexts is not None:
        output_tokens = args.output_tokens or 1024
        cases = [
            BenchmarkCase(
                key=f"custom-{context}",
                label=f"custom-{context}",
                input_tokens=context,
                output_tokens=output_tokens,
                cached=args.cache_mode == "warm",
            )
            for context in args.contexts
        ]
    else:
        cases = [
            BenchmarkCase("short", "1K->1K", 1000, 1000),
            BenchmarkCase("cached40k", "40K cached->500", 40_000, 500, cached=True),
        ]
    if args.include_ceiling:
        cases.append(
            BenchmarkCase("ceiling", "decode ceiling", 256, 1000, ceiling=True)
        )
    return cases


def validate_measurement(
    case: BenchmarkCase, trial: Trial, args: argparse.Namespace
) -> None:
    difference = abs(trial.prompt_tokens - case.input_tokens)
    if difference > args.context_tolerance:
        raise ApiError(
            f"{case.label} produced {trial.prompt_tokens} input tokens after "
            f"cache-busting, outside {case.input_tokens} +/-"
            f"{args.context_tolerance}"
        )
    if not args.allow_short_output and trial.completion_tokens != case.output_tokens:
        raise ApiError(
            f"{case.label} returned {trial.completion_tokens} output tokens, "
            f"expected {case.output_tokens}; pass --allow-short-output to keep "
            "early-EOS runs"
        )
    if case.cached:
        if trial.cached_prompt_tokens is None:
            raise ApiError(
                f"{case.label} requires usage.prompt_tokens_details.cached_tokens"
            )
        cache_ratio = trial.cached_prompt_tokens / max(trial.prompt_tokens, 1)
        if cache_ratio < args.minimum_cache_ratio:
            raise ApiError(
                f"{case.label} cached only {trial.cached_prompt_tokens}/"
                f"{trial.prompt_tokens} input tokens ({cache_ratio:.1%}); expected "
                f"at least {args.minimum_cache_ratio:.1%}"
            )


def run(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[CaseRun], dict[str, Any]]:
    arms = args.arm or [Arm("default", args.base_url)]
    if len({arm.name for arm in arms}) != len(arms):
        raise ApiError("arm names must be unique")
    cases = benchmark_cases(args)
    client = Client(
        api_key=args.api_key,
        headers=dict(args.header),
        timeout=args.timeout,
        insecure=args.insecure,
        max_tokens_field=args.max_tokens_field,
        extra_body=args.extra_body,
    )
    max_context = max(case.input_tokens for case in cases)
    real_corpus = technical_corpus(max_context * 20)
    ideal_corpus = ceiling_corpus(max_context * 20)
    prompts: dict[str, list[dict[str, str]]] = {}
    calibration: dict[str, dict[str, int]] = {}
    calibration_arm = arms[0]
    for case in cases:
        print(
            f"Calibrating {case.label} ({case.input_tokens} input tokens) "
            f"against arm {calibration_arm.name}...",
            file=sys.stderr,
            flush=True,
        )
        corpus = ideal_corpus if case.ceiling else real_corpus
        builder = ceiling_messages_for if case.ceiling else messages_for
        messages, actual, source_chars = calibrate_prompt(
            client,
            calibration_arm,
            args.model,
            case.input_tokens,
            corpus,
            builder,
            args.calibration_attempts,
            args.context_tolerance,
        )
        prompts[case.key] = messages
        calibration[case.key] = {
            "prompt_tokens": actual,
            "difference": actual - case.input_tokens,
            "source_characters": source_chars,
        }
        print(
            f"  requested={case.input_tokens}, calibrated={actual} "
            f"(difference {actual - case.input_tokens:+d})",
            file=sys.stderr,
            flush=True,
        )

    if args.warmup_tokens:
        warm_case = cases[0]
        for arm in arms:
            print(f"Warming arm {arm.name}...", file=sys.stderr, flush=True)
            client.stream_trial(
                arm,
                args.model,
                messages_for_trial(prompts[warm_case.key], marker="999999999999999999"),
                args.warmup_tokens,
                warm_case.input_tokens,
                repeat=0,
            )

    schedule = [
        (arm, case_index, case, repeat)
        for repeat in range(1, args.repeats + 1)
        for case_index, case in enumerate(cases)
        for arm in arms
    ]
    random.Random(args.seed).shuffle(schedule)
    case_runs: list[CaseRun] = []
    for index, (arm, case_index, case, repeat) in enumerate(schedule, 1):
        print(
            f"[{index}/{len(schedule)}] arm={arm.name} case={case.label} "
            f"repeat={repeat}{' (fill + cached request)' if case.cached else ''}",
            file=sys.stderr,
            flush=True,
        )
        marker = f"{(args.seed + case_index * 1_000_003 + repeat * 9176) % 10**18:018d}"
        if case.ceiling:
            messages = messages_for_trial(prompts[case.key], marker=marker)
        else:
            source_chars = calibration[case.key]["source_characters"]
            variant = technical_corpus(source_chars, nonce=int(marker))[:source_chars]
            messages = messages_for_trial(messages_for(variant), marker=marker)
        cache_fill = None
        if case.cached:
            cache_fill = client.stream_trial(
                arm,
                args.model,
                messages,
                1,
                case.input_tokens,
                repeat,
            )
        measurement = client.stream_trial(
            arm,
            args.model,
            messages,
            case.output_tokens,
            case.input_tokens,
            repeat,
        )
        validate_measurement(case, measurement, args)
        case_runs.append(
            CaseRun(
                arm=arm.name,
                case=case.key,
                repeat=repeat,
                measurement=measurement,
                cache_fill=cache_fill,
            )
        )
    rows = aggregate_rows(arms, cases, case_runs)
    metadata = {
        "model": args.model,
        "arms": [asdict(arm) for arm in arms],
        "suite": args.suite if args.contexts is None else "custom",
        "cases": [asdict(case) for case in cases],
        "repeats": args.repeats,
        "minimum_cache_ratio": args.minimum_cache_ratio,
        "seed": args.seed,
        "calibration": calibration,
        "metric": (
            "(completion_tokens - 1) / wall time from first generated SSE "
            "event to last generated SSE event"
        ),
    }
    return rows, case_runs, metadata


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    try:
        rows, case_runs, metadata = run(args)
    except (ApiError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(format_table(rows))
    print(
        "\nDecode excludes time-to-first-token. Values are medians across "
        f"{args.repeats} repeat(s)."
    )
    print(
        "Prefill is uncached input tokens / cache-fill TTFT for cached cases. "
        "Session time includes cache fill plus the measured request."
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    **metadata,
                    "summary": rows,
                    "runs": [asdict(case_run) for case_run in case_runs],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Raw results: {args.json}")


if __name__ == "__main__":
    main()
