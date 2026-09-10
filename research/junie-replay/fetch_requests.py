#!/usr/bin/env python3
"""Export Junie replay requests from raw API captures stored on Comino.

Requires Python 3.10+ locally and in the capture container, SSH access through
an existing SSH config (and VPN where required), and permission to read captures.
TeamCity runs produce the agent sessions; this reads their server-side captures,
not TeamCity artifacts. No inference is run and no remote files are changed.

  python research/junie-replay/fetch_requests.py
  python research/junie-replay/fetch_requests.py --host comino --container vllm \
      --captures-root /captures --model Qwen/Qwen3.8-27B --count 2
  ./bench.sh --requests research/junie-replay/requests-local/trajectory-1

Use --capture RELATIVE_DIRECTORY (repeatable) to restrict capture sessions under
--captures-root. Otherwise all matching model captures are considered. Like the
original export_longest.py, requests are grouped by their ISSUE DESCRIPTION and
ranked by maximum completed prompt length. Repeated attempts of the same issue
are grouped together; ongoing sessions are snapshots, not guaranteed complete
agent runs. Only complete, gap-free responses with consistent token counts are
included. Requests within each group are ordered by capture time.

Output contains only numbered request bodies and warm_up.json, in replay.py's
existing format. All request fields are preserved except stream=False. Warmup
copies the first request with max_tokens capped at 128. Manifests, response token
IDs and capture metadata are not exported. Request contents themselves remain
private and may contain secrets: generated output is ignored by Git, not safe
for publication. --output must stay under this script's ignored requests-local*/
directories. Existing output is never overwritten.
"""

import argparse
import base64
import copy
import gzip
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


class ExportError(Exception):
    pass


def completed_request(events, max_tokens):
    events = sorted(events, key=lambda event: event["seq"])
    ends = [event for event in events if event["kind"] == "request_end"]
    if (
        not ends
        or not ends[-1].get("response_complete")
        or any(
            event.get("capture_gap") or event.get("disconnected") for event in events
        )
    ):
        return None

    def body(kind):
        return json.loads(
            b"".join(
                base64.b64decode(event.get("body_b64", ""), validate=True)
                for event in events
                if event["kind"] == kind
            )
        )

    try:
        request, response = body("request_body"), body("response_body")
    except (ValueError, UnicodeError):
        return None
    if not isinstance(request, dict) or request.get("max_tokens") != max_tokens:
        return None
    if not isinstance(response, dict) or not response.get("usage"):
        return None
    issue = next(
        (
            message["content"]
            for message in request.get("messages", [])
            if message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and message["content"].startswith("## ISSUE DESCRIPTION")
        ),
        None,
    )
    if issue is None:
        return None
    usage = response["usage"]
    inputs = response.get("prompt_token_ids") or []
    outputs = [
        token
        for choice in response.get("choices", [])
        for token in (choice.get("token_ids") or [])
    ]
    if (
        len(inputs) != usage.get("prompt_tokens")
        or len(outputs) != usage.get("completion_tokens")
        or not inputs
    ):
        raise ExportError("Capture token counts are inconsistent; export stopped.")
    return (
        hashlib.sha256(issue.encode()).hexdigest(),
        events[0]["time_ns"],
        usage["prompt_tokens"],
        request,
    )


def collect_requests(root, model, count, max_tokens, captures):
    root = root.resolve()
    if captures:
        manifests = []
        for capture in captures:
            directory = (root / capture).resolve()
            if not directory.is_relative_to(root) or directory == root:
                raise ExportError(
                    "--capture must name a directory under --captures-root."
                )
            manifests.append(directory / "manifest.json")
        manifests = sorted(set(manifests))
    else:
        manifests = sorted(root.glob("*/*/manifest.json"))
    groups = {}
    for manifest in manifests:
        identity = json.loads(manifest.read_text()).get("identity", {})
        if identity.get("model") != model:
            continue
        rows = {}
        for path in sorted(manifest.parent.glob("*.jsonl.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as source:
                for line in source:
                    event = json.loads(line)
                    rows.setdefault(event["request"], []).append(event)
        for events in rows.values():
            row = completed_request(events, max_tokens)
            if row is not None:
                key, timestamp, prompt_tokens, request = row
                groups.setdefault(key, []).append((timestamp, prompt_tokens, request))
    chosen = sorted(
        groups.items(), key=lambda item: (-max(row[1] for row in item[1]), item[0])
    )[:count]
    if len(chosen) != count:
        raise ExportError(
            f"Found only {len(chosen)} eligible issue groups; requested {count}."
        )
    return [
        [row[2] for row in sorted(rows, key=lambda row: row[0])] for _, rows in chosen
    ]


def fetch_requests(args):
    command = [
        "docker",
        "exec",
        "-i",
        args.container,
        "python3",
        "-",
        "--remote",
        "--captures-root",
        args.captures_root,
        "--model",
        args.model,
        "--count",
        str(args.count),
        "--max-tokens",
        str(args.max_tokens),
    ]
    for capture in args.capture:
        command.extend(["--capture", capture])
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            args.host,
            shlex.join(command),
        ],
        input=Path(__file__).read_bytes(),
        capture_output=True,
        timeout=900,
    )
    if result.returncode:
        # Never echo captured payloads or remote diagnostics into logs.
        raise ExportError(
            "Remote export failed; check SSH, container, capture paths and token counts."
        )
    return json.loads(gzip.decompress(result.stdout))


def write_requests(output, groups):
    if output.exists() or output.is_symlink():
        raise ExportError("Output already exists; choose a new directory.")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix=".replay-", dir=output.parent) as temporary:
        stage = Path(temporary) / "export"
        stage.mkdir(mode=0o700)
        for rank, requests in enumerate(groups, 1):
            if not requests:
                raise ExportError("Cannot export an empty trajectory.")
            directory = stage / f"trajectory-{rank}"
            directory.mkdir(mode=0o700)
            width = max(2, len(str(len(requests) - 1)))
            bodies = [
                (f"{index:0{width}d}.json", request)
                for index, request in enumerate(requests)
            ]
            warmup = copy.deepcopy(requests[0])
            warmup["max_tokens"] = min(warmup["max_tokens"], 128)
            bodies.append(("warm_up.json", warmup))
            for name, original in bodies:
                body = dict(original, stream=False)
                fd = os.open(
                    directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
                with os.fdopen(fd, "w", encoding="utf-8") as target:
                    json.dump(body, target, ensure_ascii=False, indent=2)
                    target.write("\n")
        if output.exists() or output.is_symlink():
            raise ExportError("Output appeared during export; choose a new directory.")
        stage.rename(output)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="comino", help="SSH config host alias")
    parser.add_argument("--container", default="vllm")
    parser.add_argument("--captures-root", default="/captures")
    parser.add_argument(
        "--capture",
        action="append",
        default=[],
        help="Relative capture session directory (repeatable)",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3.8-27B", help="Captured model identity to select"
    )
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Select requests with this generation limit",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--remote", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.count < 1 or args.max_tokens < 1:
        parser.error("--count and --max-tokens must be positive")
    if args.host.startswith("-") or args.container.startswith("-"):
        parser.error("Invalid host or container name")
    try:
        if args.remote:
            groups = collect_requests(
                Path(args.captures_root),
                args.model,
                args.count,
                args.max_tokens,
                args.capture,
            )
            sys.stdout.buffer.write(gzip.compress(json.dumps(groups).encode()))
            return 0
        base = Path(__file__).resolve().parent
        output = (args.output or base / "requests-local").absolute()
        relative = output.resolve().relative_to(base)
        if not relative.parts or not relative.parts[0].startswith("requests-local"):
            raise ExportError(
                "--output must be under the ignored requests-local*/ directories."
            )
        if output.exists() or output.is_symlink():
            raise ExportError("Output already exists; choose a new directory.")
        groups = fetch_requests(args)
        write_requests(output, groups)
    except (
        ExportError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        EOFError,
        subprocess.TimeoutExpired,
    ) as exc:
        message = (
            str(exc)
            if isinstance(exc, ExportError)
            else "Cannot read or export captures; no payloads logged."
        )
        parser.exit(1, f"Error: {message}\n")
    for rank, requests in enumerate(groups, 1):
        print(
            f'Wrote {len(requests)} requests + warm_up.json to {output / f"trajectory-{rank}"}'
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
