"""Replay a captured Junie session against the local server and report stats.

Posts every request body in ``requests/`` (in filename order) to the
server's chat-completions endpoint and reports, from each response's
``usage`` + ``timings`` blocks (no server log needed):

  - KV cached: cached prompt tokens / total prompt tokens
  - prefill speed over the NEW (uncached) tokens
  - generation speed
  - MTP/speculative acceptance (draft_n_accepted / draft_n, tokens/round)
  - n-gram prompt-lookup share

and a weighted mean of each at the end.

The requests are a real session (contexts growing 80KB -> 130KB), so a
replay exercises the whole serving stack the way production does: APC
session/disk reuse for prefill, MTP + prompt-lookup speculation for decode.

Usage (server must be running, e.g. via start.sh):
  python research/junie-replay/replay.py [--url http://localhost:8085]
      [--requests research/junie-replay/requests]
"""

import argparse
import json
import os
import sys
import urllib.request


def post(url, path):
    with open(path) as f:
        body = json.load(f)
    body["stream"] = False
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.loads(resp.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8085")
    ap.add_argument(
        "--requests",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "requests"),
    )
    args = ap.parse_args()

    files = sorted(
        os.path.join(args.requests, f)
        for f in os.listdir(args.requests)
        if f.endswith(".json")
    )
    if not files:
        sys.exit(f"no request files in {args.requests}")

    # Inject warm_up.json as the first request to prime the KV cache with the
    # cached prompt (uses the "before commit" system prompt). Not counted in stats.
    warm_up = os.path.join(args.requests, "warm_up.json")
    if os.path.exists(warm_up):
        print(f"warming up with {os.path.basename(warm_up)}...", flush=True)
        post(args.url, warm_up)
        files = [f for f in files if os.path.basename(f) != "warm_up.json"]

    totals = {
        "prompt": 0, "cached": 0, "new": 0, "prompt_ms": 0.0,
        "gen": 0, "gen_ms": 0.0,
        "draft_n": 0, "draft_acc": 0, "rounds": 0,
        "ngram_n": 0, "ngram_acc": 0, "ngram_rounds": 0,
    }

    for path in files:
        response = post(args.url, path)
        usage = response.get("usage") or {}
        timings = response.get("timings") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        cached = int(timings.get("cache_n") or 0)
        new = int(timings.get("prompt_n") or max(0, prompt - cached))
        prompt_ms = float(timings.get("prompt_ms") or 0.0)
        gen = int(timings.get("predicted_n") or usage.get("completion_tokens") or 0)
        gen_tps = float(timings.get("predicted_per_second") or 0.0)
        gen_ms = float(timings.get("predicted_ms") or 0.0)
        prefill_tps = new / (prompt_ms / 1000.0) if prompt_ms > 0 else 0.0

        draft_n = timings.get("draft_n")
        line = (
            f"{os.path.basename(path)}: cached={cached}/{prompt} "
            f"({100.0 * cached / prompt if prompt else 0:.0f}%) | "
            f"prefill_new={new}tok @{prefill_tps:.0f}tok/s | "
            f"decode={gen}tok @{gen_tps:.1f}tok/s"
        )
        if draft_n:
            draft_acc = int(timings.get("draft_n_accepted") or 0)
            rounds = int(timings.get("draft_rounds") or 0)
            line += (
                f" | accept={100.0 * draft_acc / draft_n:.0f}% "
                f"({(draft_acc + rounds) / rounds if rounds else 0:.2f}tok/round)"
            )
            ngram_rounds = int(timings.get("ngram_rounds") or 0)
            if ngram_rounds:
                ngram_acc = int(timings.get("ngram_n_accepted") or 0)
                line += f" ngram={ngram_rounds}r/{ngram_acc}tok"
            totals["draft_n"] += draft_n
            totals["draft_acc"] += int(timings.get("draft_n_accepted") or 0)
            totals["rounds"] += rounds
            totals["ngram_n"] += int(timings.get("ngram_n") or 0)
            totals["ngram_acc"] += int(timings.get("ngram_n_accepted") or 0)
            totals["ngram_rounds"] += ngram_rounds
        print(line, flush=True)

        totals["prompt"] += prompt
        totals["cached"] += cached
        totals["new"] += new
        totals["prompt_ms"] += prompt_ms
        totals["gen"] += gen
        totals["gen_ms"] += gen_ms

    print("-" * 72)
    cached_pct = 100.0 * totals["cached"] / totals["prompt"] if totals["prompt"] else 0
    prefill_tps = (
        totals["new"] / (totals["prompt_ms"] / 1000.0) if totals["prompt_ms"] else 0
    )
    gen_tps = totals["gen"] / (totals["gen_ms"] / 1000.0) if totals["gen_ms"] else 0
    accept_pct = (
        100.0 * totals["draft_acc"] / totals["draft_n"] if totals["draft_n"] else 0
    )
    tok_per_round = (
        (totals["draft_acc"] + totals["rounds"]) / totals["rounds"]
        if totals["rounds"]
        else 0
    )
    ngram_share = 100.0 * totals["ngram_acc"] / totals["gen"] if totals["gen"] else 0
    print(
        f"mean over {len(files)} requests: KV cached {cached_pct:.0f}% "
        f"({totals['cached']}/{totals['prompt']}) | "
        f"prefill {prefill_tps:.0f}tok/s over {totals['new']} new tokens "
        f"({totals['prompt_ms'] / 1000.0:.1f}s) | "
        f"generation {gen_tps:.1f}tok/s ({totals['gen']} tokens, "
        f"{totals['gen_ms'] / 1000.0:.1f}s) | "
        f"accept {accept_pct:.0f}% of drafted ({tok_per_round:.2f}tok/round) | "
        f"ngram share {ngram_share:.0f}% of output"
    )


if __name__ == "__main__":
    main()
