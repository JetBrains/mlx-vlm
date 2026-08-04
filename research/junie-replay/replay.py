"""Replay a captured Junie session against the local server and report stats.

Posts every request body in ``requests/`` (in filename order) to the
server's chat-completions endpoint, then correlates each with the server
log to print per-request serving stats:

  - KV cached: cached prompt tokens / total prompt tokens
  - prefill speed over the NEW (uncached) tokens
  - generation speed
  - MTP acceptance (accepted tokens/round and % of drafted accepted)
  - n-gram prompt-lookup rounds and accepted tokens/round

and a weighted mean of each at the end.

The requests are a real session (contexts growing 80KB -> 130KB), so a
replay exercises the whole serving stack the way production does: APC
session/disk reuse for prefill, MTP + prompt-lookup speculation for decode.

Usage (server must be running, e.g. via start.sh):
  python research/junie-replay/replay.py [--url http://localhost:8085]
      [--log /tmp/mlx_server.log] [--requests research/junie-replay/requests]
"""

import argparse
import json
import os
import re
import sys
import urllib.request

PREFILL_RE = re.compile(
    r"Prefill completed: .*prompt_tokens=(\d+) cached_tokens=(\d+) "
    r"elapsed=([\d.]+)s"
)
DECODE_RE = re.compile(
    r"Decode completed: .*generated_tokens=(\d+) elapsed=([\d.]+)s"
)
SPEC_RE = re.compile(
    r"Speculative decode: .*rounds=(\d+) accepted_tokens_per_round=([\d.]+)"
    r"(?: accept_rate=([\d.]+)%)?"
    r"(?: ngram_rounds=(\d+) ngram_accepted_tokens_per_round=([\d.]+))?"
)


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
        json.loads(resp.read())


def parse_new_log_lines(log_path, offset):
    with open(log_path) as f:
        f.seek(offset)
        text = f.read()
    stats = {}
    m = PREFILL_RE.search(text)
    if m:
        stats["prompt"] = int(m.group(1))
        stats["cached"] = int(m.group(2))
        stats["prefill_s"] = float(m.group(3))
    m = DECODE_RE.search(text)
    if m:
        stats["gen"] = int(m.group(1))
        stats["decode_s"] = float(m.group(2))
    m = SPEC_RE.search(text)
    if m:
        stats["rounds"] = int(m.group(1))
        stats["accept_tpr"] = float(m.group(2))
        stats["accept_rate"] = float(m.group(3)) if m.group(3) else None
        stats["ngram_rounds"] = int(m.group(4) or 0)
        stats["ngram_tpr"] = float(m.group(5) or 0.0)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8085")
    ap.add_argument("--log", default="/tmp/mlx_server.log")
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

    totals = {
        "prompt": 0, "cached": 0, "new": 0, "prefill_s": 0.0,
        "gen": 0, "decode_s": 0.0, "rounds": 0, "accepted": 0.0,
        "drafted_rate_num": 0.0, "drafted_rate_den": 0,
        "ngram_rounds": 0, "ngram_tokens": 0.0,
    }

    for path in files:
        offset = os.path.getsize(args.log)
        post(args.url, path)
        s = parse_new_log_lines(args.log, offset)
        if "prompt" not in s or "gen" not in s:
            print(f"{os.path.basename(path)}: (no server log stats found)")
            continue
        new = s["prompt"] - s["cached"]
        prefill_tps = new / s["prefill_s"] if s["prefill_s"] > 0 else 0.0
        gen_tps = s["gen"] / s["decode_s"] if s["decode_s"] > 0 else 0.0
        mtp = (
            f"mtp={s['accept_tpr']:.2f}tok/round"
            + (f" ({s['accept_rate']:.0f}%)" if s.get("accept_rate") else "")
            if s.get("rounds")
            else "mtp=n/a"
        )
        ngram = (
            f" ngram={s['ngram_rounds']}r@{s['ngram_tpr']:.1f}tok"
            if s.get("ngram_rounds")
            else ""
        )
        print(
            f"{os.path.basename(path)}: cached={s['cached']}/{s['prompt']} "
            f"({100.0 * s['cached'] / s['prompt']:.0f}%) | "
            f"prefill_new={new}tok @{prefill_tps:.0f}tok/s | "
            f"decode={s['gen']}tok @{gen_tps:.1f}tok/s | {mtp}{ngram}"
        )

        totals["prompt"] += s["prompt"]
        totals["cached"] += s["cached"]
        totals["new"] += new
        totals["prefill_s"] += s["prefill_s"]
        totals["gen"] += s["gen"]
        totals["decode_s"] += s["decode_s"]
        if s.get("rounds"):
            totals["rounds"] += s["rounds"]
            totals["accepted"] += s["accept_tpr"] * s["rounds"]
            if s.get("accept_rate") is not None:
                totals["drafted_rate_num"] += s["accept_rate"] * s["rounds"]
                totals["drafted_rate_den"] += s["rounds"]
        totals["ngram_rounds"] += s.get("ngram_rounds", 0)
        totals["ngram_tokens"] += s.get("ngram_tpr", 0.0) * s.get("ngram_rounds", 0)

    print("-" * 72)
    n = len(files)
    cached_pct = 100.0 * totals["cached"] / totals["prompt"] if totals["prompt"] else 0
    prefill_tps = totals["new"] / totals["prefill_s"] if totals["prefill_s"] else 0
    gen_tps = totals["gen"] / totals["decode_s"] if totals["decode_s"] else 0
    mtp_tpr = totals["accepted"] / totals["rounds"] if totals["rounds"] else 0
    mtp_rate = (
        totals["drafted_rate_num"] / totals["drafted_rate_den"]
        if totals["drafted_rate_den"]
        else 0
    )
    ngram_tpr = (
        totals["ngram_tokens"] / totals["ngram_rounds"]
        if totals["ngram_rounds"]
        else 0
    )
    print(
        f"mean over {n} requests: KV cached {cached_pct:.0f}% "
        f"({totals['cached']}/{totals['prompt']}) | "
        f"prefill {prefill_tps:.0f}tok/s over {totals['new']} new tokens "
        f"({totals['prefill_s']:.1f}s) | "
        f"generation {gen_tps:.1f}tok/s ({totals['gen']} tokens, "
        f"{totals['decode_s']:.1f}s) | "
        f"MTP accept {mtp_tpr:.2f}tok/round ({mtp_rate:.0f}% of drafted) | "
        f"ngram {totals['ngram_rounds']} rounds @{ngram_tpr:.1f}tok"
    )


if __name__ == "__main__":
    main()
