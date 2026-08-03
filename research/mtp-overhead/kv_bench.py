"""Long-context decode benchmark: fp16 KV vs 8-bit quantized KV (Qwen3.6-27B).

Qwen3.6 is hybrid: only every 4th layer is full attention (KVCache); linear
layers keep ArraysCache recurrent state and are untouched by KV quant.
Prefills once with plain caches, snapshots all cache state, then per config
restores and optionally converts the full-attention layers via
KVCache.to_quantized(group_size, bits) before decoding.

Usage:
  .venv/bin/python research/mtp-overhead/kv_bench.py --ctx 30000 --tokens 200 \
      [--spec] [--kv-bits 8]
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault(
    "HF_HUB_CACHE", "/Users/stanislav.erokhin/.local/share/junie-local/models"
)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx

from mlx_vlm.models import cache as cache_mod
from mlx_vlm.speculative.drafters import load_drafter
from mlx_vlm.speculative.utils import (
    speculative_hidden_state,
    speculative_prefill_kwargs,
)
from mlx_vlm.utils import load

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as bench_mod

MODEL = "mlx-community/Qwen3.6-27B-4bit"
DRAFT = "mlx-community/Qwen3.6-27B-MTP-4bit"
REQ = "/tmp/req_big.json"


def build_ids(processor, target_ctx):
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    msgs = [
        m
        for m in json.load(open(REQ))["messages"]
        if isinstance(m.get("content"), str)
    ]
    content = msgs[0]["content"]
    # Scale the context by repeating the request body as extra user turns.
    ids = tok.encode(
        tok.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=False,
            tokenize=False,
        )
    )
    reps = max(1, target_ctx // max(1, len(ids)))
    chat = [{"role": "user", "content": content} for _ in range(reps)]
    chat.append({"role": "user", "content": bench_mod.LONG_FORM_SUFFIX})
    text = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    out = tok.encode(text)
    return out[-target_ctx:] if len(out) > target_ctx else out


def snapshot(pc):
    snap = []
    for c in pc:
        st = c.state
        st = list(st) if isinstance(st, list) else tuple(st)
        snap.append((st, getattr(c, "offset", None)))
    return snap


def restore(model, snap):
    lm = model.language_model
    pc = cache_mod.make_prompt_cache(lm)
    for c, (st, offset) in zip(pc, snap):
        c.state = list(st) if isinstance(st, list) else st
        if offset is not None and not isinstance(c, cache_mod.ArraysCache):
            c.offset = offset
    return pc


def quantize_full_attn(pc, group_size, bits):
    n = 0
    for i, c in enumerate(pc):
        if isinstance(c, cache_mod.KVCache):
            pc[i] = c.to_quantized(group_size=group_size, bits=bits)
            n += 1
    mx.eval([x for c in pc if hasattr(c, "bits") for kv in (c.keys, c.values) for x in kv])
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=30000)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--kv-bits", type=int, default=8)
    ap.add_argument("--kv-group-size", type=int, default=64)
    ap.add_argument("--spec", action="store_true", help="also run MTP speculative")
    ap.add_argument("--prefill-step", type=int, default=2048)
    args = ap.parse_args()

    model, processor = load(MODEL)
    drafter, _ = load_drafter(DRAFT, kind="mtp")

    ids = build_ids(processor, args.ctx)
    print(f"prompt tokens: {len(ids)}")

    t0 = time.perf_counter()
    pc, first, hidden, shared_kv = bench_mod.prefill(
        model, drafter, ids, step=args.prefill_step
    )
    print(f"prefill: {time.perf_counter() - t0:.1f}s")
    snap = snapshot(pc)

    results = {}

    def report(name, r, ref=None):
        results[name] = r
        extra = ""
        if ref is not None and "generated_tokens" in r and "generated_tokens" in ref:
            a, b = ref["generated_tokens"], r["generated_tokens"]
            n = min(len(a), len(b))
            div = next((i for i in range(n) if a[i] != b[i]), None)
            extra = f"  first_divergence_vs_fp16={div}"
        print(f"{name}: {r['tps']:.1f} tok/s{extra}")

    # 1. plain decode, fp16 KV
    r_base = bench_mod.baseline_decode(model, pc, first, args.tokens)
    report("decode_fp16", r_base)

    # 2. plain decode, quantized KV
    pc = restore(model, snap)
    nq = quantize_full_attn(pc, args.kv_group_size, args.kv_bits)
    print(f"(quantized {nq} full-attention layers to {args.kv_bits}-bit)")
    r_q = bench_mod.baseline_decode(model, pc, first, args.tokens)
    report(f"decode_kv{args.kv_bits}", r_q)

    # 3. plain decode, TurboQuant KV (fused decode-attention kernel)
    from mlx_vlm.turboquant import TurboQuantKVCache

    pc = restore(model, snap)
    for i, c in enumerate(pc):
        if isinstance(c, cache_mod.KVCache):
            pc[i] = TurboQuantKVCache.from_cache(c, bits=float(args.kv_bits))
    mx.eval([c.state for c in pc])
    r_t = bench_mod.baseline_decode(model, pc, first, args.tokens)
    report(f"decode_turbo{args.kv_bits}", r_t)

    if args.spec:
        pc = restore(model, snap)
        r = bench_mod.spec_decode(
            model, drafter, pc, first, hidden, shared_kv,
            n_tokens=args.tokens, block_size=3,
        )
        report("spec_fp16", r)

        pc = restore(model, snap)
        quantize_full_attn(pc, args.kv_group_size, args.kv_bits)
        r2 = bench_mod.spec_decode(
            model, drafter, pc, first, hidden, shared_kv,
            n_tokens=args.tokens, block_size=3,
        )
        report(f"spec_kv{args.kv_bits}", r2, ref=r)

    out = os.path.join(os.path.dirname(__file__), f"kv_results_{args.ctx}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
