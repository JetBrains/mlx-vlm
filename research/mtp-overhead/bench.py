"""Benchmark MTP speculative-decoding round overhead for Qwen3.6-27B.

Measures, in one process (model loaded once):
  - plain greedy decode baseline (fused argmax path, like the server)
  - speculative decode tok/s + acceptance for a sweep of draft block sizes
  - optional per-phase round breakdown (--phases): draft / verify / walk /
    accept_verified / rollback / rebind, with mx.eval barriers between phases

The decode loop is a B=1 specialization of ``_mtp_rounds_batch`` (the loop the
server actually runs), kept call-for-call identical so timings transfer.

Usage:
  .venv/bin/python research/mtp-overhead/bench.py --tokens 400 \
      --block-sizes 2 3 4 --phases --req /tmp/req_small.json
"""

import argparse
import json
import os
import time
from collections import defaultdict

os.environ.setdefault(
    "HF_HUB_CACHE", "/Users/stanislav.erokhin/.local/share/junie-local/models"
)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx

from mlx_vlm.models import cache as cache_mod
from mlx_vlm.speculative import mtp as mtp_mod
from mlx_vlm.speculative.common import _speculative_walk_batch
from mlx_vlm.speculative.drafters import load_drafter
from mlx_vlm.speculative.utils import (
    speculative_hidden_state,
    speculative_prefill_kwargs,
)
from mlx_vlm.utils import load

MODEL = "mlx-community/Qwen3.6-27B-4bit"
DRAFT = "mlx-community/Qwen3.6-27B-MTP-4bit"

LONG_FORM_SUFFIX = (
    "Now write a very detailed (at least 800 words) step-by-step analysis of "
    "the project structure and your implementation plan. Do not call any "
    "tools; answer in plain prose."
)


def build_prompt_ids(processor, req_path):
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    msgs = json.load(open(req_path))["messages"]
    msgs = [m for m in msgs if isinstance(m.get("content"), str)]
    msgs.append({"role": "user", "content": LONG_FORM_SUFFIX})
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok.encode(text)


def prefill(model, drafter, ids, step=2048):
    lm = model.language_model
    pc = cache_mod.make_prompt_cache(lm)
    x = mx.array([ids], dtype=mx.int32)
    while x.shape[1] > 1:
        n = min(step, x.shape[1] - 1)
        lm(x[:, :n], cache=pc, skip_logits=True)
        mx.eval([c.state for c in pc])
        x = x[:, n:]
    out = lm(x, cache=pc, **speculative_prefill_kwargs("mtp", drafter))
    first = mx.argmax(out.logits[:, -1, :], axis=-1).astype(mx.int32)
    hidden = speculative_hidden_state("mtp", out)
    shared_kv = out.shared_kv_states
    mx.eval(first, hidden)
    return pc, first, hidden, shared_kv


def baseline_decode(model, pc, first, n_tokens):
    lm = model.language_model
    y = first[:, None]
    mx.eval(y)
    start = time.perf_counter()
    for _ in range(n_tokens - 1):
        fused = getattr(lm, "fused_greedy_decode", None)
        nxt = fused(y, cache=pc) if callable(fused) else None
        if nxt is None:
            out = lm(y, cache=pc)
            nxt = mx.argmax(out.logits[:, -1, :], axis=-1)
        y = nxt.reshape(1, 1).astype(mx.int32)
        mx.eval(y)
    elapsed = time.perf_counter() - start
    return {"tokens": n_tokens - 1, "elapsed": elapsed, "tps": (n_tokens - 1) / elapsed}


def spec_decode(
    model,
    drafter,
    pc,
    first,
    hidden_prefill,
    shared_kv,
    *,
    n_tokens,
    block_size,
    force_block=True,
    phases=False,
):
    """B=1 specialization of _mtp_rounds_batch with optional phase timing."""
    lm = model.language_model

    def sampler(x):
        return mx.argmax(x, axis=-1)

    block_total = block_size
    configured = int(getattr(drafter.config, "block_size", block_size))
    drafter.prefer_requested_block_size = bool(force_block)
    drafter.reset(model)

    hidden = hidden_prefill
    if hidden.shape[1] > 1:
        hidden = hidden[:, -1:, :]
    hidden = mtp_mod._mtp_draft_hidden(lm, hidden)

    _, positions = mtp_mod._mtp_cache_positions(pc, 1)
    drafter.set_shared_kv(
        shared_kv,
        kv_offset=positions[0],
        position=mtp_mod._mtp_draft_position(mx.array(positions)),
        kv_valid_len=mx.array(positions),
        left_padding=None,
    )

    ph = defaultdict(float)

    def tick(label, t0, *arrays):
        if phases:
            arrays = [a for a in arrays if a is not None]
            if arrays:
                mx.eval(*arrays)
        dt = time.perf_counter() - t0
        ph[label] += dt
        return time.perf_counter()

    b = int(first.item())
    emitted = 1
    rounds = 0
    draft_kwargs = mtp_mod._mtp_draft_kwargs(drafter, True, sampler)
    start = time.perf_counter()

    while emitted < n_tokens:
        bs = mtp_mod._mtp_next_block_size(
            drafter, block_total, configured, n_tokens - emitted + 1
        )
        if bs <= 1:
            break

        t0 = time.perf_counter()
        b_arr = mx.array([b], dtype=mx.int32)
        draft_tokens = drafter.draft_block(
            b_arr, hidden, None, bs, sampler, mx.int32, **draft_kwargs
        )
        mx.async_eval(draft_tokens)
        t0 = tick("draft", t0, draft_tokens)

        with mx.stream(mtp_mod.generation_stream):
            verify_input = mx.concatenate([b_arr[:, None], draft_tokens], axis=1)
            verify = mtp_mod._mtp_verify_target(
                lm, verify_input, pc, sampler, sample_target_tokens=True
            )
        mx.eval(verify.target_tokens, verify.hidden)
        t0 = tick("verify", t0, verify.target_tokens, verify.hidden)

        accepted_list, new_tokens_list = _speculative_walk_batch(
            draft_tokens, verify.target_tokens, [n_tokens - emitted]
        )
        mtp_mod._record_speculative_round(
            drafter, sum(accepted_list) / len(accepted_list), bs - 1
        )
        t0 = tick("walk", t0)

        drafter.accept_verified_tokens_batch(
            verify.hidden,
            draft_tokens,
            accepted_list,
            new_tokens_list,
            sampler,
            mx.int32,
            **draft_kwargs,
        )
        t0 = tick("accept", t0, drafter._seed_token, drafter._seed_hidden)

        a = accepted_list[0]
        new_tokens = new_tokens_list[0]
        hidden = mtp_mod._mtp_draft_hidden(lm, verify.hidden[:, a : a + 1, :])
        emitted += len(new_tokens)
        if new_tokens:
            b = new_tokens[-1]
        positions[0] += a + 1

        if a < bs - 1:
            with mx.stream(mtp_mod.generation_stream):
                lm.rollback_speculative_cache(pc, verify.gdn_states, [a], bs)
            if phases:
                mx.eval([c.state for c in pc])
        t0 = tick("rollback", t0)

        rejected = bs - (a + 1)
        next_shared_kv = mtp_mod._slice_shared_kv_after_reject(
            verify.shared_kv_states, rejected
        )
        drafter.set_shared_kv(
            next_shared_kv,
            kv_offset=positions[0],
            position=mtp_mod._mtp_draft_position(mx.array(positions)),
            kv_valid_len=mx.array(positions),
            left_padding=None,
        )
        tick("rebind", t0)
        rounds += 1

    elapsed = time.perf_counter() - start
    al = drafter.accept_lens
    dl = drafter.draft_lens
    result = {
        "tokens": emitted - 1,
        "elapsed": elapsed,
        "tps": (emitted - 1) / elapsed if elapsed > 0 else 0.0,
        "rounds": rounds,
        "tokens_per_round": (emitted - 1) / rounds if rounds else 0.0,
        "mean_accept": sum(al) / len(al) if al else 0.0,
        "accept_rate": 100.0 * sum(al) / sum(dl) if dl and sum(dl) else 0.0,
        "ms_per_round": 1000.0 * elapsed / rounds if rounds else 0.0,
    }
    if phases:
        result["phase_ms_per_round"] = {
            k: 1000.0 * v / rounds for k, v in ph.items()
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--req", default="/tmp/req_small.json")
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--block-sizes", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--phases", action="store_true")
    ap.add_argument("--adaptive", action="store_true", help="let _mtp_next_block_size adapt instead of forcing")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--prefill-step", type=int, default=2048)
    args = ap.parse_args()

    print(f"loading target: {MODEL}")
    model, processor = load(MODEL)
    print(f"loading drafter: {DRAFT}")
    drafter, kind = load_drafter(DRAFT, kind="mtp")
    assert kind == "mtp"

    ids = build_prompt_ids(processor, args.req)
    print(f"prompt tokens: {len(ids)}")

    def fresh():
        t0 = time.perf_counter()
        state = prefill(model, drafter, ids, step=args.prefill_step)
        print(f"  (prefill {time.perf_counter() - t0:.1f}s)")
        return state

    results = {}

    if not args.skip_baseline:
        pc, first, hidden, shared_kv = fresh()
        r = baseline_decode(model, pc, first, args.tokens)
        results["baseline"] = r
        print(f"baseline: {r['tps']:.1f} tok/s ({r['tokens']} tokens)")
        del pc

    for bs in args.block_sizes:
        pc, first, hidden, shared_kv = fresh()
        r = spec_decode(
            model, drafter, pc, first, hidden, shared_kv,
            n_tokens=args.tokens, block_size=bs, phases=False,
            force_block=not args.adaptive,
        )
        results[f"block{bs}"] = r
        print(
            f"block={bs}: {r['tps']:.1f} tok/s  "
            f"tokens/round={r['tokens_per_round']:.2f}  "
            f"accept={r['mean_accept']:.2f}/{bs - 1} ({r['accept_rate']:.0f}%)  "
            f"ms/round={r['ms_per_round']:.1f}  rounds={r['rounds']}"
        )
        del pc

        if args.phases:
            pc, first, hidden, shared_kv = fresh()
            r = spec_decode(
                model, drafter, pc, first, hidden, shared_kv,
                n_tokens=args.tokens, block_size=bs, phases=True,
            )
            results[f"block{bs}_phases"] = r
            pm = r["phase_ms_per_round"]
            total = sum(pm.values())
            print(f"  phases (synced) ms/round total={total:.1f}:")
            for k in ("draft", "verify", "walk", "accept", "rollback", "rebind"):
                if k in pm:
                    print(f"    {k:>9}: {pm[k]:6.2f} ms ({100 * pm[k] / total:4.1f}%)")
            del pc

    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
