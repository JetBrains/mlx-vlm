"""Per-component time profile of prefill and decode for a captured Junie request.

Loads Qwen3.6-27B-MLX-4bit the way the server does on this machine (int8 W8A8
prefill patch applied, prefill step 1024, no MTP drafter), builds the prompt
of one captured request through the chat template (tools included, thinking
off), then measures where the time goes:

  - prefill: full-attention layers vs GDN (linear-attention) layers vs MLP,
    with mx.eval barriers around each submodule call (one instrumented run)
    plus an un-instrumented run for the true total.
  - decode (plain greedy, no MTP): the same breakdown per token, plus the
    fused_greedy_decode tok/s the server actually gets without a drafter.
  - per-token memory reads during decode, from the real weight/cache buffer
    sizes: weights by component, full-attn KV cache at this context length,
    GDN recurrent + conv state.

Barrier timing serializes the GPU at submodule boundaries, so the
instrumented totals run slower than the real ones; the split is what to
read, the un-instrumented totals are the denominator that matters.

Usage:
  .venv/bin/python research/junie-replay/profile_layers.py \
      [--req research/junie-replay/requests/10.json] [--decode-tokens 200]
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
import mlx.nn as nn

from mlx_vlm.int8_prefill import apply as apply_int8_prefill
from mlx_vlm.models import cache as cache_mod
from mlx_vlm.models.qwen3_5 import language as q35
from mlx_vlm.utils import load

MODEL_DIR = (
    "/Users/stanislav.erokhin/.local/share/junie-local/models/Qwen3.6-27B-MLX-4bit"
)

TIMES = defaultdict(float)
CALLS = defaultdict(int)
ENABLED = False


def _wrap(cls, name):
    orig = cls.__call__

    def timed(self, *a, **k):
        if not ENABLED:
            return orig(self, *a, **k)
        t0 = time.perf_counter()
        out = orig(self, *a, **k)
        mx.eval(out)
        TIMES[name] += time.perf_counter() - t0
        CALLS[name] += 1
        return out

    cls.__call__ = timed


class TimedHead(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def __call__(self, x):
        if not ENABLED:
            return self.inner(x)
        t0 = time.perf_counter()
        out = self.inner(x)
        mx.eval(out)
        TIMES["lm_head"] += time.perf_counter() - t0
        CALLS["lm_head"] += 1
        return out


def reset_times():
    TIMES.clear()
    CALLS.clear()


def build_prompt_ids(processor, req_path):
    # Mirrors the server's message normalization in openai.py: list content is
    # flattened to text and tool_call arguments become dicts for the template.
    from mlx_vlm.prompt_utils import extract_text_from_content

    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    body = json.load(open(req_path))
    msgs = body["messages"]
    for m in msgs:
        if m.get("content") is None:
            m["content"] = ""
        elif isinstance(m["content"], list):
            m["content"] = extract_text_from_content(m["content"])
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except (json.JSONDecodeError, TypeError):
                    fn["arguments"] = {}
    text = tok.apply_chat_template(
        msgs,
        tools=body.get("tools"),
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    return tok.encode(text)


def prefill(lm, ids, step):
    pc = cache_mod.make_prompt_cache(lm)
    x = mx.array([ids], dtype=mx.int32)
    t0 = time.perf_counter()
    while x.shape[1] > 1:
        n = min(step, x.shape[1] - 1)
        lm(x[:, :n], cache=pc, skip_logits=True)
        mx.eval([c.state for c in pc])
        x = x[:, n:]
    out = lm(x, cache=pc)
    y = mx.argmax(out.logits[:, -1, :], axis=-1).astype(mx.int32)
    mx.eval(y)
    return pc, y, time.perf_counter() - t0


def decode_plain(lm, pc, first, n_tokens):
    y = first[:, None]
    mx.eval(y)
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        out = lm(y, cache=pc)
        y = mx.argmax(out.logits[:, -1, :], axis=-1).reshape(1, 1).astype(mx.int32)
        mx.eval(y)
    return time.perf_counter() - t0


def decode_fused(lm, pc, first, n_tokens):
    y = first[:, None]
    mx.eval(y)
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        nxt = lm.fused_greedy_decode(y, cache=pc)
        if nxt is None:
            out = lm(y, cache=pc)
            nxt = mx.argmax(out.logits[:, -1, :], axis=-1)
        y = nxt.reshape(1, 1).astype(mx.int32)
        mx.eval(y)
    return time.perf_counter() - t0


def report_split(title, total, n_units, unit):
    tracked = sum(TIMES.values())
    other = max(0.0, total - tracked)
    print(f"\n{title}: instrumented total {total:.2f}s ({n_units} {unit})")
    for k in ("full_attn", "gdn", "mlp", "lm_head"):
        if k in TIMES:
            print(
                f"  {k:>9}: {TIMES[k]:7.2f}s  {100 * TIMES[k] / total:5.1f}%"
                f"  ({CALLS[k]} calls)"
            )
    print(f"  {'other':>9}: {other:7.2f}s  {100 * other / total:5.1f}%")


def weight_bytes_by_component(lm):
    from mlx.utils import tree_flatten

    buckets = defaultdict(int)
    for path, arr in tree_flatten(lm.parameters()):
        if not isinstance(arr, mx.array):
            continue
        if ".self_attn." in path:
            buckets["full_attn weights"] += arr.nbytes
        elif ".linear_attn." in path:
            buckets["gdn weights"] += arr.nbytes
        elif ".mlp." in path:
            buckets["mlp weights"] += arr.nbytes
        elif "lm_head" in path:
            buckets["lm_head weights"] += arr.nbytes
        elif "embed_tokens" in path:
            buckets["embed (not read per token)"] += arr.nbytes
        else:
            buckets["norms/other weights"] += arr.nbytes
    return buckets


def cache_bytes(pc, layers):
    kv = 0
    gdn = 0
    for layer, c in zip(layers, pc):
        state = c.state
        arrays = state if isinstance(state, (list, tuple)) else [state]
        n = sum(a.nbytes for a in arrays if isinstance(a, mx.array))
        if getattr(layer, "is_linear", False):
            gdn += n
        else:
            kv += n
    return kv, gdn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--req",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "requests", "10.json"),
    )
    ap.add_argument("--decode-tokens", type=int, default=200)
    ap.add_argument("--prefill-step", type=int, default=1024)
    ap.add_argument("--max-prompt", type=int, default=0, help="truncate prompt (smoke test)")
    args = ap.parse_args()

    print(f"loading {MODEL_DIR}")
    model, processor = load(MODEL_DIR)
    lm = model.language_model
    apply_int8_prefill()

    ids = build_prompt_ids(processor, args.req)
    if args.max_prompt:
        ids = ids[: args.max_prompt]
    print(f"prompt tokens: {len(ids)}")

    _wrap(q35.Qwen3_5Attention, "full_attn")
    _wrap(q35.Qwen3_5GatedDeltaNet, "gdn")
    _wrap(q35.Qwen3_5MLP, "mlp")
    lm.lm_head = TimedHead(lm.lm_head)

    global ENABLED

    # --- warmup: int8 weight copies, kernel JIT, page-in ---
    ENABLED = False
    wpc, wfirst, _ = prefill(lm, ids[: min(len(ids), 2048)], args.prefill_step)
    decode_plain(lm, wpc, wfirst, 4)
    decode_fused(lm, wpc, wfirst, 4)
    del wpc
    print("warmup done")

    # --- un-instrumented reference numbers ---
    pc, first, t_prefill = prefill(lm, ids, args.prefill_step)
    new_tok = len(ids)
    print(
        f"\nprefill (real, step={args.prefill_step}): {t_prefill:.2f}s"
        f"  -> {new_tok / t_prefill:.0f} tok/s"
    )

    t = decode_plain(lm, pc, first, args.decode_tokens)
    print(f"decode plain (real): {args.decode_tokens / t:.1f} tok/s")
    t = decode_fused(lm, pc, first, args.decode_tokens)
    print(f"decode fused_greedy (real, server no-MTP path): {args.decode_tokens / t:.1f} tok/s")

    # --- instrumented decode (continues on the same cache) ---
    reset_times()
    ENABLED = True
    t_dec_inst = decode_plain(lm, pc, first, args.decode_tokens)
    ENABLED = False
    ctx_len = len(ids) + 3 * args.decode_tokens
    report_split(
        f"decode split at ~{ctx_len} ctx", t_dec_inst, args.decode_tokens, "tokens"
    )

    # --- per-token memory reads during decode ---
    kv_bytes, gdn_state_bytes = cache_bytes(pc, lm.model.layers)
    buckets = weight_bytes_by_component(lm)
    print(f"\nper-decoded-token memory reads at ctx={ctx_len}:")
    total = 0
    rows = list(buckets.items()) + [
        (f"full-attn KV cache (16 layers, ctx={ctx_len})", kv_bytes),
        ("gdn recurrent+conv state (48 layers, read)", gdn_state_bytes),
    ]
    for name, n in rows:
        if "not read" in name:
            print(f"  {name:>46}: {n / 1e9:7.3f} GB (excluded)")
            continue
        total += n
        print(f"  {name:>46}: {n / 1e9:7.3f} GB")
    print(f"  {'TOTAL read per token':>46}: {total / 1e9:7.3f} GB")
    print(
        "  (gdn state is also written back each token: "
        f"+{gdn_state_bytes / 1e9:.3f} GB writes)"
    )

    del pc

    # --- instrumented prefill (fresh cache) ---
    reset_times()
    ENABLED = True
    pc2, _, t_pref_inst = prefill(lm, ids, args.prefill_step)
    ENABLED = False
    report_split("prefill split", t_pref_inst, new_tok, "tokens")
    print(
        f"\n(instrumented prefill ran {t_pref_inst / t_prefill:.2f}x the real one;"
        " read the percentages, not the seconds)"
    )
    del pc2


if __name__ == "__main__":
    main()
