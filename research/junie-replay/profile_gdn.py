"""Phase-level time profile inside Qwen3_5GatedDeltaNet for a Junie request.

Companion to profile_layers.py (same model setup: int8 prefill patch, step
1024, no MTP). Replaces the GDN __call__ with a copy that puts mx.eval
barriers between its phases and accumulates per-phase time, separately for
prefill (S>1) and decode (S==1):

  in_proj   - in_proj_qkv/z/b/a GEMMs (int8 path in prefill)
  conv      - conv-state concat/cache update + depthwise conv1d + silu + split
  qk_norm   - rms_norm on q and k
  delta     - gated_delta_update (the fused sequential-scan metal kernel)
  gate_norm - Qwen3_5RMSNormGated(out, z)
  out_proj  - output GEMM
  misc      - remainder of the call (reshapes, cache bookkeeping)

Barriers serialize the GPU at each phase edge, so decode percentages carry
some fixed per-barrier cost on the small phases; prefill (1024-token chunks)
is barely distorted.

Usage:
  .venv/bin/python research/junie-replay/profile_gdn.py \
      [--req research/junie-replay/requests/10.json] [--decode-tokens 100]
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
from mlx_vlm.models.qwen3_5.gated_delta import gated_delta_update
from mlx_vlm.utils import load

from profile_layers import MODEL_DIR, build_prompt_ids, decode_plain, prefill

PHASES = {
    "prefill": defaultdict(float),
    "decode": defaultdict(float),
}
CALLS = {"prefill": 0, "decode": 0}
ENABLED = False

PHASE_ORDER = ["in_proj", "conv", "qk_norm", "delta", "gate_norm", "out_proj", "misc"]


def install_gdn_profiler():
    orig = q35.Qwen3_5GatedDeltaNet.__call__

    def timed(self, inputs, mask=None, cache=None):
        if not ENABLED:
            return orig(self, inputs, mask=mask, cache=cache)

        B, S, _ = inputs.shape
        mode = "prefill" if S > 1 else "decode"
        acc = PHASES[mode]
        CALLS[mode] += 1
        # Drain everything queued by the preceding layers so the first tick
        # measures only GDN work.
        mx.eval(inputs)
        t_start = time.perf_counter()
        t0 = t_start
        ticked = 0.0

        def tick(name, *arrays):
            nonlocal t0, ticked
            mx.eval(*arrays)
            dt = time.perf_counter() - t0
            acc[name] += dt
            ticked += dt
            t0 = time.perf_counter()

        mixed_qkv, z, b, a = (
            self.in_proj_qkv(inputs),
            self.in_proj_z(inputs),
            self.in_proj_b(inputs),
            self.in_proj_a(inputs),
        )
        tick("in_proj", mixed_qkv, z, b, a)

        z = z.reshape(B, S, -1, self.head_v_dim)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
            if conv_state.shape[0] != B:
                conv_state = mx.zeros(
                    (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
                )
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
            )

        if mask is not None:
            if mask.shape[0] != B:
                mask = None
            else:
                mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if getattr(cache, "lengths", None) is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        if (
            S == 1
            and conv_input.shape[1] == self.conv_kernel_size
            and self.conv1d.weight.dtype in (mx.bfloat16, mx.float16)
        ):
            conv_out = nn.silu(self._causal_conv1d_decode(conv_input))
        else:
            conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        tick("conv", q, k, v, cache[0] if cache is not None else conv_out)

        state = cache[1] if cache else None
        if state is not None and state.shape[0] != B:
            state = None
        q, k = self._normalize_qk(q, k)
        tick("qk_norm", q, k)

        out, state = gated_delta_update(
            q, k, v, a, b, self.A_log, self.dt_bias, state, mask,
            use_kernel=not self.training,
        )
        tick("delta", out, state)

        if cache is not None:
            cache[1] = state
            if hasattr(cache, "advance"):
                cache.advance(S)
                q35._qwen3_5_advance_left_padding_info(cache, S)
                q35._qwen3_5_advance_lengths_info(cache, S)

        out = self.norm(out, z)
        tick("gate_norm", out)

        result = self.out_proj(out.reshape(B, S, -1))
        tick("out_proj", result)

        acc["misc"] += time.perf_counter() - t_start - ticked
        return result

    q35.Qwen3_5GatedDeltaNet.__call__ = timed


def report(mode, n_units, unit):
    acc = PHASES[mode]
    total = sum(acc.values())
    if not total:
        return
    print(f"\nGDN {mode} phases ({CALLS[mode]} calls, {n_units} {unit}):")
    for k in PHASE_ORDER:
        if acc.get(k):
            print(f"  {k:>9}: {acc[k]:7.2f}s  {100 * acc[k] / total:5.1f}%")
    print(f"  {'total':>9}: {total:7.2f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--req",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "requests", "10.json"
        ),
    )
    ap.add_argument("--decode-tokens", type=int, default=100)
    ap.add_argument("--prefill-step", type=int, default=1024)
    ap.add_argument("--max-prompt", type=int, default=0)
    args = ap.parse_args()

    print(f"loading {MODEL_DIR}")
    model, processor = load(MODEL_DIR)
    lm = model.language_model
    apply_int8_prefill()

    ids = build_prompt_ids(processor, args.req)
    if args.max_prompt:
        ids = ids[: args.max_prompt]
    print(f"prompt tokens: {len(ids)}")

    install_gdn_profiler()

    global ENABLED
    ENABLED = False
    # warmup: int8 copies, kernel JIT
    wpc, wfirst, _ = prefill(lm, ids[: min(len(ids), 2048)], args.prefill_step)
    decode_plain(lm, wpc, wfirst, 4)
    del wpc
    print("warmup done")

    ENABLED = True
    pc, first, t_prefill = prefill(lm, ids, args.prefill_step)
    ENABLED = False
    print(f"prefill wall (with GDN barriers): {t_prefill:.2f}s")
    report("prefill", len(ids), "tokens")

    ENABLED = True
    t_dec = decode_plain(lm, pc, first, args.decode_tokens)
    ENABLED = False
    print(f"\ndecode wall (with GDN barriers): {t_dec:.2f}s "
          f"({args.decode_tokens} tokens)")
    report("decode", args.decode_tokens, "tokens")


if __name__ == "__main__":
    main()
