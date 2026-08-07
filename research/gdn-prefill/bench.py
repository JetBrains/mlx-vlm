"""GDN chunked-prefill share + bf16 intra-chunk matmul prototype.

Measures, at real Qwen3.6-27B shapes (q/k 16h x 128, v 48h x 128, 48 GDN
layers):
  1. gated_delta_chunked (production fp32 path) per-layer time at T=4096
     -> x48 = GDN share of a 4096-token prefill chunk
  2. a bf16-intra-chunk variant (state scan + triangular solve stay fp32)
     -> candidate speedup and rel-err vs the fp32 path
  3. chunk-size sweep C in {64, 128, 256} for both

Run on both M4 and M5 (.venv/bin/python research/gdn-prefill/bench.py).
"""

import sys
import time

import mlx.core as mx

sys.path.insert(0, ".")
from mlx_vlm.models.qwen3_5.gated_delta import (  # noqa: E402
    _invert_unit_lower,
    gated_delta_chunked,
)

B, T = 1, 4096
HK, HV, DK, DV = 16, 48, 128, 128
N_GDN_LAYERS = 48


def bf16_chunked(q, k, v, g, beta, state, C=64):
    """gated_delta_chunked with intra-chunk matmuls in bf16.

    Kept fp32: the log-cumsum decay math, the C x C triangular solve, and
    the inter-chunk state S. Cast to bf16: q/k/v operands of every big
    matmul, and the decay factors multiplying them.

    Prototype only: unlike the production path it assumes T % C == 0 (true
    for every T here) and takes no padding/mask path.
    """
    if (rf := HV // HK) > 1:
        q = mx.repeat(q, rf, -2)
        k = mx.repeat(k, rf, -2)

    pad = (C - T % C) % C
    Tp = T + pad
    nC = Tp // C

    def rc(x, D, dt):
        return x.reshape(B, nC, C, HV, D).transpose(0, 3, 1, 2, 4).astype(dt)

    qb, kb, vb = rc(q, DK, mx.bfloat16), rc(k, DK, mx.bfloat16), rc(v, DV, mx.bfloat16)
    kf32 = rc(k, DK, mx.float32)
    g = g.reshape(B, nC, C, HV).transpose(0, 3, 1, 2).astype(mx.float32)
    beta = beta.reshape(B, nC, C, HV).transpose(0, 3, 1, 2).astype(mx.float32)

    lcg = mx.cumsum(mx.log(mx.clip(g, 1e-6, 1.0)), axis=-1)
    cumg = mx.exp(lcg)
    lower_incl = mx.tril(mx.ones((C, C), mx.float32), 0)
    strict_lower = mx.tril(mx.ones((C, C), mx.float32), -1)
    diff = mx.where(lower_incl > 0, lcg[..., :, None] - lcg[..., None, :], -1e30)
    dr = mx.exp(diff)

    KK = (kb @ mx.swapaxes(kb, -1, -2)).astype(mx.float32)
    A = beta[..., :, None] * dr * KK * strict_lower
    Tinv = _invert_unit_lower(mx.eye(C, dtype=mx.float32) + A, C)  # fp32 solve

    kbeta = ((beta * cumg)[..., :, None] * kb.astype(mx.float32)).astype(mx.bfloat16)
    U0 = Tinv.astype(mx.bfloat16) @ (beta[..., :, None].astype(mx.bfloat16) * vb)
    Kt = Tinv.astype(mx.bfloat16) @ kbeta
    M = (dr.astype(mx.bfloat16) * (qb @ mx.swapaxes(kb, -1, -2))) * mx.tril(
        mx.ones((C, C), mx.bfloat16), 0
    )
    Qeff = cumg[..., :, None].astype(mx.bfloat16) * qb - M @ Kt
    MU0 = M @ U0
    cumg_last = cumg[..., -1]
    ratio_last = mx.exp(lcg[..., -1, None] - lcg)

    S = (
        state.astype(mx.float32)
        if state is not None
        else mx.zeros((B, HV, DV, DK), mx.float32)
    )
    ys = []
    for c in range(nC):
        St = mx.swapaxes(S, -1, -2)
        Stb = St.astype(mx.bfloat16)
        ys.append(MU0[:, :, c] + Qeff[:, :, c] @ Stb)
        Uc = U0[:, :, c].astype(mx.float32) - (Kt[:, :, c] @ Stb).astype(mx.float32)
        Uscaled = ratio_last[:, :, c][..., None] * Uc
        S = (
            cumg_last[:, :, c][..., None, None] * S
            + mx.swapaxes(Uscaled, -1, -2) @ kf32[:, :, c]
        )
    Y = mx.stack(ys, axis=2).transpose(0, 2, 3, 1, 4).reshape(B, Tp, HV, DV)[:, :T]
    return Y.astype(v.dtype), S


def bench(fn, *args, iters=5, **kw):
    for _ in range(2):
        mx.eval(*fn(*args, **kw))
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(*fn(*args, **kw))
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    mx.random.seed(0)
    q = mx.random.normal((B, T, HK, DK)).astype(mx.bfloat16) * 0.1
    k = mx.random.normal((B, T, HK, DK)).astype(mx.bfloat16) * 0.1
    v = mx.random.normal((B, T, HV, DV)).astype(mx.bfloat16) * 0.1
    g = mx.clip(mx.random.uniform(shape=(B, T, HV)) * 0.1 + 0.9, 0.0, 1.0)
    beta = mx.random.uniform(shape=(B, T, HV)) * 0.9 + 0.05
    mx.eval(q, k, v, g, beta)

    dev = mx.device_info()
    print(f"device: {dev['device_name']}  T={T} q/k {HK}x{DK} v {HV}x{DV}")

    y_ref, s_ref = gated_delta_chunked(q, k, v, g, beta, None, C=64)
    mx.eval(y_ref, s_ref)

    for C in (64, 128, 256):
        t_fp32 = bench(gated_delta_chunked, q, k, v, g, beta, None, C=C)
        t_bf16 = bench(bf16_chunked, q, k, v, g, beta, None, C=C)
        y, s = bf16_chunked(q, k, v, g, beta, None, C=C)
        yerr = (
            mx.abs(y.astype(mx.float32) - y_ref.astype(mx.float32)).max()
            / mx.abs(y_ref.astype(mx.float32)).max()
        ).item()
        # fp32 path at this C vs the C=64 reference (algorithmic check)
        y2, _ = gated_delta_chunked(q, k, v, g, beta, None, C=C)
        y2err = (
            mx.abs(y2.astype(mx.float32) - y_ref.astype(mx.float32)).max()
            / mx.abs(y_ref.astype(mx.float32)).max()
        ).item()
        print(
            f"C={C:3d}: fp32 {t_fp32 * 1e3:7.2f} ms (relerr {y2err:.2e}) | "
            f"bf16 {t_bf16 * 1e3:7.2f} ms (relerr {yerr:.2e}) | "
            f"bf16 speedup {t_fp32 / t_bf16:.2f}x"
        )

    t64 = bench(gated_delta_chunked, q, k, v, g, beta, None, C=64)
    print(
        f"\nGDN share estimate: {N_GDN_LAYERS} layers x {t64 * 1e3:.1f} ms = "
        f"{N_GDN_LAYERS * t64:.2f} s per {T}-token prefill chunk"
    )
    print("compare against measured end-to-end prefill rate for this machine.")


if __name__ == "__main__":
    main()
