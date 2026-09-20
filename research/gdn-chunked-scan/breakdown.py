"""Where does gated_delta_chunked spend its time on Metal? (scratch probe)

Times the phases of the chunked scan with eval barriers at T=1024 (C sweep),
and tries two variants: mx.compile of the whole function, and a blocked
(recursive) unit-lower-triangular inversion replacing the C-row forward
substitution loop.
"""

import argparse
import time

import mlx.core as mx

from mlx_vlm.models.qwen3_5.gated_delta import (
    _invert_unit_lower,
    gated_delta_chunked,
    gated_delta_kernel,
)
from bench import make_inputs, rel_err, timeit

B, HK, DK, HV, DV = 1, 16, 128, 48, 128


def chunked_phased(q, k, v, g, beta, state, C=64, tick=None):
    """Copy of gated_delta_chunked with phase barriers."""
    B_, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    in_dtype = v.dtype
    if (rf := Hv // Hk) > 1:
        q = mx.repeat(q, rf, -2)
        k = mx.repeat(k, rf, -2)

    pad = (C - T % C) % C
    Tp = T + pad
    nC = Tp // C

    def rc(x, D):
        return x.reshape(B_, nC, C, Hv, D).transpose(0, 3, 1, 2, 4).astype(mx.float32)

    q, k, v = rc(q, Dk), rc(k, Dk), rc(v, Dv)
    g = g.reshape(B_, nC, C, Hv).transpose(0, 3, 1, 2)
    beta = beta.reshape(B_, nC, C, Hv).transpose(0, 3, 1, 2)
    tick("reshape", q, k, v, g, beta)

    lcg = mx.cumsum(mx.log(mx.clip(g, 1e-6, 1.0)), axis=-1)
    cumg = mx.exp(lcg)
    lower_incl = mx.tril(mx.ones((C, C), mx.float32), 0)
    strict_lower = mx.tril(mx.ones((C, C), mx.float32), -1)
    diff = mx.where(lower_incl > 0, lcg[..., :, None] - lcg[..., None, :], -1e30)
    dr = mx.exp(diff)
    KK = k @ mx.swapaxes(k, -1, -2)
    A = beta[..., :, None] * dr * KK * strict_lower
    tick("gates+KK", A, dr, cumg)

    Tinv = _invert_unit_lower(mx.eye(C, dtype=mx.float32) + A, C)
    tick("Tinv", Tinv)

    kbeta = (beta * cumg)[..., :, None] * k
    U0 = Tinv @ (beta[..., :, None] * v)
    Kt = Tinv @ kbeta
    M = dr * (q @ mx.swapaxes(k, -1, -2)) * lower_incl
    Qeff = cumg[..., :, None] * q - M @ Kt
    MU0 = M @ U0
    cumg_last = cumg[..., -1]
    ratio_last = mx.exp(lcg[..., -1, None] - lcg)
    tick("intra GEMMs", U0, Kt, Qeff, MU0)

    S = state.astype(mx.float32)
    ys = []
    for c in range(nC):
        St = mx.swapaxes(S, -1, -2)
        ys.append(MU0[:, :, c] + Qeff[:, :, c] @ St)
        Uc = U0[:, :, c] - Kt[:, :, c] @ St
        Uscaled = ratio_last[:, :, c][..., None] * Uc
        S = (
            cumg_last[:, :, c][..., None, None] * S
            + mx.swapaxes(Uscaled, -1, -2) @ k[:, :, c]
        )
    Y = mx.stack(ys, axis=2).transpose(0, 2, 3, 1, 4).reshape(B_, Tp, Hv, Dv)[:, :T]
    Y = Y.astype(in_dtype)
    tick("inter scan", Y, S)
    return Y, S


def blocked_inv_unit_lower(L, C):
    """Recursive block inversion of unit-lower-triangular L: log2(C) rounds."""
    if C <= 8:
        return _invert_unit_lower(L, C)
    h = C // 2
    A = L[..., :h, :h]
    Bb = L[..., h:, :h]
    D = L[..., h:, h:]
    Ai = blocked_inv_unit_lower(A, h)
    Di = blocked_inv_unit_lower(D, C - h)
    lower_left = -(Di @ Bb @ Ai)
    top = mx.concatenate([Ai, mx.zeros_like(Bb.swapaxes(-1, -2))], axis=-1)
    bot = mx.concatenate([lower_left, Di], axis=-1)
    return mx.concatenate([top, bot], axis=-2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--c-values", type=int, nargs="+", default=[32, 64])
    args = ap.parse_args()
    T = 1024
    q, k, v, g, beta, state = make_inputs(T)

    # phase breakdown
    for C in args.c_values:
        from collections import defaultdict

        acc = defaultdict(float)

        def run():
            t0 = time.perf_counter()

            def tick(name, *arrays):
                nonlocal t0
                mx.eval(*arrays)
                acc[name] += time.perf_counter() - t0
                t0 = time.perf_counter()

            chunked_phased(q, k, v, g, beta, state, C=C, tick=tick)

        run()
        acc.clear()
        for _ in range(args.iters):
            run()
        total = sum(acc.values())
        print(f"\nchunked C={C} phases (mean over {args.iters}, total {1000*total/args.iters:.3f} ms):")
        for name, tsec in acc.items():
            print(f"  {name:>12}: {1000*tsec/args.iters:7.3f} ms  {100*tsec/total:5.1f}%")

    # variants
    y_ref, s_ref = gated_delta_kernel(q, k, v, g, beta, state)
    mx.eval(y_ref, s_ref)

    def run_kernel():
        y, s = gated_delta_kernel(q, k, v, g, beta, state)
        mx.eval(y, s)

    print(f"\nfused kernel: {1000*timeit(run_kernel, args.iters):.3f} ms")

    for C in args.c_values:
        # blocked inversion swapped in via monkeypatch
        import mlx_vlm.models.qwen3_5.gated_delta as gd

        orig = gd._invert_unit_lower
        gd._invert_unit_lower = lambda Tm, c: blocked_inv_unit_lower(Tm, c)
        try:
            y, s = gated_delta_chunked(q, k, v, g, beta, state, C=C)
            mx.eval(y, s)
            err = rel_err(y, y_ref)

            def run_blocked():
                y, s = gated_delta_chunked(q, k, v, g, beta, state, C=C)
                mx.eval(y, s)

            print(
                f"chunked C={C} + blocked Tinv: "
                f"{1000*timeit(run_blocked, args.iters):.3f} ms (y rel_err vs kernel {err:.1e})"
            )
        finally:
            gd._invert_unit_lower = orig

        # mx.compile whole function (with blocked inv too)
        gd._invert_unit_lower = lambda Tm, c: blocked_inv_unit_lower(Tm, c)
        try:
            compiled = mx.compile(
                lambda q, k, v, g, beta, state: gated_delta_chunked(
                    q, k, v, g, beta, state, C=C
                )
            )
            y, s = compiled(q, k, v, g, beta, state)
            mx.eval(y, s)
            err = rel_err(y, y_ref)

            def run_compiled():
                y, s = compiled(q, k, v, g, beta, state)
                mx.eval(y, s)

            print(
                f"chunked C={C} + blocked Tinv + mx.compile: "
                f"{1000*timeit(run_compiled, args.iters):.3f} ms (y rel_err vs kernel {err:.1e})"
            )
        finally:
            gd._invert_unit_lower = orig


if __name__ == "__main__":
    main()
