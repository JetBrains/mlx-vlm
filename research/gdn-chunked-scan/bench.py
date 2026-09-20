"""Microbench: fused sequential delta-scan kernel vs chunked parallel scan on Metal.

Motivation (see research/junie-replay/profile_gdn.py, M5 Max, junie req 10):
the fused gated-delta kernel is 29% of GDN prefill time (~9% of total
prefill, ~2.8s) because it loops sequentially over all T tokens of a chunk
inside one launch. gated_delta_chunked already exists (used on CUDA) and
replaces the T-loop with parallel intra-chunk GEMMs + a T/C inter-chunk scan.
This bench asks whether it beats the fused kernel on Apple Metal too.

Runs at the exact Qwen3.6-27B GDN shapes (Hk=16, Dk=128, Hv=48, Dv=128,
B=1), checks numerics of both against the fp32 ops reference, then times
kernel vs chunked over a sweep of T and chunk size C.

Usage:
  .venv/bin/python research/gdn-chunked-scan/bench.py [--iters 30]
"""

import argparse
import time

import mlx.core as mx

from mlx_vlm.models.qwen3_5.gated_delta import (
    _compute_g_beta,
    gated_delta_chunked,
    gated_delta_kernel,
    gated_delta_ops,
)

B, HK, DK, HV, DV = 1, 16, 128, 48, 128


def make_inputs(T, seed=0):
    mx.random.seed(seed)
    inv = DK**-0.5
    q = (inv**2) * mx.fast.rms_norm(
        mx.random.normal((B, T, HK, DK)), None, 1e-6
    ).astype(mx.bfloat16)
    k = inv * mx.fast.rms_norm(mx.random.normal((B, T, HK, DK)), None, 1e-6).astype(
        mx.bfloat16
    )
    v = mx.random.normal((B, T, HV, DV)).astype(mx.bfloat16)
    a = mx.random.normal((B, T, HV)).astype(mx.bfloat16)
    b = mx.random.normal((B, T, HV)).astype(mx.bfloat16)
    A_log = mx.log(mx.random.uniform(low=0.5, high=16, shape=(HV,)))
    dt_bias = mx.ones(HV)
    g, beta = _compute_g_beta(A_log, a, b, dt_bias)
    state = 0.1 * mx.random.normal((B, HV, DV, DK)).astype(mx.float32)
    mx.eval(q, k, v, g, beta, state)
    return q, k, v, g, beta, state


def rel_err(x, ref):
    x = x.astype(mx.float32)
    ref = ref.astype(mx.float32)
    return (mx.abs(x - ref).max() / (mx.abs(ref).max() + 1e-9)).item()


def timeit(fn, iters):
    fn()  # warmup (kernel JIT, shape caches)
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--t-values", type=int, nargs="+", default=[1024])
    ap.add_argument("--c-values", type=int, nargs="+", default=[16, 32, 64, 128])
    args = ap.parse_args()

    # --- numerics at T=256 against the fp32 ops reference ---
    q, k, v, g, beta, state = make_inputs(256)
    y_ref, s_ref = gated_delta_ops(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        g, beta, state,
    )
    y_k, s_k = gated_delta_kernel(q, k, v, g, beta, state)
    mx.eval(y_ref, s_ref, y_k, s_k)
    print(f"kernel  vs ops ref: y rel_err={rel_err(y_k, y_ref):.2e} "
          f"state rel_err={rel_err(s_k, s_ref):.2e}")
    for C in args.c_values:
        y_c, s_c = gated_delta_chunked(q, k, v, g, beta, state, C=C)
        mx.eval(y_c, s_c)
        print(f"chunked C={C:>3} vs ops ref: y rel_err={rel_err(y_c, y_ref):.2e} "
              f"state rel_err={rel_err(s_c, s_ref):.2e}")

    # --- speed ---
    for T in args.t_values:
        q, k, v, g, beta, state = make_inputs(T)

        def run_kernel():
            y, s = gated_delta_kernel(q, k, v, g, beta, state)
            mx.eval(y, s)

        t_k = timeit(run_kernel, args.iters)
        print(f"\nT={T}: fused kernel {1000 * t_k:8.3f} ms")
        for C in args.c_values:

            def run_chunked():
                y, s = gated_delta_chunked(q, k, v, g, beta, state, C=C)
                mx.eval(y, s)

            t_c = timeit(run_chunked, args.iters)
            print(
                f"T={T}: chunked C={C:>3} {1000 * t_c:8.3f} ms "
                f"({t_k / t_c:5.2f}x vs kernel)"
            )


if __name__ == "__main__":
    main()
