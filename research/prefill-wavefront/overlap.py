"""Can Metal overlap the GDN delta-scan kernel with big GEMMs? (micro test)

The delta scan runs ~200k threads for ~1.7ms (low occupancy on an M5 Max);
prefill GEMMs can in principle fill the idle cores IF MLX streams map to
concurrent Metal execution. This test times, at Qwen3.6 shapes:

  A. N delta scans alone (one stream)
  B. M MLP-sized GEMMs alone (one stream)
  C. both interleaved on ONE stream (baseline: serialized)
  D. both on TWO streams (candidate: overlapped)

If D ~= max(A, B) instead of A + B, wavefront pipelining has headroom.

Usage: .venv/bin/python research/prefill-wavefront/overlap.py
"""

import time

import mlx.core as mx

from mlx_vlm.models.qwen3_5.gated_delta import gated_delta_kernel

B, HK, DK, HV, DV = 1, 16, 128, 48, 128
T = 1024
N_SCANS = 12  # 48 GDN layers / 4 (only some would overlap at once)
M_GEMMS = 12

# MLP GEMM at prefill shapes: [1024, 5120] x [5120, 17408] bf16
GM, GK, GN = 1024, 5120, 17408


def make_scan():
    q = mx.random.normal((B, T, HK, DK)).astype(mx.bfloat16) * DK**-0.5
    k = mx.random.normal((B, T, HK, DK)).astype(mx.bfloat16) * DK**-0.5
    v = mx.random.normal((B, T, HV, DV)).astype(mx.bfloat16)
    g = mx.random.uniform(low=0.9, high=1.0, shape=(B, T, HV)).astype(mx.float32)
    beta = mx.random.uniform(shape=(B, T, HV)).astype(mx.float32)
    state = mx.zeros((B, HV, DV, DK), dtype=mx.float32)
    mx.eval(q, k, v, g, beta, state)
    return q, k, v, g, beta, state


def timeit(fn, iters=10):
    fn()
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main():
    scan_in = make_scan()
    x = mx.random.normal((GM, GK)).astype(mx.bfloat16)
    w = mx.random.normal((GN, GK)).astype(mx.bfloat16)
    mx.eval(x, w)

    s_default = mx.default_stream(mx.gpu)
    s2 = mx.new_stream(mx.gpu)

    def scans(stream, n=N_SCANS):
        outs = []
        with mx.stream(stream):
            st = scan_in[5]
            for _ in range(n):
                y, st = gated_delta_kernel(*scan_in[:5], st)
                outs.append(y)
        return outs, st

    def gemms(stream, m=M_GEMMS):
        with mx.stream(stream):
            acc = x
            for _ in range(m):
                acc = (acc @ w.T)[:, :GK] * 1e-3  # keep shapes stable, chained
        return acc

    def run_a():
        outs, st = scans(s_default)
        mx.eval(st)

    def run_b():
        acc = gemms(s_default)
        mx.eval(acc)

    def run_c():
        outs, st = scans(s_default)
        acc = gemms(s_default)
        mx.eval(st, acc)

    def run_d():
        outs, st = scans(s_default)
        acc = gemms(s2)
        mx.eval(st, acc)

    ta = timeit(run_a)
    tb = timeit(run_b)
    tc = timeit(run_c)
    td = timeit(run_d)
    print(f"A scans alone ({N_SCANS}x):            {1000 * ta:7.2f} ms")
    print(f"B gemms alone ({M_GEMMS}x):            {1000 * tb:7.2f} ms")
    print(f"C both, one stream (serial):     {1000 * tc:7.2f} ms  (A+B={1000 * (ta + tb):.2f})")
    print(f"D both, two streams (overlap?):  {1000 * td:7.2f} ms  (max(A,B)={1000 * max(ta, tb):.2f})")
    saved = tc - td
    print(f"overlap recovered {1000 * saved:.2f} ms = {100 * saved / tc:.0f}% of serial time")


if __name__ == "__main__":
    main()
