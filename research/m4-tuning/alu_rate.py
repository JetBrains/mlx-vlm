"""Raw M4 GPU ALU rates: fp32/fp16 FMA and simdgroup_matrix throughput.

Establishes the true prefill compute ceiling on M4 (no NAX). The question:
does fp16 run at 2x the fp32 ALU rate (as on A-series), and do
half-element simdgroup matrices beat float ones? MSL's
simdgroup_multiply_accumulate requires one element type for all operands,
so MLX's steel GEMM (which accumulates in fp32) runs its math on float
simdgroup matrices even for half/bf16 inputs — if half8x8 MMA is 2x, a
half-accumulate GEMM path would raise the ceiling above what MLX reaches.

Methodology follows research/int8-nax/mma_rate.py: register-resident
unrolled loops, independent accumulators to hide FMA latency, accumulators
consumed into the output, mx.eval inside the timing loop.

Caveat: trust the sgmat_* rows, not the fma_* ones. Re-run on M5 Max
(2026-08-07) the fma loops report 97-121 TFLOPS, which is above that GPU's
plausible FMA peak — the compiler still folds part of the cross-coupled
chain there. The M4 conclusion rests on the simdgroup rows, which land in a
sane range on both chips.
"""

import time

import mlx.core as mx

N_TG = 2048
N_TH = 256
OUTER = 64
UNROLL = 64
N_ACC = 8


def _run(name, src, flops_per_thread, iters=20):
    k = mx.fast.metal_kernel(
        name=name, input_names=["seed"], output_names=["out"], source=src
    )

    def call():
        return k(
            inputs=[mx.array([1.0], dtype=mx.float32)],
            grid=(N_TG * N_TH, 1, 1),
            threadgroup=(N_TH, 1, 1),
            output_shapes=[(N_TG * N_TH,)],
            output_dtypes=[mx.float32],
        )[0]

    mx.eval(call())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(call())
    mx.synchronize()
    dt = (time.perf_counter() - t0) / iters
    total = flops_per_thread * N_TG * N_TH
    print(f"{name:22s} {dt * 1e3:7.2f} ms  {total / dt / 1e12:6.2f} TFLOPS")


def fma_src(t, width=1):
    vt = t if width == 1 else f"{t}{width}"
    accs = "\n".join(
        f"    {vt} a{i} = {vt}({t}(seed[0]) * {t}({0.125 * (i + 1)}f));"
        for i in range(N_ACC)
    )
    # multiplicatively cross-coupled accumulators: fma(a_i, a_j, x) is a
    # nonlinear recurrence with no closed form, so fast-math cannot collapse
    # the unrolled chain (linear variants of this bench inflated 4-5x).
    body = "\n".join(
        f"        a{i} = fma(a{i}, a{(i + 1) % N_ACC}, x);" for i in range(N_ACC)
    )
    reduce = " + ".join(
        f"a{i}" if width == 1 else f"a{i}.x + a{i}.y" for i in range(N_ACC)
    )
    return f"""
    uint gid = thread_position_in_grid.x;
{accs}
    {vt} x = {vt}({t}(0.999f));
    for (int o = 0; o < {OUTER}; ++o) {{
#pragma unroll
      for (int u = 0; u < {UNROLL}; ++u) {{
{body}
      }}
    }}
    out[gid] = float({reduce});
"""


def sg_src(t):
    accs = "\n".join(
        f"    simdgroup_matrix<{t}, 8, 8> c{i} = "
        f"make_filled_simdgroup_matrix<{t}, 8, 8>({t}(0));"
        for i in range(N_ACC)
    )
    body = "\n".join(
        f"        simdgroup_multiply_accumulate(c{i}, a, b, c{i});"
        for i in range(N_ACC)
    )
    reduces = "\n".join(
        f"    simdgroup_store(c{i}, buf, 8); "
        f"threadgroup_barrier(mem_flags::mem_threadgroup); "
        f"s += float(buf[lane]);"
        for i in range(N_ACC)
    )
    return f"""
    uint gid = thread_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x % 32;
    simdgroup_matrix<{t}, 8, 8> a =
        make_filled_simdgroup_matrix<{t}, 8, 8>({t}(seed[0]) * {t}(0.01f));
    simdgroup_matrix<{t}, 8, 8> b =
        make_filled_simdgroup_matrix<{t}, 8, 8>({t}(0.999f));
{accs}
    for (int o = 0; o < {OUTER}; ++o) {{
#pragma unroll
      for (int u = 0; u < {UNROLL // 8}; ++u) {{
{body}
      }}
    }}
    threadgroup {t} buf[64];
    float s = 0.0f;
{reduces}
    out[gid] = s;
"""


print(f"device: {mx.device_info()['device_name']}")
print(f"{N_TG} tgs x {N_TH} threads, {N_ACC} accumulators\n")

fma_flops = 2 * OUTER * UNROLL * N_ACC
_run("fma_fp32", fma_src("float"), fma_flops)
_run("fma_fp16", fma_src("half"), fma_flops)
_run("fma_fp16x2", fma_src("half", 2), fma_flops * 2)

# one 8x8x8 simdgroup MMA = 1024 FLOP per simdgroup of 32 lanes
sg_flops = 2 * 8 * 8 * 8 * OUTER * (UNROLL // 8) * N_ACC // 32
_run("sgmat_f32", sg_src("float"), sg_flops)
_run("sgmat_f16", sg_src("half"), sg_flops)
_run("sgmat_bf16", sg_src("bfloat"), sg_flops)
