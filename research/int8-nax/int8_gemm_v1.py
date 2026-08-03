"""W8A8 int8 GEMM prototype on M5 NAX via mx.fast.metal_kernel + MPP tensor ops.

Milestone 1 of research/int8-nax/README.md §7: standalone int8 GEMM at the
Qwen3.6-27B MLP shapes, validated against an exact int32 reference and
benchmarked against MLX's bf16 NAX GEMM and 4-bit qmm.

Y[M,N] = (Xq[M,K] @ Wq[N,K].T) * xs[M] ⊗ ws[N]
  Xq, Wq int8 (symmetric per-row / per-output-channel), accumulation int32,
  scales applied in-register via the cooperative destination tensor.

v1 restrictions: M % TM == 0, N % TN == 0 (fine for benchmark shapes).
"""

import time

import mlx.core as mx

TM, TN = 64, 32  # threadgroup output tile (matmul2d descriptor m, n)
NSIMD = 4        # simdgroups per threadgroup -> 128 threads

HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
using namespace metal;
"""

SRC_TMPL = """
    constexpr int M = {M};
    constexpr int N = {N};
    constexpr int K = {K};
    constexpr int TM = {TM};
    constexpr int TN = {TN};

    uint2 tgid = threadgroup_position_in_grid.xy;

    constexpr auto desc = matmul2d_descriptor(
        TM, TN, static_cast<int>(dynamic_extent),
        /*transpose_left=*/false, /*transpose_right=*/true,
        /*relaxed_precision=*/false,
        matmul2d_descriptor::mode::multiply_accumulate);

    matmul2d<desc, execution_simdgroups<{NSIMD}>> op;

    // Row-major X[M,K]: extent(0)=K (contiguous), extent(1)=M.
    // Row-major W[N,K] used as transposed right operand: extent(0)=K, extent(1)=N.
    auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        (device int8_t*)xq, dextents<int32_t, 2>(K, M));
    auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        (device int8_t*)wq, dextents<int32_t, 2>(K, N));

    auto tA = A.slice(0, int(tgid.y) * TM);
    auto tB = B.slice(0, int(tgid.x) * TN);

    auto cT = op.get_destination_cooperative_tensor<
        decltype(tA), decltype(tB), int32_t>();

#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        if (cT.is_valid_element(i)) cT[i] = 0;
    }}

    op.run(tA, tB, cT);

#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        if (cT.is_valid_element(i)) {{
            auto idx = cT.get_multidimensional_index(i);
            int n = int(tgid.x) * TN + idx[0];
            int m = int(tgid.y) * TM + idx[1];
            out[m * N + n] = bfloat(float(cT[i]) * xs[m] * ws[n]);
        }}
    }}
"""

_kernel_cache = {}


def int8_gemm(xq, xs, wq, ws):
    M, K = xq.shape
    N = wq.shape[0]
    assert M % TM == 0 and N % TN == 0, (M, N)
    key = (M, N, K)
    if key not in _kernel_cache:
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=f"int8_gemm_{M}x{N}x{K}",
            input_names=["xq", "wq", "xs", "ws"],
            output_names=["out"],
            header=HEADER,
            source=SRC_TMPL.format(M=M, N=N, K=K, TM=TM, TN=TN, NSIMD=NSIMD),
        )
    k = _kernel_cache[key]
    return k(
        inputs=[xq, wq, xs, ws],
        grid=(N // TN * 32 * NSIMD, M // TM, 1),
        threadgroup=(32 * NSIMD, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[mx.bfloat16],
    )[0]


def quantize_per_row(a):
    """Symmetric per-row int8: returns (int8 values, fp32 scales)."""
    s = mx.maximum(mx.abs(a).max(axis=1), 1e-8).astype(mx.float32) / 127.0
    q = mx.clip(mx.round(a.astype(mx.float32) / s[:, None]), -127, 127).astype(
        mx.int8
    )
    return q, s


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())  # eval every iteration: MLX is lazy (see README §2)
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    mx.random.seed(0)
    K, N = 5120, 17408  # Qwen3.6-27B MLP up/gate shape

    for M in [512, 2048, 8192]:
        x = mx.random.normal((M, K)).astype(mx.bfloat16)
        w = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
        mx.eval(x, w)

        xq, xs = quantize_per_row(x)
        wq, ws = quantize_per_row(w)
        mx.eval(xq, xs, wq, ws)

        # exact integer reference
        ref_i32 = xq.astype(mx.float32) @ wq.astype(mx.float32).T
        ref = (ref_i32 * xs[:, None] * ws[None, :]).astype(mx.bfloat16)
        got = int8_gemm(xq, xs, wq, ws)
        mx.eval(ref, got)
        err = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max()
        rel = err / mx.abs(ref.astype(mx.float32)).max()
        status = "OK" if rel.item() < 1e-2 else "MISMATCH"
        print(f"M={M:5d} correctness vs int32 reference: max abs err "
              f"{err.item():.5f} (rel {rel.item():.2e}) {status}")
        if status != "OK":
            continue

        # end-to-end quantization error vs the bf16 matmul (context only)
        full = x @ w.T
        qerr = (mx.abs(got.astype(mx.float32) - full.astype(mx.float32)).mean()
                / mx.abs(full.astype(mx.float32)).mean())
        print(f"        w8a8-vs-bf16 mean rel err: {qerr.item():.4f}")

        flops = 2.0 * M * K * N
        t_i8 = bench(lambda: int8_gemm(xq, xs, wq, ws))
        t_bf = bench(lambda: x @ w.T)

        q = dict(bits=4, group_size=64, mode="affine")
        w4, s4, b4 = mx.quantize(w, **q)
        mx.eval(w4, s4, b4)
        t_q4 = bench(
            lambda: mx.quantized_matmul(x, w4, s4, b4, transpose=True, **q))

        t_quant = bench(lambda: quantize_per_row(x)[0])

        print(f"        int8 gemm : {t_i8*1e3:8.3f} ms  {flops/t_i8/1e12:6.1f} TOPS-eq")
        print(f"        bf16 gemm : {t_bf*1e3:8.3f} ms  {flops/t_bf/1e12:6.1f} TF")
        print(f"        qmm 4bit  : {t_q4*1e3:8.3f} ms  {flops/t_q4/1e12:6.1f} TF-eq")
        print(f"        act-quant : {t_quant*1e3:8.3f} ms (per-call overhead, unfused)")
        print(f"        speedup vs bf16: {t_bf/t_i8:.2f}x   vs qmm4: {t_q4/t_i8:.2f}x")
        print()


if __name__ == "__main__":
    main()
