"""W8A8 int8 GEMM for Apple M5 neural accelerators (NAX) via MPP tensor ops.

Building blocks for selective int8 prefill (research/int8-nax/README.md §7):

- quantize_rows(x):      bf16/f32 [M,K] -> (int8 [M,K], f32 [M]) per-row symmetric
- quantize_weight(w):    like quantize_rows, for [N,K] weights (per-output-channel)
- int8_gemm(xq,xs,wq,ws[,bias]): bf16 [M,N] = (xq @ wq.T) * xs⊗ws (+ bias)

Requires: M5-class GPU (Metal 4 tensor ops / MetalPerformancePrimitives),
mlx >= 0.32. GEMM tile config chosen by sweep (int8_gemm_sweep.py):
TM=128, TN=128, 8 simdgroups, internal K loop -> ~91 TOPS-eq at MLP shapes
(1.55x MLX's bf16 NAX GEMM, which runs ~58 TF at the same shape).

Arbitrary M is supported (edge tiles bounds-checked); K and N must be
multiples of 32 (holds for all transformer projections in practice).
"""

import mlx.core as mx

_HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
using namespace metal;
"""

# One threadgroup (256 threads) per row: pass 1 absmax-reduce, pass 2 quantize.
_QUANT_SRC = """
    constexpr int K = {K};
    constexpr int NTH = 256;

    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = tid % 32;
    uint sg = tid / 32;

    const device {T}* xrow = x + size_t(row) * K;

    float amax = 0.0f;
    for (int i = tid; i < K; i += NTH) {{
        amax = max(amax, fabs(float(xrow[i])));
    }}
    amax = simd_max(amax);

    threadgroup float tg_max[NTH / 32];
    if (lane == 0) tg_max[sg] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = tg_max[lane % (NTH / 32)];
    amax = simd_max(amax);

    float scale = max(amax, 1e-8f) / 127.0f;
    float inv = 1.0f / scale;
    if (tid == 0) xs[row] = scale;

    device int8_t* qrow = xq + size_t(row) * K;
    for (int i = tid; i < K; i += NTH) {{
        qrow[i] = int8_t(clamp(rint(float(xrow[i]) * inv), -127.0f, 127.0f));
    }}
"""

_GEMM_SRC = """
    constexpr int N = {N};
    constexpr int K = {K};
    constexpr int TM = 128;
    constexpr int TN = 128;

    uint2 tgid = threadgroup_position_in_grid.xy;
    const int M = m_dim[0];

    constexpr auto desc = matmul2d_descriptor(
        TM, TN, static_cast<int>(dynamic_extent),
        /*transpose_left=*/false, /*transpose_right=*/true,
        /*relaxed_precision=*/true,
        matmul2d_descriptor::mode::multiply_accumulate);

    matmul2d<desc, execution_simdgroups<8>> op;

    // Row-major X[M,K] -> extents (K, M); row-major W[N,K] as transposed
    // right operand -> extents (K, N).
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
            if (m < M && n < N) {{
                float v = float(cT[i]) * xs[m] * ws[n];
                {BIAS_LINE}
                out[size_t(m) * N + n] = bfloat(v);
            }}
        }}
    }}
"""

_TM, _TN, _NSIMD = 128, 128, 8
_quant_cache = {}
_gemm_cache = {}


def quantize_rows(x):
    """Per-row symmetric int8 quantization. x: [M, K] bf16/f16/f32."""
    M, K = x.shape
    tname = {mx.bfloat16: "bfloat", mx.float16: "half", mx.float32: "float"}[
        x.dtype
    ]
    key = (K, tname)
    if key not in _quant_cache:
        _quant_cache[key] = mx.fast.metal_kernel(
            name=f"rowquant_{K}_{tname}",
            input_names=["x"],
            output_names=["xq", "xs"],
            header=_HEADER,
            source=_QUANT_SRC.format(K=K, T=tname),
        )
    k = _quant_cache[key]
    xq, xs = k(
        inputs=[x],
        grid=(M * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(M, K), (M,)],
        output_dtypes=[mx.int8, mx.float32],
    )
    return xq, xs


def quantize_weight(w):
    """Per-output-channel symmetric int8 for a [N, K] weight (offline)."""
    s = mx.maximum(mx.abs(w).max(axis=1), 1e-8).astype(mx.float32) / 127.0
    q = mx.clip(
        mx.round(w.astype(mx.float32) / s[:, None]), -127, 127
    ).astype(mx.int8)
    return q, s


def int8_gemm(xq, xs, wq, ws, bias=None):
    """bf16 [M,N] = (xq[M,K] @ wq[N,K].T) * xs[M] ⊗ ws[N] (+ bias[N])."""
    M, K = xq.shape
    N = wq.shape[0]
    assert K % 32 == 0 and N % _TN == 0, (K, N)
    key = (N, K, bias is not None)
    if key not in _gemm_cache:
        names = ["xq", "wq", "xs", "ws", "m_dim"]
        bias_line = ""
        if bias is not None:
            names.append("bias")
            bias_line = "v += float(bias[n]);"
        _gemm_cache[key] = mx.fast.metal_kernel(
            name=f"i8gemm_{N}x{K}{'_b' if bias is not None else ''}",
            input_names=names,
            output_names=["out"],
            header=_HEADER,
            source=_GEMM_SRC.format(N=N, K=K, BIAS_LINE=bias_line),
        )
    k = _gemm_cache[key]
    m_dim = mx.array([M], dtype=mx.int32)
    inputs = [xq, wq, xs, ws, m_dim]
    if bias is not None:
        inputs.append(bias)
    ntg_m = (M + _TM - 1) // _TM
    return k(
        inputs=inputs,
        grid=(N // _TN * 32 * _NSIMD, ntg_m, 1),
        threadgroup=(32 * _NSIMD, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[mx.bfloat16],
    )[0]


def w8a8_linear(x, wq, ws, bias=None):
    """Drop-in matmul: bf16 x [.., K] @ int8 weight [N, K].T -> bf16 [.., N]."""
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1])
    xq, xs = quantize_rows(x2)
    y = int8_gemm(xq, xs, wq, ws, bias=bias)
    return y.reshape(*orig_shape[:-1], wq.shape[0])
