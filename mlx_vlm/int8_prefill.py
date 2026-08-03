"""Selective W8A8 int8 prefill on Apple M5 neural accelerators (NAX).

For prefill-sized calls on the large language-model projections (MLP and,
with the default "all" scope, the attention/linear-attention projections),
replaces the 4-bit quantized matmul with an int8 x int8 -> int32 GEMM running
on the M5 GPU neural accelerators via Metal Performance Primitives tensor ops:

  - activations: per-token (per-row) dynamic symmetric int8, custom kernel
  - weights: per-output-channel symmetric int8, derived lazily (once per
    module) by dequantizing the resident 4-bit weights; kept alongside them
    (~17 GB extra for Qwen3.6-27B, fine on a 128 GB machine)
  - accumulation int32, scales applied in-register, bf16 output

Decode-sized calls (rows < ROW_THRESHOLD) keep the 4-bit quantized kernels,
so decode speed and numerics are completely unchanged. Attention, lm_head,
embeddings and the vision tower are untouched.

Measured on M5 Max (research/int8-nax/): int8 GEMM ~91 TOPS-eq vs ~58 TF for
MLX's bf16 NAX GEMM at the MLP shapes; fused (quant + GEMM) 1.49x over bf16
and 1.66x over 4-bit qmm at M=2048.

Requires an M5-class GPU (Metal 4 tensor ops). Usage: call apply() any time
before serving traffic (weight init is lazy, or call warmup(model) to
pre-build). The server applies it at startup with --int8-prefill
(MLX_VLM_INT8_PREFILL=1).
"""

import logging
import os
from collections import OrderedDict

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# Only calls with at least this many rows (tokens) take the int8 path; below
# it the 4-bit qmm kernels win (they are ~2x faster than bf16 at decode
# sizes). This is also what keeps generation on the current kernels: decode
# calls have 1..O(draft block) rows, far below the threshold.
ROW_THRESHOLD = 512

# Scope of layers routed to W8A8 (env MLX_VLM_INT8_SCOPE):
#   "all" (default): every large language-model projection — MLP plus
#       attention/linear-attention (q/k/v/o, in_proj_qkv/z, out_proj).
#   "mlp": only the MLP projections (gate/up 5120->17408, down 17408->5120),
#       the more conservative choice if a quality eval flags "all".
# Either way lm_head is excluded (N > MAX_OUT) and tiny projections such as
# linear_attn.in_proj_a/b (N=48) fail the N % 128 tile requirement.
SCOPE = os.environ.get("MLX_VLM_INT8_SCOPE", "all")
MLP_SHAPES = {(17408, 5120), (5120, 17408)}
MAX_OUT = 32768
MIN_DIM = 1024

_HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
using namespace metal;
"""

# One threadgroup (256 threads) per row: absmax reduce, then quantize.
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

# Threadgroup computes a 128x128 output tile with 8 simdgroups; matmul2d
# loops over K internally (dynamic_extent). Edge tiles are bounds-checked by
# the tensor extents and the epilogue guard, so any M works.
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

    // Row-major X[M,K] -> extents (K, M); row-major W[N,K] used as the
    // transposed right operand -> extents (K, N).
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
_quant_kernels = {}
_gemm_kernels = {}
# id(module) -> (wq int8 [N,K], ws fp32 [N]); modules live for server lifetime
_int8_weights = {}


def _quantize_rows(x):
    M, K = x.shape
    tname = {mx.bfloat16: "bfloat", mx.float16: "half", mx.float32: "float"}[
        x.dtype
    ]
    key = (K, tname)
    if key not in _quant_kernels:
        _quant_kernels[key] = mx.fast.metal_kernel(
            name=f"i8p_rowquant_{K}_{tname}",
            input_names=["x"],
            output_names=["xq", "xs"],
            header=_HEADER,
            source=_QUANT_SRC.format(K=K, T=tname),
        )
    return _quant_kernels[key](
        inputs=[x],
        grid=(M * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(M, K), (M,)],
        output_dtypes=[mx.int8, mx.float32],
    )


def _int8_gemm(xq, xs, wq, ws, bias=None):
    M, K = xq.shape
    N = wq.shape[0]
    key = (N, K, bias is not None)
    if key not in _gemm_kernels:
        names = ["xq", "wq", "xs", "ws", "m_dim"]
        bias_line = ""
        if bias is not None:
            names.append("bias")
            bias_line = "v += float(bias[n]);"
        _gemm_kernels[key] = mx.fast.metal_kernel(
            name=f"i8p_gemm_{N}x{K}{'_b' if bias is not None else ''}",
            input_names=names,
            output_names=["out"],
            header=_HEADER,
            source=_GEMM_SRC.format(N=N, K=K, BIAS_LINE=bias_line),
        )
    inputs = [xq, wq, xs, ws, mx.array([M], dtype=mx.int32)]
    if bias is not None:
        inputs.append(bias)
    return _gemm_kernels[key](
        inputs=inputs,
        grid=(N // _TN * 32 * _NSIMD, (M + _TM - 1) // _TM, 1),
        threadgroup=(32 * _NSIMD, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[mx.bfloat16],
    )[0]


def _weights_for(m: nn.Module):
    """Per-output-channel int8 weights for a QuantizedLinear, built once."""
    entry = _int8_weights.get(id(m))
    if entry is None:
        w = mx.dequantize(
            m["weight"],
            m["scales"],
            m.get("biases"),
            group_size=m.group_size,
            bits=m.bits,
            mode=getattr(m, "mode", "affine"),
        )
        ws = mx.maximum(mx.abs(w).max(axis=1), 1e-8).astype(mx.float32) / 127.0
        wq = mx.clip(
            mx.round(w.astype(mx.float32) / ws[:, None]), -127, 127
        ).astype(mx.int8)
        mx.eval(wq, ws)
        entry = (wq, ws)
        _int8_weights[id(m)] = entry
    return entry


def _eligible(m: nn.Module, k_dim: int) -> bool:
    n = m["weight"].shape[0]
    if n % _TN or k_dim % 32:
        return False
    if SCOPE == "mlp":
        return (n, k_dim) in MLP_SHAPES
    return n <= MAX_OUT and min(n, k_dim) >= MIN_DIM


# q/k/v (and gate/up) are called with the *same* activation tensor; quantize
# it once and reuse. Entries hold a strong reference to the input, so the
# id() key stays valid for the entry's lifetime.
_act_cache = OrderedDict()
_ACT_CACHE_SIZE = 4


def _quantize_rows_cached(x, k_dim):
    key = id(x)
    entry = _act_cache.get(key)
    if entry is not None and entry[0] is x:
        _act_cache.move_to_end(key)
        return entry[1], entry[2]
    xq, xs = _quantize_rows(x.reshape(-1, k_dim))
    _act_cache[key] = (x, xq, xs)
    if len(_act_cache) > _ACT_CACHE_SIZE:
        _act_cache.popitem(last=False)
    return xq, xs


def apply():
    """Patch nn.QuantizedLinear to route eligible prefill calls to W8A8."""
    ql_orig = nn.QuantizedLinear.__call__

    def ql_call(self, x):
        k_dim = x.shape[-1]
        rows = x.size // k_dim
        if rows < ROW_THRESHOLD or not _eligible(self, k_dim):
            return ql_orig(self, x)
        wq, ws = _weights_for(self)
        xq, xs = _quantize_rows_cached(x, k_dim)
        bias = self["bias"] if "bias" in self else None
        y = _int8_gemm(xq, xs, wq, ws, bias=bias)
        return y.reshape(*x.shape[:-1], wq.shape[0])

    nn.QuantizedLinear.__call__ = ql_call
    logger.info(
        "int8 NAX prefill patch applied (row threshold %d, scope %s)",
        ROW_THRESHOLD,
        SCOPE,
    )


def warmup(model: nn.Module):
    """Pre-build int8 weights for all eligible modules (optional)."""
    count = 0
    for _, m in model.named_modules():
        if isinstance(m, nn.QuantizedLinear):
            n, kp = m["weight"].shape
            k = kp * 32 // m.bits
            if _eligible(m, k):
                _weights_for(m)
                count += 1
    logger.info("int8 NAX prefill: %d modules pre-quantized", count)
