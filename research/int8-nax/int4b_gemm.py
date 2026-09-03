# int8 x int4b_format -> int32 GEMM on M5 NAX, vs int8 x int8 at the same
# shapes. Tests the README "remaining lever": feed 4-bit weights to matmul2d
# directly (packed nibbles as the right operand), no int8 weight copy.
#
# Findings (M5 Max, macOS 27, mlx 0.32):
#   - int4b works only as a *memory* operand (device tensor); cooperative
#     input tensors reject it. Data handle type is `device uchar*`.
#     K must be dynamic or a multiple of 32.
#   - Nibble order: LOW nibble first (element 2k in bits 0-3, element 2k+1
#     in bits 4-7) — same order as MLX's affine 4-bit packing. Signed 4-bit
#     two's complement. Exact int32 match vs numpy.
#   - Tiling: int4b wants TM=96 (int8's best 128x128 collapses to ~58 TOPS).
#     Best found: TM=96 TN=128 NS=8 -> ~103 TOPS at M=2048, vs ~88 for the
#     best int8 x int8 config. Halved B-fetch bandwidth beats the int8 copy.
import time
import numpy as np
import mlx.core as mx

HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
using namespace metal;
"""

SRC_TMPL = """
    constexpr int N = {N};
    constexpr int K = {K};
    constexpr int TM = {TM};
    constexpr int TN = {TN};

    uint2 tgid = threadgroup_position_in_grid.xy;
    const int M = m_dim[0];

    constexpr auto desc = matmul2d_descriptor(
        TM, TN, static_cast<int>(dynamic_extent),
        /*transpose_left=*/false, /*transpose_right=*/true,
        /*relaxed_precision=*/true,
        matmul2d_descriptor::mode::multiply_accumulate);

    matmul2d<desc, execution_simdgroups<{NS}>> op;

    auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        (device int8_t*)xq, dextents<int32_t, 2>(K, M));
    auto B = tensor<device {BT}, dextents<int32_t, 2>, tensor_inline>(
        ({BPT})wq, dextents<int32_t, 2>(K, N));

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
            if (m < M && n < N) out[size_t(m) * N + n] = cT[i];
        }}
    }}
"""

I8, I4 = "int8_t", "metal::int4b_format"
BEST_TILE = {I8: (128, 128, 8), I4: (96, 128, 8)}  # from /tmp sweep, M=2048
_kernels = {}

def gemm(xq, w_buf, N, K, bt):
    """xq int8 [M,K]; w_buf = int8 [N,K] (bt=I8) or packed uint8 [N,K/2]
    (bt=I4, low nibble first). Returns raw int32 accumulators [M,N]."""
    M = xq.shape[0]
    TM, TN, NS = BEST_TILE[bt]
    key = (N, K, bt)
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(
            name=f"i4b_gemm_{N}x{K}_{'i4' if '4' in bt else 'i8'}",
            input_names=["xq", "wq", "m_dim"], output_names=["out"],
            header=HEADER, source=SRC_TMPL.format(
                N=N, K=K, TM=TM, TN=TN, NS=NS, BT=bt,
                BPT="device uchar*" if "4" in bt else "device int8_t*"))
    TM, TN, NS = BEST_TILE[bt]
    return _kernels[key](
        inputs=[xq, w_buf, mx.array([M], dtype=mx.int32)],
        grid=(N // TN * 32 * NS, (M + TM - 1) // TM, 1),
        threadgroup=(32 * NS, 1, 1),
        output_shapes=[(M, N)], output_dtypes=[mx.int32])[0]

def pack_nibbles(w4):
    """w4 int8 [N,K], values in [-8,7] -> packed uint8 [N,K/2], low first."""
    u = (w4.astype(np.int32) & 0xF).astype(np.uint8)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)

# ---------------- correctness ----------------
rng = np.random.default_rng(0)
M, K, N = 300, 160, 256  # deliberately off-tile M; K multiple of 32
x = rng.integers(-127, 128, size=(M, K), dtype=np.int8)
w4 = rng.integers(-8, 8, size=(N, K), dtype=np.int8)
ref = x.astype(np.int32) @ w4.astype(np.int32).T
xq = mx.array(x)

r8 = np.array(gemm(xq, mx.array(w4), N, K, I8))
r4 = np.array(gemm(xq, mx.array(pack_nibbles(w4)), N, K, I4))
print("int8  exact:", np.array_equal(r8, ref))
print("int4b exact:", np.array_equal(r4, ref))

# ---------------- speed ----------------
def bench(label, fn, macs):
    mx.eval(fn()); mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(5): mx.eval(fn())
    mx.synchronize()
    t = (time.perf_counter() - t0) / 5
    print(f"{label:24s} {t*1e3:7.2f} ms   {2*macs/t/1e12:6.1f} TOPS-eq")

for (Mb, Kb, Nb) in [(2048, 5120, 17408), (2048, 17408, 5120),
                     (4096, 5120, 17408), (4096, 17408, 5120),
                     (8192, 5120, 17408)]:
    print(f"\n-- M={Mb} K={Kb} N={Nb}")
    x = mx.array(rng.integers(-127, 128, size=(Mb, Kb), dtype=np.int8))
    w8 = mx.array(rng.integers(-8, 8, size=(Nb, Kb), dtype=np.int8))
    w4p = mx.array(rng.integers(0, 256, size=(Nb, Kb // 2), dtype=np.uint8))
    macs = Mb * Kb * Nb
    bench("int8 x int8", lambda: gemm(x, w8, Nb, Kb, I8), macs)
    bench("int8 x int4b", lambda: gemm(x, w4p, Nb, Kb, I4), macs)
