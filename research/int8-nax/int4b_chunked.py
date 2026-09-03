# Option (b) from quantization-and-scales.md S9: int8 x int4b GEMM with the
# K reduction split into C-wide chunks, each chunk's int32 partial sum scaled
# by a per-(row,chunk) weight scale and accumulated in fp32. This preserves
# group-granular scales (C=64 keeps the stored group-64 grid exactly; C=128
# merges pairs of groups onto a shared scale, etc.) at the cost of NCH =
# K/C matmul2d calls + epilogues instead of one.
#
# Measures that cost vs the single-shot int4b GEMM (~98 TOPS) and checks
# exactness vs numpy.
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
    constexpr int C = {C};        // chunk width along K
    constexpr int NCH = K / C;
    constexpr int CAP = {CAP};    // destination capacity per thread

    uint2 tgid = threadgroup_position_in_grid.xy;
    const int M = m_dim[0];

    constexpr auto desc = matmul2d_descriptor(
        TM, TN, C,
        /*transpose_left=*/false, /*transpose_right=*/true,
        /*relaxed_precision=*/true,
        matmul2d_descriptor::mode::multiply_accumulate);

    matmul2d<desc, execution_simdgroups<{NS}>> op;

    auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        (device int8_t*)xq, dextents<int32_t, 2>(K, M));
    auto B = tensor<device metal::int4b_format, dextents<int32_t, 2>, tensor_inline>(
        (device uchar*)wq, dextents<int32_t, 2>(K, N));

    auto cT = op.get_destination_cooperative_tensor<
        decltype(A.slice(0, 0)), decltype(B.slice(0, 0)), int32_t>();

    float facc[CAP];
    int   gn[CAP];   // global n per local element (layout fixed across runs)
    int   gm[CAP];
#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        facc[i] = 0.0f;
        if (cT.is_valid_element(i)) {{
            auto idx = cT.get_multidimensional_index(i);
            gn[i] = int(tgid.x) * TN + idx[0];
            gm[i] = int(tgid.y) * TM + idx[1];
        }} else {{
            gn[i] = -1; gm[i] = -1;
        }}
    }}

    for (int ch = 0; ch < NCH; ++ch) {{
#pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i)
            if (cT.is_valid_element(i)) cT[i] = 0;

        auto tA = A.slice(ch * C, int(tgid.y) * TM);
        auto tB = B.slice(ch * C, int(tgid.x) * TN);
        op.run(tA, tB, cT);

#pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
            if (cT.is_valid_element(i) && gn[i] < N)
                facc[i] += float(cT[i]) * wsc[size_t(gn[i]) * NCH + ch];
        }}
    }}

#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        if (gm[i] >= 0 && gm[i] < M && gn[i] < N)
            out[size_t(gm[i]) * N + gn[i]] = facc[i] * xs[gm[i]];
    }}
"""

_kernels = {}

def chunked_gemm(xq, xs, wp, wsc, N, K, C, TM=96, TN=128, NS=8):
    M = xq.shape[0]
    cap = (TM * TN + 32 * NS - 1) // (32 * NS)
    key = (N, K, C, TM, TN, NS)
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(
            name=f"i4bc_{N}x{K}_c{C}_{TM}x{TN}x{NS}",
            input_names=["xq", "wq", "wsc", "xs", "m_dim"],
            output_names=["out"], header=HEADER,
            source=SRC_TMPL.format(N=N, K=K, C=C, TM=TM, TN=TN, NS=NS, CAP=cap))
    return _kernels[key](
        inputs=[xq, wp, wsc, xs, mx.array([M], dtype=mx.int32)],
        grid=(N // TN * 32 * NS, (M + TM - 1) // TM, 1),
        threadgroup=(32 * NS, 1, 1),
        output_shapes=[(M, N)], output_dtypes=[mx.float32])[0]

def pack_nibbles(w4):
    u = (w4.astype(np.int32) & 0xF).astype(np.uint8)
    return (u[:, 0::2] | (u[:, 1::2] << 4)).astype(np.uint8)

# ---------------- correctness ----------------
rng = np.random.default_rng(1)
M, K, N, C = 200, 256, 256, 64
x = rng.integers(-127, 128, size=(M, K), dtype=np.int8)
w4 = rng.integers(-8, 8, size=(N, K), dtype=np.int8)
wsc = rng.uniform(0.001, 0.05, size=(N, K // C)).astype(np.float32)
xs = rng.uniform(0.001, 0.05, size=(M,)).astype(np.float32)

acc = np.zeros((M, N), dtype=np.float64)
for ch in range(K // C):
    part = x[:, ch*C:(ch+1)*C].astype(np.int64) @ w4[:, ch*C:(ch+1)*C].astype(np.int64).T
    acc += part * wsc[None, :, ch]
ref = acc * xs[:, None]

r = np.array(chunked_gemm(mx.array(x), mx.array(xs), mx.array(pack_nibbles(w4)),
                          mx.array(wsc), N, K, C))
rel = np.abs(r - ref) / np.maximum(np.abs(ref), 1e-6)
print(f"chunked C={C}: max rel err = {rel.max():.2e}")

# ---------------- speed ----------------
def bench(label, fn, macs):
    mx.eval(fn()); mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(5): mx.eval(fn())
    mx.synchronize()
    t = (time.perf_counter() - t0) / 5
    print(f"{label:36s} {t*1e3:7.2f} ms   {2*macs/t/1e12:6.1f} TOPS-eq")

for (Mb, Kb, Nb) in [(2048, 5120, 17408), (2048, 17408, 5120)]:
    print(f"\n-- M={Mb} K={Kb} N={Nb}")
    x = mx.array(rng.integers(-127, 128, size=(Mb, Kb), dtype=np.int8))
    xs_b = mx.array(rng.uniform(0.001, 0.05, size=(Mb,)).astype(np.float32))
    wp = mx.array(rng.integers(0, 256, size=(Nb, Kb // 2), dtype=np.uint8))
    macs = Mb * Kb * Nb
    for C in (64, 128, 256, 512, 1024):
        wsc_b = mx.array(rng.uniform(0.001, 0.05, size=(Nb, Kb // C)).astype(np.float32))
        bench(f"chunked C={C}",
              lambda: chunked_gemm(x, xs_b, wp, wsc_b, Nb, Kb, C), macs)
