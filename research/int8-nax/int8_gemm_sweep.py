"""Tile-size sweep for the int8 NAX GEMM prototype (see int8_gemm_v1.py).

Sweeps (TM, TN, NSIMD, K-loop style) at the Qwen3.6-27B MLP shape to find a
configuration that approaches the 120-TOPS int8 MMA peak measured in
mma_rate.py. MLX's own bf16 NAX GEMM uses bm128/bn128/bk256.
"""

import time

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

    constexpr auto desc = matmul2d_descriptor(
        TM, TN, {KDESC},
        /*transpose_left=*/false, /*transpose_right=*/true,
        /*relaxed_precision=*/true,
        matmul2d_descriptor::mode::multiply_accumulate);

    matmul2d<desc, execution_simdgroups<{NSIMD}>> op;

    auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        (device int8_t*)xq, dextents<int32_t, 2>(K, {M}));
    auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
        (device int8_t*)wq, dextents<int32_t, 2>(K, N));

    auto cT = op.get_destination_cooperative_tensor<
        decltype(A.slice(0, 0)), decltype(B.slice(0, 0)), int32_t>();

#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        if (cT.is_valid_element(i)) cT[i] = 0;
    }}

{BODY}

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

BODY_DYNK = """
    auto tA = A.slice(0, int(tgid.y) * TM);
    auto tB = B.slice(0, int(tgid.x) * TN);
    op.run(tA, tB, cT);
"""

BODY_KLOOP = """
    constexpr int TK = {TK};
    for (int k = 0; k < K; k += TK) {{
        auto tA = A.template slice<TK, TM>(k, int(tgid.y) * TM);
        auto tB = B.template slice<TK, TN>(k, int(tgid.x) * TN);
        op.run(tA, tB, cT);
    }}
"""


def make_kernel(M, N, K, TM, TN, NSIMD, TK=None):
    if TK is None:
        body = BODY_DYNK
        kdesc = "static_cast<int>(dynamic_extent)"
    else:
        body = BODY_KLOOP.replace("{TK}", str(TK)).replace("{{", "{").replace(
            "}}", "}")
        kdesc = str(TK)
    name = f"i8g_{TM}x{TN}x{TK or 0}_s{NSIMD}_{M}"
    src = SRC_TMPL.format(M=M, N=N, K=K, TM=TM, TN=TN, NSIMD=NSIMD,
                          KDESC=kdesc, BODY=body)
    k = mx.fast.metal_kernel(name=name, input_names=["xq", "wq", "xs", "ws"],
                             output_names=["out"], header=HEADER, source=src)

    def run(xq, xs, wq, ws):
        return k(inputs=[xq, wq, xs, ws],
                 grid=(N // TN * 32 * NSIMD, M // TM, 1),
                 threadgroup=(32 * NSIMD, 1, 1),
                 output_shapes=[(M, N)],
                 output_dtypes=[mx.bfloat16])[0]

    return run


def bench(fn, iters=15, warmup=4):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    mx.random.seed(0)
    M, K, N = 2048, 5120, 17408
    x = mx.random.normal((M, K)).astype(mx.bfloat16)
    w = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
    xs = mx.maximum(mx.abs(x).max(axis=1), 1e-8).astype(mx.float32) / 127.0
    ws = mx.maximum(mx.abs(w).max(axis=1), 1e-8).astype(mx.float32) / 127.0
    xq = mx.clip(mx.round(x.astype(mx.float32) / xs[:, None]), -127, 127).astype(mx.int8)
    wq = mx.clip(mx.round(w.astype(mx.float32) / ws[:, None]), -127, 127).astype(mx.int8)
    mx.eval(xq, xs, wq, ws)

    ref = ((xq.astype(mx.float32) @ wq.astype(mx.float32).T)
           * xs[:, None] * ws[None, :])
    mx.eval(ref)
    flops = 2.0 * M * K * N

    t_bf = bench(lambda: x @ w.T)
    print(f"bf16 reference: {t_bf*1e3:7.3f} ms  {flops/t_bf/1e12:5.1f} TF\n")

    configs = [
        # (TM, TN, NSIMD, TK)
        (64, 32, 4, None),     # v1 baseline
        (64, 64, 4, None),
        (128, 64, 4, None),
        (128, 128, 4, None),
        (128, 64, 8, None),
        (128, 128, 8, None),
        (256, 128, 8, None),
        (64, 64, 4, 128),
        (128, 64, 4, 128),
        (128, 128, 4, 128),
        (128, 128, 4, 256),
        (128, 128, 8, 256),
        (128, 64, 4, 256),
        (128, 64, 4, 512),
        (128, 128, 16, None),
        (128, 256, 8, None),
        (256, 256, 8, None),
        (128, 128, 8, 512),
        (128, 128, 8, 1024),
    ]
    for TM, TN, NSIMD, TK in configs:
        if M % TM or N % TN or (TK and K % TK):
            continue
        tag = f"TM{TM} TN{TN} simd{NSIMD} TK{TK or 'dyn'}"
        try:
            run = make_kernel(M, N, K, TM, TN, NSIMD, TK)
            got = run(xq, xs, wq, ws)
            mx.eval(got)
            err = mx.abs(got.astype(mx.float32) - ref).max().item()
            ok = "OK " if err < ref.abs().max().item() * 1e-2 else "BAD"
            t = bench(lambda: run(xq, xs, wq, ws))
            print(f"{tag:32s} {t*1e3:7.3f} ms  {flops/t/1e12:5.1f} TOPS-eq  "
                  f"[{ok} err {err:.4f}]  vs bf16 {t_bf/t:4.2f}x")
        except Exception as e:
            print(f"{tag:32s} FAILED: {str(e)[:140]}")


if __name__ == "__main__":
    main()
