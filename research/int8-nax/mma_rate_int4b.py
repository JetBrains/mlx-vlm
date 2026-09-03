# int8 x int4b -> int32 raw NAX MMA rate, alongside the known combos.
# Companion to mma_rate.py; see README.md.
import time
import mlx.core as mx

HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

template <typename AT, typename BT, typename CT, int TM, int TN, int TK, int ITERS>
METAL_FUNC int run_mma_loop(uint seed) {
    constexpr auto desc = matmul2d_descriptor(
        TM, TN, TK, false, false, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, metal::execution_simdgroup> op;
    auto a = op.template get_left_input_cooperative_tensor<AT, BT, CT>();
    auto b = op.template get_right_input_cooperative_tensor<AT, BT, CT>();
    auto c = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), CT>();
    for (int i = 0; i < int(a.get_capacity()); i++) a[i] = AT(seed & 3);
    for (int i = 0; i < int(b.get_capacity()); i++) b[i] = BT((seed >> 2) & 3);
    for (int i = 0; i < int(c.get_capacity()); i++) c[i] = CT(0);
    for (int it = 0; it < ITERS; it++) {
        op.run(a, b, c);
    }
    int acc = 0;
    for (int i = 0; i < int(c.get_capacity()); i++) acc += int(c[i]);
    return acc;
}
"""

SRC_TMPL = """
    uint gid = thread_position_in_grid.x;
    int r = run_mma_loop<{AT}, {BT}, {CT}, {TM}, {TN}, {TK}, {ITERS}>(gid);
    if (r == -12345) out[0] = r;  // never true; keeps the loop live
"""

ITERS = 20000
TGS = 2048
TPT = 128

def bench(name, AT, BT, CT, TM, TN, TK):
    src = SRC_TMPL.format(AT=AT, BT=BT, CT=CT, TM=TM, TN=TN, TK=TK, ITERS=ITERS)
    try:
        k = mx.fast.metal_kernel(name=f"mma_{name}", input_names=["dummy"],
                                 output_names=["out"], header=HEADER, source=src)
        f = lambda: k(inputs=[mx.zeros(1, dtype=mx.int32)],
                      grid=(TGS*TPT,1,1), threadgroup=(TPT,1,1),
                      output_shapes=[(1,)], output_dtypes=[mx.int32])
        mx.eval(f()); mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(3): mx.eval(f())
        mx.synchronize()
        t = (time.perf_counter() - t0) / 3
        simds = TGS * (TPT // 32)
        ops = 2.0 * TM * TN * TK * ITERS * simds
        print(f"{name:26s} tile {TM}x{TN}x{TK}: {ops/t/1e12:7.1f} TOPS")
    except Exception as e:
        print(f"{name:26s} FAILED: {str(e)[-400:]}")

bench("int8_int8", "int8_t", "int8_t", "int32_t", 16, 32, 32)
bench("int8_int4b_k32", "int8_t", "metal::int4b_format", "int32_t", 16, 32, 32)
bench("int8_int4b_k64", "int8_t", "metal::int4b_format", "int32_t", 16, 32, 64)
bench("uint8_uint4b_k32", "uint8_t", "metal::uint4b_format", "int32_t", 16, 32, 32)
bench("bf16_int4b", "bfloat", "metal::int4b_format", "float", 16, 32, 16)
