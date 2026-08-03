import mlx.core as mx

header = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""
src = """
    uint i = thread_position_in_grid.x;
    out[i] = inp[i] + 1.0f;
"""
try:
    k = mx.fast.metal_kernel(
        name="topstest", input_names=["inp"], output_names=["out"],
        header=header, source=src)
    r = k(inputs=[mx.zeros(4)], grid=(4,1,1), threadgroup=(4,1,1),
          output_shapes=[(4,)], output_dtypes=[mx.float32])
    mx.eval(r)
    print("MPP header compiles in mx.fast.metal_kernel: OK ->", r[0].tolist())
except Exception as e:
    print("FAILED:", str(e)[:800])
