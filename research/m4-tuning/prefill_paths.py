"""All candidate prefill GEMM paths on M4 at the model's MLP shape.

Answers: can any 8-bit format beat the shipped dequant(4b)->bf16 GEMM path
on M4? M4 has no integer matrix hardware, so 8-bit only changes memory
traffic and in-kernel unpack cost, not the FMA rate (f16 == f32 == 15.7 TF
simdgroup peak, see alu_rate.py).

Also tests gate/up fusion (one 5120->34816 GEMM vs two 5120->17408).
"""

import time

import mlx.core as mx

M, K, N = 4096, 5120, 17408


def bench(fn, iters=10):
    for _ in range(3):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def report(name, t, flops):
    print(f"{name:34s} {t * 1e3:7.2f} ms  {flops / t / 1e12:6.2f} TFLOPS-eq")


mx.random.seed(0)
x = mx.random.normal((M, K)).astype(mx.bfloat16)
w = (mx.random.normal((N, K)) * 0.05).astype(mx.bfloat16)
w2 = (mx.random.normal((2 * N, K)) * 0.05).astype(mx.bfloat16)
mx.eval(x, w, w2)
flops = 2 * M * K * N

print(f"device: {mx.device_info()['device_name']}  M={M} K={K} N={N}\n")

report("bf16 GEMM", bench(lambda: x @ w.T), flops)
xh, wh = x.astype(mx.float16), w.astype(mx.float16)
mx.eval(xh, wh)
report("fp16 GEMM", bench(lambda: xh @ wh.T), flops)

for bits, gs, mode in [
    (4, 64, "affine"),
    (8, 64, "affine"),
    (8, 32, "affine"),
]:
    wq, sc, bi = mx.quantize(w, group_size=gs, bits=bits)
    mx.eval(wq, sc, bi)
    report(
        f"qmm {bits}-bit gs{gs} {mode}",
        bench(
            lambda: mx.quantized_matmul(
                x, wq, sc, bi, transpose=True, group_size=gs, bits=bits
            )
        ),
        flops,
    )
    report(
        f"dequant({bits}b gs{gs}) + bf16 GEMM",
        bench(lambda: x @ mx.dequantize(wq, sc, bi, group_size=gs, bits=bits).T),
        flops,
    )

# mxfp8 (per-32-block scale, no bias) if this mlx supports it
try:
    wq, sc = mx.quantize(w, group_size=32, bits=8, mode="mxfp8")
    mx.eval(wq, sc)
    report(
        "qmm mxfp8 gs32",
        bench(
            lambda: mx.quantized_matmul(
                x, wq, sc, None, transpose=True, group_size=32, bits=8, mode="mxfp8"
            )
        ),
        flops,
    )
except Exception as e:
    print(f"qmm mxfp8: unsupported ({str(e)[:60]})")

# gate/up fusion: two N GEMMs vs one 2N GEMM (dequant path, as shipped)
wq4, sc4, bi4 = mx.quantize(w, group_size=64, bits=4)
uq4, usc4, ubi4 = mx.quantize(w2[N:], group_size=64, bits=4)
fq4, fsc4, fbi4 = mx.quantize(w2, group_size=64, bits=4)
mx.eval(wq4, sc4, bi4, uq4, usc4, ubi4, fq4, fsc4, fbi4)


def two_gemms():
    g = x @ mx.dequantize(wq4, sc4, bi4, group_size=64, bits=4).T
    u = x @ mx.dequantize(uq4, usc4, ubi4, group_size=64, bits=4).T
    return g, u


def one_gemm():
    return x @ mx.dequantize(fq4, fsc4, fbi4, group_size=64, bits=4).T


t2 = bench(two_gemms)
t1 = bench(one_gemm)
report("gate/up: two dequant GEMMs", t2, 2 * flops)
report("gate/up: one fused 2N GEMM", t1, 2 * flops)
print(f"fusion speedup: {t2 / t1:.3f}x")
