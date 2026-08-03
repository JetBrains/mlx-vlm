import time
import mlx.core as mx

def bench(fn, iters=20, warmup=5):
    for _ in range(warmup): mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())          # force each iteration to actually run
    mx.synchronize()
    return (time.perf_counter() - t0) / iters

N = 4096
for dt in [mx.float32, mx.float16, mx.bfloat16]:
    a = mx.random.normal((N, N)).astype(dt)
    b = mx.random.normal((N, N)).astype(dt)
    mx.eval(a, b)
    t = bench(lambda: a @ b)
    print(f"{N}x{N} {dt}: {2*N**3/t/1e12:6.2f} TFLOPS")

# redo the key prefill comparison with the fixed harness
K, Nn = 5120, 17408
w = mx.random.normal((Nn, K)).astype(mx.bfloat16) * 0.02
q = dict(bits=4, group_size=64, mode="affine")
wq, sc, bi = mx.quantize(w, **q)
mx.eval(w, wq, sc, bi)
print()
for M in [64, 512, 2048, 8192]:
    x = mx.random.normal((M, K)).astype(mx.bfloat16); mx.eval(x)
    tb = bench(lambda: x @ w.T)
    tq = bench(lambda: mx.quantized_matmul(x, wq, sc, bi, transpose=True, **q))
    td = bench(lambda: x @ mx.dequantize(wq, sc, bi, **q).T)
    gf = lambda t: 2*M*K*Nn/t/1e12
    print(f"M={M:5d}  bf16 {tb*1e3:7.3f}ms ({gf(tb):5.1f}TF)  qmm4 {tq*1e3:7.3f}ms ({gf(tq):5.1f}TF)  dq+gemm {td*1e3:7.3f}ms ({gf(td):5.1f}TF)")
