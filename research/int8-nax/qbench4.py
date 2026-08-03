import time
import mlx.core as mx

def bench(fn, iters=20, warmup=5):
    for _ in range(warmup): mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): out = fn()
    mx.eval(out); mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

q = dict(bits=4, group_size=64, mode="affine")
for K, N, tag in [(5120, 7168, "attn qkv"), (5120, 17408, "mlp up"), (17408, 5120, "mlp down")]:
    w = mx.random.normal((N, K)).astype(mx.bfloat16) * 0.02
    wq, sc, bi = mx.quantize(w, **q)
    mx.eval(wq, sc, bi)
    print(f"\n{tag} ({K}->{N}):")
    for M in [512, 2048, 8192]:
        x = mx.random.normal((M, K)).astype(mx.bfloat16); mx.eval(x)
        t_qmm = bench(lambda: mx.quantized_matmul(x, wq, sc, bi, transpose=True, **q))
        # exactly what dequant_prefill.py does per call:
        t_dq  = bench(lambda: x @ mx.dequantize(wq, sc, bi, **q).T)
        print(f"  M={M:5d}  qmm: {t_qmm:7.3f}ms   dequant+gemm: {t_dq:7.3f}ms   ratio {t_qmm/t_dq:4.2f}x")
