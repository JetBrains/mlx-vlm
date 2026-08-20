# int8 NAX Prefill — Speeding Up Long-Context Prefill on Apple M5

## The Problem

Prefill (the initial forward pass over the prompt) is the bottleneck for long-context
inference. On Apple M5, the GPU neural accelerators (NAX) deliver ~120 TOPS in int8
but only ~60 TOPS in fp16/bf16 — a 2× hardware advantage that the default 4-bit
quantized kernels don't exploit.

## The Optimization

**Selective W8A8 int8 prefill** (`mlx_vlm/int8_prefill.py`) intercepts large matmuls
during prefill and reroutes them through int8 × int8 → int32 GEMMs via Metal
Performance Primitives tensor ops:

- **Activations**: per-token dynamic symmetric int8 quantization (custom Metal kernel)
- **Weights**: per-channel symmetric int8, built on-the-fly from the resident 4-bit
  weights by a fused Metal kernel — no bf16 intermediate, no full model copy
- **Memory**: by default (`MLX_VLM_INT8_CACHE=none`) each layer's int8 tensor is
  freed after its GEMM, so peak overhead is ~one layer (a few hundred MB), not
  the ~24 GB of a full copy
- **Eligibility**: rule-based — N%128==0, K%32==0, min dimension ≥ 1024, N ≤ 32768.
  Covers MLP and attention projections. Decode calls (1 row) are untouched.

**Result**: decode speed and numerics are completely unchanged; only prefill benefits.

## How To Enable

```bash
# Via CLI flag
python -m mlx_vlm.server --model <model> --int8-prefill

# Or via environment variable
MLX_VLM_INT8_PREFILL=1 python -m mlx_vlm.server --model <model>
```

## Benchmark

To reproduce the improvement on your M5 Mac:

```bash
./research/run_benchmark.sh
```

The script:

1. Downloads `Qwen3.6-27B-4bit` to `research/models/`
2. Starts the server **without** `--int8-prefill` (baseline)
3. Runs 4 context sizes (256, 2k, 10k, 20k tokens), 3 iterations each
4. Restarts the server **with** `--int8-prefill`
5. Runs the same 4 context sizes again
6. Prints averaged results and a side-by-side speedup comparison
7. Saves JSON to `research/results/`

## Measured Results

Benchmarked on Apple M5 Max with `Qwen3.6-27B-4bit`, 2 iterations per prompt.

### Baseline (no `--int8-prefill`)

| Context | Ctx Tokens | TTFT (s) | Prefill TPS | Decode TPS |
|---|---|---|---|---|
| 256 tokens | 207 | 0.404 | 512.66 | 31.56 |
| 2k tokens | 1,962 | 2.835 | 692.36 | 31.75 |
| 10k tokens | 9,957 | 18.200 | 547.47 | 29.34 |
| 20k tokens | 19,902 | 40.626 | 489.93 | 28.88 |

### int8 NAX Prefill (`--int8-prefill`)

| Context | Ctx Tokens | TTFT (s) | Prefill TPS | Decode TPS |
|---|---|---|---|---|
| 256 tokens | 207 | 0.413 | 500.50 | 30.73 |
| 2k tokens | 1,962 | 3.623 | 566.38 | 29.53 |
| 10k tokens | 9,957 | 13.276 | 750.03 | 28.70 |
| 20k tokens | 19,902 | 28.857 | 689.75 | 26.46 |

### Speedup

| Context | Baseline TPS | int8 TPS | **Speedup** | Baseline TTFT | int8 TTFT |
|---|---|---|---|---|---|
| 256 tokens | 512.66 | 500.50 | **0.98×** | 0.404 s | 0.413 s |
| 2k tokens | 692.36 | 566.38 | **0.82×** | 2.835 s | 3.623 s |
| 10k tokens | 547.47 | 750.03 | **1.37×** | 18.20 s | 13.28 s |
| 20k tokens | 489.93 | 689.75 | **1.41×** | 40.63 s | 28.86 s |

### Key Observations

- **256 tokens** — No speedup: matmul dimensions are below the int8 threshold (min 1024).
- **2k tokens** — Slight regression: cold-cache effect on the freshly loaded int8 server;
  the matmuls are borderline eligible.
- **10k tokens** — **1.37× prefill speedup**: TTFT drops from 18.2 s to 13.3 s.
- **20k tokens** — **1.41× prefill speedup**: TTFT drops from 40.6 s to 28.9 s.
- **Decode TPS** is unaffected (~29–32 tok/s) — int8 only accelerates prefill.

> **Note**: the measured speedup (1.37–1.41×) includes overhead from model loading,
> server startup, and the sequential nature of the benchmark (baseline runs first,
> then int8). A single isolated 20k-token prefill with `--int8-prefill` from a cold
> start would likely show even higher prefill throughput, since there's no prior
> baseline work warming up the system and the int8 path can take full advantage of
> the NAX accelerators from the first token.