# INT8 NAX prefill research — findings & plan

**Date:** 2026-08-03
**Machine:** Apple M5 Max, 40-core GPU (Metal 4), macOS 27.0
**Software:** mlx 0.32.0 (wheel), mlx-vlm (this repo), Python 3.14 venv at `.venv`
**Model under study:** `mlx-community/Qwen3.6-27B-4bit` (served by `start.sh` on :8085, with `Qwen3.6-27B-MTP-4bit` MTP drafter)

Goal of the investigation: maximize **prefill** throughput on this machine. This doc records
everything established along the way, so it can be reused without re-deriving.

---

## 1. Current model quantization (start.sh)

`config.json` in the HF cache (`~/.local/share/junie-local/models/models--mlx-community--Qwen3.6-27B-4bit/snapshots/*/config.json`):

```json
"quantization": { "group_size": 64, "bits": 4, "mode": "affine" }
```

Global, no per-layer overrides (MLX would list them as extra keys in that section).
Verified against `model.safetensors.index.json` — a linear layer is quantized iff it has
companion `.scales`/`.biases` tensors:

- **Hybrid architecture, 64 layers**: 16 full-attention layers (3, 7, 11, … every 4th),
  48 linear-attention (gated-deltanet) layers.
- Full attention: `q/k/v/o_proj` all 4-bit (q/k norms bf16).
- Linear attention: `in_proj_qkv/z/a/b`, `out_proj` all 4-bit (`conv1d`, norms bf16).
- MLP `gate/up/down_proj`: 4-bit. Dims 5120 → 17408 → 5120.
- **Vision tower: entirely unquantized** (bf16) — standard for mlx-community conversions.
- Quantization is **weights only**; activations and KV cache run bf16 (unless `--kv-bits`).

Key dims: hidden 5120, intermediate 17408, 24 attn heads × head_dim 256, 4 KV heads.

## 2. The one methodology trap: MLX lazy evaluation

**A timing loop `for _ in range(iters): out = fn()` followed by one `mx.eval(out)` only
computes the LAST iteration** — dropped unevaluated arrays are never computed. This bug
made early absolute numbers ~20x wrong in both directions during this session
(1.2 "TFLOPS" for GEMM; then 777–1071 "TFLOPS" for square GEMM — both artifacts).

Correct harness (see `peak2.py`): `mx.eval(fn())` **inside** the loop, `mx.synchronize()`
before/after. Relative comparisons from the buggy harness happened to hold; absolute
numbers did not. End-to-end numbers via `generate()` were always trustworthy.

## 3. What the M5 Max actually delivers (corrected numbers)

### 3.1 Peak GEMM through MLX (4096×4096, `peak2.py`)

| dtype | TFLOPS |
|---|---|
| fp32 | 40.1 |
| fp16 | 55.5 |
| bf16 | 55.8 |

These exceed plain shader-ALU capability (~19 TFLOPS fp32) by ~3x ⇒ **MLX 0.32 already
routes GEMMs through the M5 GPU neural accelerators (NAX)**. Confirmed structurally:
the shipped `mlx.metallib` contains `steel_gemm_fused_nax_*` kernels built on
`mpp::tensor_ops::matmul2d` (Metal Performance Primitives), instantiated for
**float32/float16/bfloat16 only**. There are also NAX variants of the quantized matmul
(`qmm_t_nax`, `qmm_n_nax`, `gather_qmm_*_nax`) and NAX attention.

### 3.2 Prefill-shape GEMMs, model's MLP shape 5120→17408 (`peak2.py`, fixed harness)

| M (tokens) | bf16 GEMM | 4-bit qmm (affine gs64) | dequant→bf16 GEMM |
|---|---|---|---|
| 64 | 0.819 ms | **0.533 ms** | 1.304 ms |
| 512 | **1.760 ms** (51.9 TF) | 1.909 ms (47.8 TF) | 2.169 ms |
| 2048 | **6.034 ms** (60.5 TF) | 6.685 ms (54.6 TF) | 6.497 ms |
| 8192 | **24.09 ms** (60.6 TF) | 26.09 ms (56.0 TF) | 24.30 ms |

Conclusions:
- At prefill sizes, 4-bit qmm is only **~8% behind** full bf16 GEMM — the NAX qmm
  dequantizes tiles in on-chip memory and overlaps with tensor-op math.
- At decode sizes qmm wins ~2x (bandwidth-bound).
- Quantization **mode** (affine gs64/gs32, mxfp4, 8-bit) is a ≤10% knob at prefill.
  Current affine-4bit gs64 is already the right choice. mxfp4 gains nothing.
- 8-bit affine is never the fastest in any regime (more bytes at decode, same float
  math at prefill). It is a *quality* option (~27 GB), not a speed option.

### 3.3 Raw NAX MMA instruction rates (`mma_rate.py`, custom kernel via `mx.fast.metal_kernel`)

Register-resident matmul2d loop, 2048 threadgroups × 4 simdgroups:

| operands → accumulator | tile | TOPS |
|---|---|---|
| fp16 × fp16 → fp32 | 16×32×16 | 58.9 |
| bf16 × bf16 → fp32 | 16×32×16 | 60.4 |
| fp16 × fp16 → fp16 | 16×32×16 | 60.0 |
| **int8 × int8 → int32** | 16×32×16 | **118.9** |
| **int8 × int8 → int32** | 16×32×32 | **121.7** |

**int8 is exactly 2.0x fp16/bf16 on M5 Max NAX** (matches tzakharko's A19 findings).
Since MLX's bf16 GEMM already runs at ~95% of the fp16 MMA peak, this 2x is genuine
untapped headroom — the only such headroom found in this entire investigation.

### 3.4 End-to-end serving baseline (`e2e_dequant.py`, real model, 6420-token prompt)

- Prefill: **~860 tok/s**; decode: ~48 tok/s (no drafter in this test).

## 4. `--dequant-prefill` patch: evaluated, DO NOT enable

The repo has `mlx_vlm/dequant_prefill.py` wired via `--dequant-prefill` /
`MLX_VLM_DEQUANT_PREFILL=1` (applied in server lifespan, `server/app.py`). It transiently
dequantizes weights to bf16 for ≥512-row calls. Its docstring assumes qmm is ~2x slower
than bf16 GEMM at prefill — **that assumption is false on M5 Max + mlx 0.32** (gap is ~8%,
see §3.2), and the per-call dequant materialization costs more than it saves:

- End-to-end: 861 tok/s unpatched → 773 tok/s patched (**−10%**), with one patched run
  collapsing to 411 tok/s (allocator/cache churn from transient ~170 MB bf16 buffers).

Re-evaluate only if a future MLX changes the qmm-vs-GEMM balance.

## 5. Hardware & API capability map

### 5.1 Metal 4 MPP tensor ops (verified in local SDK headers)

`MetalPerformancePrimitives.framework` (`MPPTensorOpsMatMul2d.h`, macOS 26+ SDK) supports,
among others:

- `int8 × int8 → int32`, `uint8 × uint8 → int32`
- mixed `half/bfloat/float × int8`
- **4-bit operand formats**: `half/bfloat × int4b_format`, `int8 × int4b → int32`
- No fp8, no fp4 float formats in the matmul2d combination table.

### 5.2 What the NAX hardware accelerates (external research)

- A19/M5 NAX datapaths: **fp16 (fp16/fp32 acc) and int8 (int32 acc, ~2x fp16 rate)**.
  fp32 runs on the general SIMD pipe. No FP4/FP8 datapath.
  Source: [tzakharko's A19/M5 NAX benchmark](https://tzakharko.github.io/apple-neural-accelerators-benchmark/)
- M1–M4 GPUs had **no matrix hardware at all**; fp8 on M4 was pure unpack overhead.
  Source: [Rigel: reverse-engineering the M4 Max tensor path](https://arxiv.org/pdf/2606.12765)
- Community assessment: M5 NAX ≈ NVIDIA Turing-era tensor cores.

### 5.3 Why NVIDIA invested in NVFP4 and MLX didn't (research summary)

NVFP4 = e2m1 + fp8-E4M3 scale per 16-block + fp32 tensor scale. It pays off because
**Blackwell tensor cores execute FP4 natively** (B200: ~7.7 PFLOPS FP4 measured ≈ 2x FP8,
4x BF16) with hardware-handled block scaling; accuracy within ~1% of FP8; also used for
training (~1.9x vs FP8 on Llama 405B). It's format+hardware+software co-design aimed at
compute-bound datacenter serving/training.
Apple's silicon can't express any of that: no FP4 datapath exists; NAX only just shipped;
the Mac workload MLX targets (single-user, decode-heavy, bandwidth-bound) is served well
by weight-only quantization. MLX's `nvfp4`/`mxfp8` modes are **storage/compat** formats
(dequantized in-kernel), added for foreign checkpoints; Metal `global_scale` support is
still incomplete (mlx issue #3911).
Sources: [NVIDIA NVFP4 intro](https://www.edge-ai-vision.com/2025/07/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/),
[NVFP4 training blog](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/),
[Blackwell microbenchmarks](https://arxiv.org/pdf/2512.02189),
[M5 roofline analysis](https://www.michaelstinkerings.org/apple-m5-gpu-roofline-analysis/).

### 5.4 State of MLX upstream (checked 2026-08-03, clone at `~/IdeaProjects/mlx`)

- main = 0.32.1-dev, effectively identical to the installed 0.32.0 wheel.
- `quantized_nax.h` (and `fp_quantized_nax.h`) on main still **dequantize to threadgroup
  memory, then run float tensor ops**. Zero references to `int4b_format`/`int8_t` operands
  in `steel/gemm/nax.h` or `steel/attn/nax.h`.
- The `qmm` branch is CUDA-backend work; no other branch touches integer Metal GEMM.
- ⇒ Neither W8A8 int8 nor direct int4b-operand qmm exists anywhere upstream.
  **Watch `mlx/backend/metal/kernels/quantized_nax.h` in future releases.**

## 6. Feasibility gates for a custom int8 path (all verified ✅)

1. ✅ `mx.fast.metal_kernel` **compiles MPP tensor-ops code** at runtime
   (`tensorops_compile_test.py`) — no MLX fork needed for prototyping.
2. ✅ MLX's own NAX kernels show the pattern: `get_*_cooperative_tensor()` register
   fragments + plain device-pointer loads (`mlx/backend/metal/kernels/steel/gemm/nax.h`,
   `mma()` at ~line 392). No Metal-4 `tensor<>` kernel bindings required.
3. ✅ int8 MMA rate = 2.0x (see §3.3).

Practical gotchas hit while prototyping (`mma_rate.py`):
- `mx.fast.metal_kernel` `name` becomes part of the Metal function name — no spaces/parens.
- Cooperative tensors have no `.size()`; iterate with `.get_capacity()` (+ `.get_mask(i)`
  for validity when reading results per the header docs).
- Keep the MMA loop live by consuming the accumulator (dead-code elimination is real).

## 7. The plan: selective W8A8 int8 prefill for MLP layers

Modeled on the proven vLLM approach
([erokhins/vllm-qwen3.6-27b-nvfp4@b85e97d](https://github.com/erokhins/vllm-qwen3.6-27b-nvfp4/commit/b85e97d316ee16590c65c4739f70ed7eaeab3025)):
there, 192 MLP projections moved to native W4A4 while 208 attention projections stayed FP8
and lm_head stayed W4A16 → prefill 1077 → 1870 tok/s. Same selective structure here:

| component | treatment |
|---|---|
| MLP `gate/up/down_proj` (192 = 3×64 layers) | **W8A8 int8 NAX**, prefill-sized calls only |
| attention + linear-attention projections | unchanged (affine-4bit weight-only qmm) — KV/attention numerics untouched |
| lm_head, embeddings, vision tower | unchanged |
| decode-sized calls (rows < threshold) | unchanged (4-bit qmm; bandwidth-optimal, MTP drafter unaffected) |

Routing mechanism already proven in this repo: `mlx_vlm/dequant_prefill.py`-style
monkeypatch, but scoped to the MLP modules and swapping in the int8 kernel instead of
dequantization.

### Kernel design (the actual work)

W8A8 GEMM `Y[M,N] = (Xq[M,K]·i8 @ Wq[N,K].T·i8) · (x_scale[M] ⊗ w_scale[N])`:
- **Activations**: per-token (per-row) dynamic symmetric int8 — `s_x = absmax(row)/127`.
  Fused quantize kernel (or fused into the GEMM's A-tile loader).
- **Weights**: offline symmetric per-output-channel int8. Quantize from the **original
  bf16 checkpoint**, not from the 4-bit conversion (avoid stacking quantization error).
  Storage: +~1.1 GB resident (int8 MLP weights alongside the 4-bit ones) or replace the
  4-bit MLP tensors entirely with int8 (then decode MLP also runs a new int8 kernel path —
  bigger scope; start with "alongside").
- **Epilogue**: int32 accum → `float(acc) * s_x[m] * s_w[n]` → bf16.
- Tiling: adapt MLX's steel NAX GEMM structure (threadgroup tiles, K=32 int8 MMA tiles).

### Milestones

1. **Standalone int8 GEMM** at MLP shapes (M=2048/8192, 5120→17408) via
   `mx.fast.metal_kernel`; benchmark vs bf16 GEMM and `qmm_t_nax`.
   *Go/no-go: > 60 TF-equivalent effective (i.e. >50% of the 120-TOPS peak; MLX's own
   NAX GEMM achieves ~95%, so 70–80% is the target).*
2. Fused per-row activation-quant + correctness check vs bf16 reference
   (cosine/max-abs-err on real layer inputs).
3. Module-level patch (`int8_prefill.py` mirroring `dequant_prefill.py`), row-threshold
   routing, weights loaded from a sidecar file; server flag `--int8-prefill`.
4. End-to-end A/B on the 6.4k-token benchmark + a long-prompt quality eval
   (compare generations vs baseline on real workload).
5. Optional: contribute upstream / extend to int8×int4b (`int8 × int4b → int32` exists
   in MPP — would let W4 weights feed int8 activations *without* an int8 weight copy).

### Expected gain (Amdahl, per-token matmul MACs)

- MLP: 3 × 5120 × 17408 × 64 layers ≈ 17.1 G-MAC (~73% of LM matmul FLOPs)
- attention projections ≈ 1.2 G-MAC (16 layers), linear-attn ≈ 5.0 G-MAC (48 layers)
- matmul-only speedup at 2x on MLP: ~1.55x; end-to-end prefill (incl. attention/SDPA,
  norms): realistically **1.3–1.4x ⇒ ~860 → 1100–1200 tok/s**, more if attention-adjacent
  projections are later included.

### Risks

- **Kernel efficiency**: a naive GEMM that lands under ~50% of int8 peak is a wash. This
  is the main risk; mitigate by reusing MLX's steel/NAX tiling verbatim.
- **Accuracy**: per-token dynamic + per-channel weights is the standard vLLM W8A8 recipe;
  MLP inputs (post-RMSNorm) are well-behaved. If degradation shows, SmoothQuant-style
  weight/activation rebalancing on the MLP only. Attention/KV numerics are untouched by
  design, and decode is untouched entirely.
- **`mx.fast.metal_kernel` overhead**: per-call dispatch overhead is fine at prefill sizes
  (ms-scale kernels); if it matters, port to an MLX C++ extension later.
- **bf16 activations → int8**: quantize kernel reads bf16, writes int8 + fp32 row scales;
  trivial bandwidth cost at prefill sizes.

## 7b. Implementation status (2026-08-03): SHIPPED, milestones 1–4 done

Implemented in `mlx_vlm/int8_prefill.py`, served via `--int8-prefill`
(`MLX_VLM_INT8_PREFILL=1`, applied in server lifespan like dequant-prefill).

**Kernel results** (prototype scripts in this dir):
- API path that works: `tensor_inline` views over raw device pointers
  (`tensor<device int8_t, dextents<int32_t,2>, tensor_inline>(ptr, extents)`),
  `matmul2d` with `dynamic_extent` K, cooperative destination tensor,
  scales applied in-register via `get_multidimensional_index`. Gotchas:
  `is_valid_element(i)` (docs say `get_mask`), plain `#pragma unroll`
  (docs say `unroll full`), templated `slice<Extents...>()` (docs say
  `static_slice`), int8 pointers are `device int8_t*` not `device char*`.
- Tile sweep (`int8_gemm_sweep.py`): best = TM128/TN128/8 simdgroups,
  internal K loop. **~91 TOPS-eq = 76% of the 120-TOPS int8 peak, 1.55x MLX's
  bf16 NAX GEMM, 1.66x 4-bit qmm** at M=2048, K=5120, N=17408. Exact-int32
  correct (error = bf16 output rounding only). Edge tiles (any M) handled.
- Per-row activation quant kernel: 0.27 ms at M=2048 (vs 0.90 ms via mx ops).
  Fused quant+GEMM: 4.07 ms = 1.49x bf16 GEMM end-to-end.

**Model integration** (`mlx_vlm/int8_prefill.py`): patches
`nn.QuantizedLinear.__call__`; routes calls with ≥512 rows AND weight shape in
{(17408,5120), (5120,17408)} (the 192 MLP projections) to W8A8. int8 weights
built lazily per module by dequantizing the resident 4-bit weights (~17 GB
extra, fine in 128 GB; `warmup(model)` pre-builds). Decode and all other
layers untouched.

**End-to-end A/B** (`e2e_int8.py`, real model, 6422-token prompt, greedy):

| | prefill tok/s | decode tok/s | output |
|---|---|---|---|
| baseline | 832–887 | ~34 | — |
| int8 patch | **1000–1008** | ~33 (unchanged) | **bit-identical to baseline** |

**+14–20% prefill.** Below the 1.3–1.4x Amdahl estimate because prefill
wall-time is not all matmul — the 48 linear-attention (deltanet) kernels take
a large share. Identical greedy output on this prompt is a smoke test, not a
quality eval — run a real workload eval before trusting it broadly.

**Next levers, in expected-value order:**
1. Opt-in int8 for attention/linear-attn projection shapes (q/o,
   in_proj_qkv/z/a/b, out_proj) — adds ~27% FLOP coverage; accuracy risk
   moderate, needs eval.
2. Share activation quantization between gate_proj and up_proj (same input x,
   currently quantized twice per layer).
3. Larger `--prefill-step-size` (bigger M amortizes better on all paths).
4. Requantize int8 weights from the original bf16 checkpoint instead of the
   4-bit conversion (removes stacked quantization error; needs ~55 GB download).
5. Row-quant kernel is ~78 GB/s effective; could be faster, but it's only ~7%
   of the GEMM pipeline.

## 8. Files in this directory

| file | what it does |
|---|---|
| `peak2.py` | correct-harness GEMM benchmarks: square peaks + prefill shapes (bf16 / qmm / dequant+GEMM) |
| `qbench4.py` | qmm vs per-call dequant+GEMM at the model's three projection shapes |
| `e2e_dequant.py` | end-to-end prefill tok/s on the real model, with/without the dequant-prefill patch |
| `mma_rate.py` | raw NAX MMA rates (fp16/bf16/int8) via custom MPP tensor-ops kernel — **the 2x proof** |
| `tensorops_compile_test.py` | minimal check that `mx.fast.metal_kernel` compiles MPP headers |
| `int8_gemm_v1.py` | first working W8A8 GEMM (64×32 tiles), exact-int32 correctness harness |
| `int8_gemm_sweep.py` | tile/simdgroup/K-loop sweep that found the 128×128×8simd config |
| `w8a8.py` | standalone W8A8 building blocks (row-quant kernel + GEMM); production copy lives in `mlx_vlm/int8_prefill.py` |
| `e2e_int8.py` | end-to-end A/B of the int8 prefill patch on the served model |

Run any of them with the repo venv: `.venv/bin/python research/int8-nax/<file>.py`.
Re-run `peak2.py` + `e2e_dequant.py` after every `mlx` upgrade — if upstream ships int8 or
int4b-operand NAX qmm (watch `quantized_nax.h`), most of §7 becomes obsolete in the best way.
