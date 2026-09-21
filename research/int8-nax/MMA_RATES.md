# NAX matmul2d rates: every supported type combination

**Date:** 2026-09-21
**Machine:** Apple M5 Max, 40-core GPU, macOS 27.2 (Xcode 27, Metal 4.1 — all spec Table 7.3 rows available, incl. int4/int2/fp8/fp4)
**Harness:** `mma_rate_all.py` (raw per-combo data: `mma_rate_all.json`)
**Baseline:** int8 × int8 → int32 = **117.8 TOPS** ≡ relative 1.00

Methodology: register-pressure-free rate test — A/B tiles staged once in threadgroup
memory as `tensor_inline` views (packed formats take a `uchar*` handle), 20 000
back-to-back `matmul2d` ops per simdgroup into a cooperative destination,
2048 threadgroups × 4 simdgroups, best of 4 tile shapes
{16×32×32, 32×32×32, 32×32×64, 64×32×32}. Sequential execution, idle GPU.
Cross-checks against the register-resident harness (`mma_rate.py`): int8 117.8 vs
121.7, fp16 63.4 vs 60.0 — within ±3%. int8 rows need the K=64 or M=64 tile to
peak; 16×32×32 loses ~20% on int rows (threadgroup-load bound).

Tables below use common names; the MSL/API spellings (what you write in the
kernel) are:

| common name | MSL / API name | what it is |
|---|---|---|
| fp16 | `half` | IEEE 754 half: 1 sign + 5 exp + 10 mantissa bits |
| bf16 | `bfloat` | bfloat16: 1 + 8 + 7 (fp32 range, less precision) |
| fp32 | `float` | IEEE 754 single precision |
| int8 / uint8 | `int8_t` / `uint8_t` | signed/unsigned 8-bit integer |
| int4 / uint4 | `int4b_format` / `uint4b_format` | packed 4-bit integer, 2 per byte — weight (B) operand only |
| int2 / uint2 | `int2b_format` / `uint2b_format` | packed 2-bit integer, 4 per byte — weight (B) operand only |
| fp8_e4m3 | `metal_fp8_e4m3_format` | 8-bit float, 1 + 4 + 3; clamps at ±448, no inf |
| fp8_e5m2 | `metal_fp8_e5m2_format` | 8-bit float, 1 + 5 + 2; wider range, overflows to ±inf |
| fp4_e2m1 | `metal_fp4_e2m1_format` | packed 4-bit float, 1 + 2 + 1, 2 per byte |
| int32 | `int32_t` | 32-bit integer (int accumulator) |

`fp32 (relaxed)` = same fp32 storage but with the descriptor's
`relaxed_precision=true`, which lets the unit truncate the mantissa before
multiplying (tf32-style) and is what routes fp32 onto NAX.

## Result: there are exactly four speed tiers

| tier | datapath | combos | TOPS | rel |
|---|---|---|---|---|
| 1 | **int8 MMA** | int8×int8, uint8×uint8 → int32 | 117.5–117.8 | **1.00** |
| 1b | int8 MMA + weight unpack | int8/uint8 × int2/int4 → int32 | 96–103 | **0.82–0.88** |
| 2 | **fp16 MMA** | *everything else* (fp16/bf16/fp8/fp32-relaxed/mixed, any accumulator) | 56–65 | **~0.50–0.55** |
| 3 | fp16 MMA + fp4 A-unpack | fp4×fp4 | 48–52 | **0.41–0.44** |
| 4 | fp32 SIMD ALUs (no NAX) | fp32×fp32 strict | 14.6 | **0.12** |

## Full table (relative to int8×int8 = 1.00)

| A | B | C | TOPS | rel |
|---|---|---|---|---|
| int8 | int8 | int32 | 117.8 | 1.00 |
| uint8 | uint8 | int32 | 117.5 | 1.00 |
| int8 | int2 | int32 | 103.1 | 0.88 |
| uint8 | uint2 | int32 | 100.9 | 0.86 |
| int8 | int4 | int32 | 99.9 | 0.85 |
| uint8 | uint4 | int32 | 96.1 | 0.82 |
| fp8_e4m3 | fp8_e4m3 | fp16 | 64.6 | 0.55 |
| fp16 | fp8_e4m3 | fp32 | 64.3 | 0.55 |
| fp16 | fp8_e5m2 | fp32 | 64.3 | 0.55 |
| fp8_e4m3 | fp8_e4m3 | fp32 | 64.3 | 0.55 |
| fp8_e5m2 | fp8_e5m2 | fp16 | 64.2 | 0.54 |
| fp16 | fp8_e4m3 | fp16 | 64.0 | 0.54 |
| fp16 | fp8_e5m2 | fp16 | 63.7 | 0.54 |
| bf16 | uint2 | fp32 | 63.6 | 0.54 |
| fp16 | fp16 | fp16 | 63.4 | 0.54 |
| fp16 | int2 | fp16 | 63.4 | 0.54 |
| fp16 | uint2 | fp32 | 63.4 | 0.54 |
| bf16 | fp16 | fp32 | 63.3 | 0.54 |
| bf16 | fp16 | fp16 | 63.2 | 0.54 |
| bf16 | uint4 | fp32 | 63.2 | 0.54 |
| fp16 | uint2 | fp16 | 63.2 | 0.54 |
| fp8_e5m2 | fp8_e5m2 | fp32 | 63.1 | 0.54 |
| bf16 | int4 | fp32 | 62.4 | 0.53 |
| bf16 | fp16 | bf16 | 62.3 | 0.53 |
| fp16 | bf16 | fp32 | 62.3 | 0.53 |
| bf16 | uint4 | bf16 | 61.9 | 0.53 |
| fp16 | int2 | fp32 | 61.9 | 0.53 |
| bf16 | uint2 | bf16 | 61.9 | 0.53 |
| bf16 | int2 | fp32 | 61.9 | 0.53 |
| fp16 | fp4_e2m1 | fp16 | 61.9 | 0.53 |
| fp16 | bf16 | bf16 | 61.8 | 0.52 |
| bf16 | fp32 | fp32 | 61.7 | 0.52 |
| bf16 | bf16 | bf16 | 61.6 | 0.52 |
| bf16 | bf16 | fp32 | 61.4 | 0.52 |
| fp16 | fp4_e2m1 | fp32 | 61.4 | 0.52 |
| fp16 | uint8 | fp16 | 60.9 | 0.52 |
| int8 | fp16 | fp16 | 60.8 | 0.52 |
| fp16 | int8 | fp16 | 60.7 | 0.52 |
| bf16 | int8 | fp32 | 60.5 | 0.51 |
| fp16 | fp16 | fp32 | 60.3 | 0.51 |
| fp16 | fp32 | fp32 | 60.3 | 0.51 |
| fp16 | bf16 | fp16 | 60.3 | 0.51 |
| fp16 | uint4 | fp32 | 60.2 | 0.51 |
| fp32 | fp16 | fp32 | 60.1 | 0.51 |
| uint8 | fp16 | fp32 | 60.0 | 0.51 |
| bf16 | uint8 | fp32 | 59.9 | 0.51 |
| fp16 | uint4 | fp16 | 59.9 | 0.51 |
| uint8 | fp16 | fp16 | 59.8 | 0.51 |
| uint8 | fp32 | fp32 | 59.8 | 0.51 |
| fp32 | bf16 | fp32 | 59.8 | 0.51 |
| fp16 | uint8 | fp32 | 59.7 | 0.51 |
| fp32 | fp32 (relaxed) | fp32 | 59.6 | 0.51 |
| uint8 | bf16 | fp32 | 59.5 | 0.51 |
| bf16 | int8 | bf16 | 59.3 | 0.50 |
| bf16 | int4 | bf16 | 59.2 | 0.50 |
| int8 | fp16 | fp32 | 59.1 | 0.50 |
| bf16 | int2 | bf16 | 58.9 | 0.50 |
| fp32 | int8 | fp32 | 58.7 | 0.50 |
| fp16 | int8 | fp32 | 58.5 | 0.50 |
| int8 | fp32 | fp32 | 58.4 | 0.50 |
| uint8 | bf16 | bf16 | 58.3 | 0.49 |
| fp16 | int4 | fp16 | 58.3 | 0.49 |
| fp16 | int4 | fp32 | 58.3 | 0.49 |
| bf16 | uint8 | bf16 | 57.7 | 0.49 |
| int8 | bf16 | fp32 | 57.6 | 0.49 |
| fp32 | uint8 | fp32 | 57.5 | 0.49 |
| int8 | bf16 | bf16 | 55.7 | 0.47 |
| fp4_e2m1 | fp4_e2m1 | fp16 | 52.0 | 0.44 |
| fp4_e2m1 | fp4_e2m1 | fp32 | 48.5 | 0.41 |
| **fp32** | **fp32 (strict)** | **fp32** | **14.6** | **0.12** |

## Findings

1. **Nothing beats int8×int8.** It is the only full-rate integer path; every
   float-involving combo (including fp8) runs at ≈½ rate.
2. **fp8 is a storage format, not a speed format, on M5** — fp8×fp8 = 0.55,
   i.e. exactly the fp16 datapath. Same for every fp16/bf16-mixed combo:
   mixing in an int8 or fp8 operand converts it to the float pipeline.
3. **fp4×fp4 is *slower* than fp16** (0.41–0.44) — the A-side unpack costs ~15%
   on top of the fp16 rate. There is no fp4 datapath.
4. **Packed weight formats on the int path keep most of the int8 rate:**
   W4A8 (int8×int4) = 0.85, W2A8 (int8×int2) = 0.88. This is the *instruction*
   rate with tiles resident in threadgroup memory — it isolates the in-datapath
   unpack tax (~15%) and hides bandwidth. **At the GEMM level the ranking
   flips:** with weights streamed from device memory, int8×int4 measured
   **+23%** over int8×int8 at the M=2048 gate/up shape (98.4 vs 79.7 TOPS,
   §7c of the README on the int8-nax branch history, commit `ef0b9dd`;
   int4 needs its own tiling, TM=96 not 128) because halved B-fetch bandwidth
   more than pays for the unpack. Both numbers are correct; they bracket the
   compute-bound vs bandwidth-bound regimes. The W4A8 integration blocker is
   scale granularity, analyzed to closure in §9.2–9.4 there (group-chunked
   accumulation 2–14× too slow; per-row int4 re-encoding +17–19% weight error;
   rank-1 scheme correct but speed-par — net win of full W4A8 is only the
   requant pass ~5% + memory, unless int4 weights are encoded from the
   original bf16 checkpoint).
5. **`relaxed_precision` fp32 = 0.51 vs strict fp32 = 0.12** — the relaxed flag
   is what routes fp32 through NAX (tf32-style); strict fp32 falls to the
   SIMD ALUs at 14.6 TOPS.
6. Accumulator width (fp16/fp32, bf16/fp32, int32) never changes the rate
   (±3% noise). Signedness never changes the rate. Which side (A/B) holds the
   mixed type doesn't matter.
7. Int rows are the hardest to feed: they only reach peak with 32×32×64 or
   64×32×32 tiles; at 16×32×32 the int8 rate drops ~20% (threadgroup-load
   bound), while float rows barely care. Real int8/W4A8 GEMM kernels should
   keep K-tiles ≥ 64.
