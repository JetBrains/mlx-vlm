# GDN chunked-prefill optimization — scouting

`bench.py` measures `gated_delta_chunked` (the fp32 chunked deltanet prefill
in `mlx_vlm/models/qwen3_5/gated_delta.py`) at real Qwen3.6-27B shapes,
plus a prototype variant with bf16 intra-chunk matmuls (state scan and the
C x C triangular solve stay fp32), plus a chunk-size sweep.

## M4 Max results (2026-08-05)

| C | fp32 | bf16-intra | bf16 speedup |
|---|---|---|---|
| **64 (prod)** | **25.0 ms** | 22.7 ms | 1.10x |
| 128 | 56.2 ms | 55.1 ms | 1.02x |
| 256 | 2889 ms | 3024 ms | — |

bf16 rel-err vs fp32 path: 1.1e-2 (bf16-epsilon scale; would need a quality
eval before shipping).

**GDN share of M4 prefill: ~6.5%** — 48 layers x 26 ms = 1.25 s per
4096-token chunk vs 19.1 s total (214 tok/s). With the bf16 variant at
1.10x, the end-to-end ceiling of this lever on M4 is <1%. **Not worth
pursuing on M4**: prefill there is dominated by the 4-bit GEMMs
(~72% of wall, already near the ~14 TFLOPS ALU ceiling — see
`research/m4-tuning/`). On M5 the same kernel is a ~25% share; see below.

C=64 is confirmed optimal; the O(C^2) triangular solve kills larger chunks.

## M5 Max results (2026-08-07) — this is where the lever is

| C | fp32 | bf16-intra | bf16 speedup |
|---|---|---|---|
| **64 (prod)** | **22.7 ms** | 20.3 ms | 1.12x |
| 128 | 51.2 ms | 48.8 ms | 1.05x |
| 256 | 166.4 ms | 165.2 ms | 1.01x |

NAX makes the GEMMs ~4x faster (60 TFLOPS bf16, `research/m4-tuning/prefill_paths.py`
on this box) while GDN barely moves: **22.7 ms/layer on M5 vs 25.0 on M4, a
1.1x gain against the GEMMs' 4x.** fp32 has no NAX datapath, the 64x128 tiles
sit far below peak, and the T/64-step inter-chunk scan is launch-latency-bound.

**GDN share of M5 prefill: ~23-26%** — 48 layers x 22.5 ms = 1.08 s per
4096-token chunk, against ~4.1 s of wall for that chunk at the measured
~1000 tok/s with `--int8-prefill` (~4.8 s at the 860 tok/s unpatched rate).
That is ~4x the M4 relative share, and it makes GDN the second-largest
consumer of M5 prefill after the quantized GEMMs.

Consequences for what to build:
- The bf16-intra-chunk dtype lever is still small: 1.12x on the kernel is
  ~2-3% end-to-end, and it costs 1.1e-2 rel-err (needs a quality eval).
- The structural lever is the one worth designing: the nC=64 sequential
  state-update steps at T=4096 are latency-bound, so batching/fusing that
  scan (or moving it into a Metal kernel like the decode path already has)
  attacks the bulk of the 1.08 s. This bench is the A/B harness for it.
- C=64 remains optimal on both chips, but note M5 degrades far more gracefully
  at C=256 (166 ms vs M4's 2889 ms — M4 falls off an occupancy/memory cliff
  the M5 does not).
