# M4 serving tuning — findings

**Measured:** 2026-08-05
**Machine:** Apple M4 Max, 16-core CPU (12P+4E), 64 GB, macOS 25.6 (Darwin 25.6.0)
**Software:** mlx 0.32.0, this repo, Python 3.13 venv at `.venv`
**Model:** `mlx-community/Qwen3.6-27B-4bit` + `Qwen3.6-27B-MTP-4bit` drafter

The serving stack was tuned on M5 Max (see `research/int8-nax/README.md`).
This doc re-establishes the numbers on M4 Max, which has **no GPU matrix
hardware** (NAX is M5+ only), and records the M4-optimal configuration the
launcher now selects automatically.

## Microbenchmarks (M=2048, MLP shape 5120→17408)

| path | ms | rate |
|---|---|---|
| bf16 GEMM | 26.4 | 13.8 TFLOPS |
| 4-bit qmm | 28.2 | 13.0 TFLOPS-eq |
| dequant → bf16 GEMM | 25.4 | 14.4 TFLOPS-eq |
| fused act-quant + int8 MPP GEMM | 26.7 | 13.7 TOPS-eq |

- The MPP tensor-op int8 kernels **compile and run correctly on M4** (rel
  err ~1.2%), but at exactly bf16-GEMM speed — the 2x int8 datapath is NAX
  hardware, which M4 lacks. `--int8-prefill` is pure overhead here.
- bf16 GEMM peak is ~13.8 TFLOPS vs ~56 on M5 Max NAX (~4x gap), so M4
  prefill is hard compute-bound near ~250 tok/s theoretical for a 27B dense
  model.
- Unlike M5 (qmm within 8% of GEMM), on M4 transient dequantization beats
  qmm — the `--dequant-prefill` patch, rejected on M5, wins here.

## End-to-end (real Junie session replay, 11 requests, APC warm)

| config | prefill tok/s | decode tok/s |
|---|---|---|
| M5-tuned baseline (int8-prefill, step 4096, MTP d3) | 191 | 29.2 |
| no prefill patch (plain qmm) | 193 | 28.8 |
| **dequant-prefill, step 4096, MTP d3 (shipped)** | **214** | **29.9** |
| dequant-prefill, step 8192 | 211 | 29.9 |
| dequant-prefill, step 2048 | 212 | 29.2 |
| dequant-prefill, no drafter | 213 | 14.9 |
| dequant-prefill, MTP block 4 | 199 | 25.2 |
| dequant-prefill, MTP block 2 | 202 | 28.1 |

Run-to-run noise is ~±3%; decode varies with generation length mix.

Conclusions:
- **`--dequant-prefill` replaces `--int8-prefill` on M1–M4**: +11% prefill.
- **Prefill step size: anything on the 1024–4096 plateau.** On the replay
  workload 2048/4096/8192 are within noise (per-request new-token chunks are
  mostly <4096). A dedicated cold-prefill sweep on a 14.4k prompt
  (`step_sweep.py`) shows a flat speed plateau across 1024–4096
  (253–255 tok/s) but rising peak memory (22.2 / 24.0 / 26.9 GB), a -2%
  dip at 8192 (31.7 GB) and a -12% cliff at 16384 (46 GB, brushing the
  55.6 GB GPU working-set limit). The shipped `prefill_step_size` of 1024
  sits at plateau speed with the lowest peak — worth keeping on 64 GB boxes,
  where APC sessions live alongside the model. (M5 has its own reason to
  prefer a larger step: it amortizes the int8 per-chunk weight requant.)
- **MTP speculation is a 2.0x decode win on M4** (14.9 → 29.9 tok/s), an
  even better ratio than M5's 1.45–1.6x, because base decode is slower
  while the ~3-token verify forward stays bandwidth-bound. Depth 3
  re-verified optimal (2 and 4 both slower), matching the M5 sweep
  (`research/mtp-overhead`).
- APC/seed-warmup behave identically to M5 (93% KV cached on the replay).
- **n-gram base window 4 re-verified optimal on M4** (sweep on the replay):
  base 2 never fires (drafts fall under MLX_VLM_NGRAM_MIN=3) → 28.9 tok/s,
  base 4 → 29.9, base 8 → 25.8. Same optimum as the M5 session tuning.

`mlx_vlm/server/junie/launch.py` picks the prefill patch by chip generation
(`machdep.cpu.brand_string`): the `int8_prefill` setting means "use the
fastest prefill path this machine has", and resolves to `--int8-prefill` on
M5+, `--dequant-prefill` on M1–M4, and the stock quantized kernels on an
unrecognized GPU.

## Raw prefill: every storage format measured (`prefill_paths.py`)

M=4096 at the MLP shape (5120→17408), M4 Max:

| path | TFLOPS-eq |
|---|---|
| bf16 GEMM / fp16 GEMM | 14.80 / 14.88 |
| **dequant(4b gs64) + bf16 GEMM (shipped)** | **14.57** |
| dequant(8b gs64 / gs32) + bf16 GEMM | 14.61 / 14.66 |
| qmm 4-bit gs64 | 13.07 |
| qmm 8-bit gs64 / gs32 | 12.93 / 12.84 |
| qmm mxfp8 gs32 | 12.70 |
| int8 MPP tensor-ops kernel | 13.7 (+ per-chunk requant cost) |
| gate/up fused into one 2N GEMM | 0.994x vs two GEMMs — no gain |

**8-bit (any flavor) does not help on M4.** Without integer matrix
hardware, storage format only changes unpack cost: every qmm pays a ~12%
in-kernel unpack penalty regardless of bits, every dequant+GEMM variant
sits at GEMM parity, and weights bandwidth is ~0.1% of chunk time
(compute-bound). The shipped dequant(4b) path is within ~2% of the best
possible GEMM rate on this GPU; the remaining e2e gap to the ~250 tok/s
ceiling is non-GEMM work (GDN 6.5%, SDPA, elementwise, small-chunk
requests), not the matmul format.

`alu_rate.py` establishes why: fp32 and fp16 FMA run at the same rate on M4
(no 2x half datapath), and `simdgroup_multiply_accumulate` is identical for
float and half operands, so MLX's fp32-accumulating steel GEMM is already at
~88% of the 15.7 TFLOPS simdgroup peak. There is no dtype lever left.

## Prefill in real usage: the seed checkpoint

Raw prefill rate on M4 is compute-ceiling-bound, so the real-usage wins are
in *avoiding* prefill:

- **The pinned seed never served real first requests.** Hybrid-model APC
  sessions resume only from recurrent-state checkpoints (`APCSession` in
  `mlx_vlm/apc.py`: full-attention K/V is sliceable, GDN state is not), and
  a request checkpoints only near its own full length — at
  `len - APC_EXACT_PREFIX_GUARD_TOKENS` (mid-prefill) and at `len`. A seed
  body captured from real traffic ends with a session-specific user message,
  so both checkpoints land *past* the point where every real first request
  diverges. Result: a new Junie session's first request paid the full
  stable-prefix prefill (~57 s for 14.5k tokens on M4) with `cached=0`.
- **Fix:** two-stage seed warmup — replay the seed's leading system
  message(s) alone first, so a pinned checkpoint lands at the cross-session
  stable boundary (that stage's unshared tail falls inside the 16-token
  guard), then replay the full seed. Measured on M4: new-session first
  request **57 s → 0.94 s** (cached 15076/15117), surviving server restarts
  via the disk tier (both stages restore in <1 s). Replay bench unchanged
  (93% cached).
- Steady-state fixed overhead is ~0.33 s per fully-cached request
  (template render + tokenize + APC lookup + schedule); worth profiling
  with py-spy if TTFT needs another shave.

The finding is architecture-independent — an M5 box with the same seed body
sees the same first-request win.

## Reproducing

The A/B harness used for these runs (`serve.sh`, a start.sh-flavoured
launcher) is gone with `start.sh`. Launch flags now come from
`server-config.json`, so an A/B is: edit the setting
(`int8_prefill`, `prefill_step_size`, `draft_model`, ...), `./serverctl.sh
stop && ./serverctl.sh start`, then `./bench.sh` to replay the captured
session (`research/junie-replay`). The microbenchmarks here are standalone:

    .venv/bin/python research/m4-tuning/prefill_paths.py
    .venv/bin/python research/m4-tuning/alu_rate.py
    .venv/bin/python research/m4-tuning/step_sweep.py   # loads the model
