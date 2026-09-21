# Wavefront (pipelined) prefill on Metal: blocked by the runtime

**Idea.** During prefill, chunk *c+1* at layer *L* depends only on chunk
*c+1*'s output from layer *L−1* and layer *L*'s cache state left by chunk
*c* — so shallow layers could process chunk *c+1* while deep layers finish
chunk *c*. The GDN delta-scan kernel runs ~200k threads for ~1.7 ms/layer
(low occupancy, ~9% of total prefill idle-ish GPU; see
`research/gdn-chunked-scan/`), and the next chunk's GEMMs could in theory
fill those idle cores.

**Result: dead on arrival with MLX 0.32.2 — GPU streams do not execute
concurrently on Metal.** Two experiments (`overlap.py`, M5 Max):

1. 12 delta scans (19.0 ms alone) + 12 MLP-sized bf16 GEMMs (36.1 ms
   alone), independent, on one stream: 54.9 ms; on **two streams: 53.5 ms**.
   Full overlap would give ~36 ms; we recovered 2%.
2. The cleanest possible case — two fully independent low-occupancy scan
   chains: one stream 37.2 ms, **two streams 36.7 ms**. No concurrency at
   all.

So MLX streams order work but everything is serialized into the same GPU
execution timeline; Metal's own hazard-scoped concurrent dispatch is not
reachable from MLX today. Without kernel-level concurrency, no schedule of
chunks/layers can overlap anything, regardless of how legal the data
dependencies are.

**Upstream status (checked 2026-09-21).** What MLX shipped in 2026 is
*thread safety*, not GPU concurrency: mlx#3078 ("concurrent inference of
independent models") was closed 2026-04-24 with "MLX has thread safety
support in 0.31.2 now" — streams became thread-local and eval is safe from
multiple threads. We tested that exact pattern too (two threads, each with
its own stream created in-thread, mlx 0.32.2): 37.2 ms vs 38.0 ms
sequential — still zero GPU overlap, same for mx.async_eval on two streams
(37.2 ms). Maintainer expectation disagrees with measurement: awni wrote
(Jan 2026, #3078) that the two-stream pattern "will in theory run both
models in parallel up to the capacity of the GPU", and the docs describe a
Metal stream as its own command queue. Discussion #1956 explains the likely
serializer: MLX allocates buffers untracked
(ResourceHazardTrackingModeUntracked) and orders encoders with explicit
Fences. No open PR/WIP for cross-stream kernel concurrency was found; the
vllm-metal RFC #188 documents the same pain from the other direction.
Given the docs/maintainer claim vs observed behavior, `overlap.py` is a
ready-made repro for an upstream issue.

**What would change this:**
- MLX gaining truly concurrent GPU streams (or concurrent dispatch within
  a command buffer). Re-run `overlap.py` after MLX upgrades; if test 2
  ever shows ~2x, revisit — the prize is bounded by the scan time, ~9% of
  prefill, minus pipeline bubbles.
- Custom Metal encoding outside MLX (own command queues + shared buffers):
  far outside this codebase's maintenance budget for a ≤9% ceiling.

Note this is about intra-request latency. Cross-request throughput
batching (B=2 prefill) is unaffected by this finding — the scan kernel
parallelizes over the batch dimension natively.

Files: `overlap.py` — the two stream-concurrency tests.

Measured 2026-09-21, M5 Max, mlx 0.32.2, branch `research/m5-gdn-chunked-scan`.
