# MTP speculative-decoding round overhead — Qwen3.6-27B-4bit + MTP drafter

Question: acceptance is 2.24 tokens/round, so target forwards drop ~2.2x, but
wall-clock speedup is only ~1.45-1.6x. Where does the difference go, and can
`--draft-block-size` or code changes reclaim it?

Setup: M5, 137 GB. Target `mlx-community/Qwen3.6-27B-4bit`, drafter
`mlx-community/Qwen3.6-27B-MTP-4bit` (block_size=3, 1 layer, uses the
target's embeddings + lm_head). Prompt: 7.4k-token Junie agent request,
greedy, 400 generated tokens. `bench.py` runs a B=1 copy of the server's
`_mtp_rounds_batch` loop; `verify_probe.py` isolates forward variants.
Run-to-run noise is ±5%; treat <2 tok/s deltas as ties.

## Results

Block-size sweep (fixed via `prefer_requested_block_size`):

| config      | tok/s | tokens/round | ms/round |
|-------------|-------|--------------|----------|
| baseline    | 29.0  | 1.00         | 34.5     |
| block=2     | 41.6  | 1.73         | 41.7     |
| **block=3** | 46.4  | 2.23         | 48.0     |
| block=4     | 42.3  | 2.43         | 57.5     |
| block=5     | 39.5  | 2.59         | 65.6     |
| adaptive 4-6| 46-49 | 2.23         | 45-48    |

**block=3 (the drafter's configured depth, i.e. the default) is optimal.**
The adaptive controller (`_effective_mtp_block_size`) never expands past 3
because the full-prefix hit rate stays below its 0.65 threshold — matching
the fixed-sweep result that deeper blocks lose. No start.sh change needed.

Per-phase breakdown at block=3 (mx.eval barrier after each phase):

| phase                          | ms/round | share |
|--------------------------------|----------|-------|
| verify fwd + argmax head       | 46.6     | 91%   |
| draft (1 drafter fwd + head)   | 2.2      | 4%    |
| accept_verified (1 fwd + head) | 2.3      | 4%    |
| walk + rollback + rebind       | 0.3      | <1%   |

Forward-variant probe (4k ctx):

| variant                              | ms    |
|--------------------------------------|-------|
| S=1 plain decode forward             | 32.7  |
| S=3 plain forward (no GDN capture)   | 36.9  |
| S=3 verify (GDN state capture)       | 39.3  |
| argmax/logits head, T=1 vs T=3       | ~1.6 / ~1.9 |

## Conclusions

- The drafter is NOT the overhead: ~4.5 ms/round ≈ 0.13 target forwards
  (2 drafter forwards; ~1.5 ms of each is the shared 250k-vocab lm_head
  read, which is memory-bound and irreducible).
- The gap is the **verify premium**: a 3-token verify forward costs ~1.35x a
  1-token forward. Round = 1.39 forward-equivalents → speedup ceiling
  2.23/1.39 = 1.60x, which is what we measure. Components:
  - +4.3 ms: T=3 quantized matmuls run ~15% below T=1 efficiency, spread
    over ~450 matmul dispatches (mlx qmm small-T path; the repo's custom
    B>1 verify kernels are no faster at B=1 — verified, bypass is correct).
  - +2.3 ms: GDN verify kernel writes fp32 intermediate states
    [B,T,Hv,Dv,Dk] per linear layer for rollback.
  - +1.9 ms: argmax head over 3 positions.
- Server layer adds no measurable per-round overhead (51 ms/round at 15k ctx
  ≈ bench 48 ms at 7.4k + ctx growth).
- Eliminating ALL non-verify overhead would gain <10%. Remaining levers are
  model-level, in rough order of expected value:
  1. KV quantization (`--kv-bits 8`) for long Junie contexts (verify reads
     the full-attention KV once per round; at 100k ctx that dominates).
  2. bf16 (instead of fp32) GDN intermediate-state capture: ~5% decode,
     small numerics risk after rollbacks.
  3. A deeper/tree drafter to raise tokens/round — the only path to >2x.

Acceptance visibility: the server now logs per-request
`Speculative decode: request=... kind=mtp rounds=N accepted_tokens_per_round=X accept_rate=Y%`
(`mlx_vlm/server/generation.py::_log_speculative_stats`), or `engaged=no` if
the drafter loaded but never ran.
