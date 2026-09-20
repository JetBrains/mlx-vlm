# Chunked parallel GDN delta scan on Metal: negative result

**Question.** Prefill profiling on M5 Max (junie replay request 10, 31,710
prompt tokens; see `research/junie-replay/profile_layers.py` and
`profile_gdn.py`) showed the fused gated-delta scan kernel takes 29% of GDN
prefill time ≈ **9% of total prefill (~2.8s of ~30s)**. The kernel
(`qwen3_5/gated_delta.py: gated_delta_kernel`) loops sequentially over all T
tokens of a 1024-token chunk inside one launch, parallel only over
48 heads × 128 Dv lanes — so the GPU is mostly idle along T. The chunked
parallel formulation `gated_delta_chunked` (already used for the CUDA path)
replaces the T-loop with intra-chunk GEMMs + a T/C inter-chunk scan. Does it
beat the fused kernel on Apple Metal?

**Answer: no.** At the exact Qwen3.6-27B GDN shapes (B=1, Hk=16, Dk=128,
Hv=48, Dv=128, T=1024, M5 Max, `bench.py` / `breakdown.py`):

| variant                                   | ms / 1024-token layer-chunk |
|-------------------------------------------|------|
| fused sequential kernel (current)          | **1.74** |
| chunked C=16                               | 5.23 |
| chunked C=32 (best stock C)                | 4.43 |
| chunked C=64                               | 5.48 |
| chunked C=128                              | 11.63 |
| chunked C=32 + blocked Tinv                | 4.04 |
| chunked C=32 + blocked Tinv + mx.compile   | **3.06** |

Numerics are not the issue: chunked y rel-err vs the fp32 ops reference is
3.4e-3 (the fused kernel itself is at 2.7e-3, both ~bf16 noise).

**Why chunked loses here.** Phase breakdown at C=32 (`breakdown.py`):
reshape 9%, gate/KK 8%, triangular inverse 19%, intra-chunk GEMMs 23%,
inter-chunk scan 41%. All of it is small batched fp32 GEMMs (C×Dk with
C=32..64) and a Python loop of nC=16..32 tiny state updates — as
launch/occupancy-bound as the thing it replaces, plus ~4x the memory traffic
(fp32 intermediates, [C,C] decay/attention matrices per head per chunk). The
~20x CUDA win quoted in the docstring is vs the *per-token ops loop*
(backends with no fused kernel), not vs this Metal kernel, which at
~1.7 µs/token of sequential chain is already fast.

Fixes tried:
- **Blocked (recursive) unit-lower-triangular inverse** instead of the
  C-row forward-substitution loop: helps (C=64: Tinv 2.45 ms → total
  3.9 ms), kept in `breakdown.py`, but not nearly enough.
- **mx.compile** over the whole chunked function: −25%, still 1.8x slower
  than the kernel.

Not tried, and why we stopped:
- Associative (log-depth) scan over the nC inter-chunk state recurrence:
  the state map per chunk is affine with a Dk×Dk matrix coefficient per
  head, so composing maps costs batched 128³ GEMMs per round — estimated
  ~1.5 ms alone, does not close a 1.3 ms gap on a path capped at 9% of
  prefill.
- Writing the chunked algorithm as one fused Metal kernel (parallel over
  chunks, sequential only across nC boundaries). This is the remaining
  credible idea, but it is a new-kernel project for a ≤9%-of-prefill /
  ≤3%-of-request ceiling; MLP GEMMs (48% of prefill) remain the better
  target per unit of effort.
- Wavefront overlap (start layers 0..k on chunk c+1 while deep layers
  finish chunk c): dependency-legal, but needs multi-stream execution and
  removing the per-chunk `mx.eval` the prefill loop and APC checkpointing
  rely on. Same ceiling; a gputrace check of actual GPU idleness during
  the scan should precede any attempt.

**Conclusion: keep the fused sequential kernel for the Metal prefill path.**

Files:
- `bench.py` — kernel vs chunked correctness (vs fp32 ops reference) and
  speed sweep over T and C.
- `breakdown.py` — phase timings inside the chunked scan, blocked-inverse
  and mx.compile variants.

Measured 2026-09-21, M5 Max 128GB, mlx in repo `.venv`, branch
`research/m5-gdn-chunked-scan`.
