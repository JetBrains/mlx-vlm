# From stored 4-bit weights to int8/int4b GEMMs: what the numbers actually mean

A ground-up explanation of how the model's weights are stored, what 4-bit
affine quantization is, how the int8 NAX prefill path re-encodes weights, and
why the `int8 × int4b` hardware path (§7c of README.md) runs into a *scale
granularity* problem even though its integer arithmetic is identical to the
int8 path.

Everything is grounded in the model we serve: `Qwen3.6-27B-4bit`
(hidden 5120, MLP intermediate 17408, quantization `{group_size: 64, bits: 4,
mode: affine}`).

---

## 1. The starting point: what a linear layer stores

A linear layer is `y = x @ W.T`: it maps a 5120-long activation vector to,
say, 17408 outputs. Its weight matrix `W` has shape `[N, K]` = `[17408, 5120]`
— about 89 million real numbers, one row per output channel.

Trained weights are real numbers like `0.0123`, `-0.0089`, `0.2711`. The
natural storage is bf16 (2 bytes each). For the whole 27B-parameter model
that's ~55 GB — too much to be comfortable, and (more importantly for decode
speed) too many bytes to stream from memory for every generated token. That's
the entire motivation for quantization: **store each weight in fewer bits,
accepting a controlled rounding error.**

## 2. What "4-bit quantization" actually means

4 bits can encode only the integers 0..15 — sixteen distinct values. A
quantizer maps each real weight onto this tiny grid and remembers how to map
back. The **affine** scheme used here reconstructs a weight as:

```
w_real ≈ q · s + b        q ∈ {0, 1, ..., 15}
```

- `q` — the stored 4-bit code
- `s` — the **scale**: the step size between adjacent grid points
- `b` — the **bias** (offset): where grid point q=0 sits

Encoding is the reverse: `q = round((w_real − b) / s)`, clamped to 0..15.

Concrete micro-example. Say a bunch of weights lie in the range
`[−0.11, +0.19]`. Choose `s = (0.19 − (−0.11)) / 15 = 0.02` and `b = −0.11`.
Then the representable values are exactly:

```
q:      0      1      2      3     ...    15
value: -0.11  -0.09  -0.07  -0.05  ...  +0.19
```

A real weight `0.0123` becomes `q = round((0.0123+0.11)/0.02) = 6`, which
decodes back to `6·0.02 − 0.11 = 0.01`. The rounding error (here 0.0023) is
at most `s/2` per weight. **The quality of 4-bit quantization is entirely
determined by how small you can make `s`** — and `s` is forced by the spread
of the values it must cover: `s = (max − min) / 15`.

## 3. Why groups of 64: one scale can't fit everybody

If a single `(s, b)` pair had to cover the *entire* weight matrix, `s` would
be sized by the most extreme weight anywhere, and the millions of small
weights would all collapse onto a couple of grid points. Real weight matrices
have wildly non-uniform ranges — one region of a row might live in ±0.01
while another has spikes at ±0.3.

The fix is to give each small **group** of weights its own `(s, b)`. MLX's
`group_size: 64` means: walk along a row (the K dimension) and cut it into
consecutive chunks of 64 weights; each chunk gets its own scale and bias,
chosen from just those 64 values.

For our 5120-long rows that's **80 independent (s, b) pairs per row**. The
group in ±0.01 gets a step of `0.02/15 ≈ 0.0013`; the group with ±0.3 spikes
gets a step of `0.04`. Neither ruins the other. This is the whole reason
4-bit works at all: *16 levels is enough only because each 64-weight
neighborhood gets its own 16 levels.*

The trade-off is bookkeeping: the scales and biases are extra data stored in
bf16 alongside the codes.

## 4. How this is physically laid out on disk / in memory

For a quantized linear layer, the safetensors file holds three tensors:

| tensor | shape | dtype | contents |
|---|---|---|---|
| `weight` | `[N, K/8]` | uint32 | the 4-bit codes, 8 per 32-bit word |
| `scales` | `[N, K/64]` | bf16 | one `s` per group |
| `biases` | `[N, K/64]` | bf16 | one `b` per group |

Packing order inside each uint32 word: **low nibble first**. Bits 0–3 hold
element `8i`, bits 4–7 hold element `8i+1`, … bits 28–31 hold element `8i+7`
(this is what the requant kernel's `(wrd >> (4*j)) & 0xF` walks over, and —
verified in `int4b_gemm.py` — it's also exactly the order the hardware's
`int4b_format` expects). A 64-weight group spans exactly 8 consecutive words,
so no word ever straddles two groups.

Cost accounting per weight: 4 bits of code + (16+16) bits of scale/bias
shared by 64 weights = **4.5 bits/weight**, ~15 GB for the model versus ~55 GB
in bf16.

One more fact that matters later: the codes are **unsigned** (0..15) and the
signedness of the actual weight lives in `b` (typically `b ≈ −7.5·s`, roughly
centering the grid on zero — but per-group, not exactly).

## 5. How this is normally computed with (the qmm kernel)

This is *weights-only* quantization: activations stay bf16. MLX's quantized
matmul (`qmm`) never materializes a dequantized copy — inside the kernel,
each tile of packed codes is unpacked, `q·s + b` applied on the fly in
on-chip memory, and the multiply-accumulate happens in **float**. On M5 the
float math runs on the NAX units, and the in-kernel dequant overlaps with it
well enough that qmm is only ~8% slower than a pure bf16 GEMM at prefill
shapes (README §3.2).

Note the group structure costs nothing here: dequantization happens *before*
the multiply, element by element, so each weight simply uses its own group's
`(s, b)`. The scale-granularity problem of the next sections does not exist
on the float path. The float path's limitation is different: it can't use the
int8 datapath, which is 2× faster (README §3.3).

## 6. The int8 prefill path: re-encoding onto a per-channel grid

To use the 2× int8 NAX datapath, both operands must be int8 *integers*, and
the hardware accumulates raw integer products into int32. The products have
no idea about scales — so all scaling must be applied *after* the sum, which
constrains where scales are allowed to live (next section).

What `int8_prefill.py` feeds the hardware:

**Activations** — quantized on the fly, one scale per row (= per token):

```
s_x[m] = max(|x[m, :]|) / 127          x_q[m, k] = round(x[m, k] / s_x[m])
```

Symmetric (no bias), 8 bits, computed by a tiny kernel per prefill call.

**Weights** — this is the step that needs unpacking in detail.

### 6.1 Why the weights must be re-encoded at all

The hardware multiplies *integers* and knows nothing about scales. Whatever
integers we hand it, we must be able to turn the accumulated int32 back into
real numbers afterwards — and (as §7 proves) that back-conversion can only
apply **one scale per row**. The stored 4-bit codes don't qualify: code `9`
in group 3 and code `9` in group 40 mean *different real values* (each
group has its own `s` and `b`). Feeding raw codes to the hardware would add
apples to oranges inside the accumulator, and no epilogue could fix it.

So we need new integers, on a grid where one number per row tells the whole
story:

```
w_real ≈ w_q8 · s_w[n]        w_q8 ∈ {−127..127},  ONE s_w per row n
```

Getting there is a **decode → re-encode** round trip, done by the fused
requant kernel for every weight:

1. **Decode** the stored code to its real value: `w_real = q · s[g] + b[g]`,
   using *its own group's* scale and bias. (This is exact — it's the same
   reconstruction the qmm float path uses.)
2. **Re-encode** that real value on the new per-row grid:
   `w_q8 = round(w_real / s_w[n])`, clamped to ±127.

where the row scale is sized so the row's biggest weight lands on ±127:

```
s_w[n] = max over row n of |w_real| / 127
```

(The kernel gets this max without a separate pass: for a group with scale s
and bias b, the largest |decoded value| is `max(|b|, |15·s + b|)` — the two
ends of the group's grid — so the row max is the max of that expression over
the row's 80 groups. ~10 MB of these row scales are kept permanently.)

### 6.2 A worked example: one row, a loud group and a quiet group

Take one output row whose 80 groups include these two:

- **Group A (quiet):** weights in `[−0.010, +0.010]`.
  Stored grid: `s_A = 0.020/15 ≈ 0.00133` — sixteen points, 0.00133 apart.
- **Group B (loud):** weights in `[−0.30, +0.30]`.
  Stored grid: `s_B = 0.60/15 = 0.04` — sixteen points, 0.04 apart.

The row's biggest |weight| is 0.30, so the new int8 grid has step

```
s_w = 0.30 / 127 ≈ 0.00236       255 points: −0.300, −0.29764, ..., +0.300
```

Now re-encode a weight from each group and watch the error:

**Loud weight** `w_real = 0.28` (a point on group B's grid):
`w_q8 = round(0.28 / 0.00236) = 119`, which decodes to `119 · 0.00236 =
0.28084`. Added error ≈ 0.0008. Group B's *own* grid step was 0.04 — the
int8 grid is **17× finer** than what this weight was already rounded to.
The re-encoding is essentially transparent for it.

**Quiet weight** `w_real = 0.0067` (a point on group A's grid):
`w_q8 = round(0.0067 / 0.00236) = 3`, decoding to `0.00708`. Added error
≈ 0.0004. Group A's own step was 0.00133, so its weights were already
carrying rounding errors up to ±0.00067. The int8 grid's step (0.00236) is
coarser than group A's — about **1.8×** — so the quiet group does lose a
little precision. But 1.8× coarser than an already-fine grid is a small,
bounded nudge: every group-A weight moves by at most `s_w/2 ≈ 0.0012`, on
weights the original format located to within ±0.00067 anyway.

Picture the three grids around zero (not to scale):

```
group B grid (step .04):     |———————|———————|———————|———————|
int8 row grid (step .0024):  |·|·|·|·|·|·|·|·|·|·|·|·|·|·|·|·|   ← finer than B, slightly coarser than A
group A grid (step .0013):   |.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|.|
```

The int8 grid sits *between* the two originals: much finer than the loud
group's grid, slightly coarser than the quiet group's. That's the trade the
requant makes, row by row, for all 80 groups at once.

### 6.3 The general rule: when is one int8 row-grid "fine enough"?

Compare step sizes. A group whose values span `±g` has step `2g/15`. The row
grid, sized by the row max `R`, has step `R/127`. The row grid is *finer*
than that group's own grid whenever

```
R/127 < 2g/15    ⟺    R/g < 254/15 ≈ 17
```

So: **every group whose range is within ~17× of the row's loudest weight
gets a strictly finer grid than it originally had** — for those weights the
requant adds essentially nothing (they were already rounded more coarsely).
Only groups quieter than 1/17th of the row max come out coarser, and even
for them the absolute error stays ≤ `R/254` — about 0.4% of the row's
largest weight, on weights that are themselves tiny and were already noisy
at the ±half-step level. Trained MLP rows rarely have >17× spread between a
group's range and the row max, which is why the collapse is near-lossless in
practice (the e2e smoke test stayed bit-identical to baseline).

Contrast with the naive alternative — quantizing straight to 4 bits with one
row scale: step `2R/15`, which is 17× *coarser* than int8's `R/127`. A quiet
group at `R/g = 10` would have step `2R/15` against values spanning `±R/10`
— barely 1.5 usable levels, i.e. its weights collapse to {−step, 0, +step}.
That's precisely the int4b problem of §8; the int8 path avoids it purely by
having 254 levels instead of 15 to spread across the row.

Two more details worth stating explicitly:

- **The errors stack, but the stack is dominated by the first term.** Total
  error = (original 4-bit group-affine error, already baked into the model
  everyone serves) + (requant re-rounding, ≤ half an int8 step). The second
  term is the small one; the requant does not "re-quantize to 8 bits from
  scratch", it re-rounds values that already sit on a coarse grid onto a
  usually-finer grid.
- **Symmetric on purpose.** The new grid has no bias `b`. A per-row bias
  would actually still be factorable (see §7's row-sum remark), but symmetric
  keeps the epilogue to two multiplies, and int8 has enough levels that
  wasting a bit of range on asymmetry costs nothing measurable.

An analogy, if it helps: the stored format is 80 short tape recordings, each
made at its own volume knob setting so that quiet passages and loud passages
both use the tape's full range. The requant re-records them all onto one
higher-fidelity tape at a single volume setting. The loud passages fit
easily; the quiet ones lose a whisper of resolution because the new medium
is good enough to afford a single setting. Re-recording onto another *low*
fidelity tape at one volume (the int4b temptation) is where the quiet
passages drown.

## 7. Where scales are allowed to live: the factoring argument

Now the core algebra. The hardware computes, for output cell (m, n):

```
acc[m, n] = Σₖ  x_q[m, k] · w_q8[n, k]          (int32, exact)
```

The real value we want is:

```
y[m, n] = Σₖ  (x_q[m,k] · s_x[m]) · (w_q8[n,k] · s_w[n])
```

Both scales are **constant with respect to k**, the summation index. Constants
factor out of sums:

```
y[m, n] = s_x[m] · s_w[n] · Σₖ x_q·w_q8  =  s_x[m] · s_w[n] · acc[m, n]
```

So the epilogue is two multiplies per output cell (`float(cT[i]) * xs[m] *
ws[n]` in the kernel) — essentially free. This only worked because the scale
had **at most per-row / per-column granularity**. The moment a scale varies
*along k* — which is exactly what group-64 means — it's trapped inside the
sum:

```
Σₖ x_q[k] · w_q4[k] · s[n, k/64]      ← s changes every 64 terms
  = s[n,0]·(partial sum of k=0..63) + s[n,1]·(k=64..127) + ... + s[n,79]·(k=5056..5119)
```

You'd need the 80 partial sums *separately*, but the hardware hands you one
merged int32 per cell. The information is gone. (Recovering it would mean 80
chunked GEMM accumulations of K=64 each — far too fine to keep the MMA
pipeline busy; that's option (b) in §9.)

The same argument explains why everything is **symmetric** (bias-free) on the
integer path. With an affine weight `q·s + b`, the product expands to
`x_q·q·s + x_q·b`, and the `b` term drags `Σₖ x_q[m,k]` (a per-row activation
sum) into the result. That's actually *fixable* — one extra reduction per
row, and the matmul2d API even provides
`get_row_reduction_destination_cooperative_tensor` for it — but per-**group**
biases `b[n, k/64]` re-break it the same way group scales do: you'd need
per-group activation sums, not one per row.

## 8. The int4b path and where the tension appears

Metal's `matmul2d` supports `int8 × int4b_format → int32`: the left operand
is int8 activations exactly as today, the right operand is **packed 4-bit
codes read directly from memory** (the hardware unpacks in the load path).
Benchmarked in `int4b_gemm.py`: bit-exact, and at its best tiling (TM=96)
equal to or *faster* than our int8×int8 GEMM — halving the weight-fetch
bandwidth pays for the unpacking. It would also delete the per-chunk requant
pass and its transient int8 copies.

But look at what the hardware demands of the operand: plain **signed 4-bit
two's complement integers** (−8..7), participating in one long int32
accumulation. By §7, whatever scale reconstructs them must be constant along
k — one scale per row. So to use int4b we must re-encode the weights onto:

```
w_real ≈ w_q4 · s_w[n]        w_q4 ∈ {−8, ..., 7},  one s_w per row
```

That is a **16-level grid stretched over an entire 5120-long row** — strictly
coarser than the stored format's 16 levels *per 64-weight group*. Redo the
§6 level-counting with 4 bits: if the quiet group's range is ~10× below the
row max, it now lands on ~1.5 levels. Its weights all collapse to −1/0/+1
steps of the row-wide grid. That is real, potentially serious quantization
error — added on top of the error the model already carries.

Compare the three encodings side by side for one row whose groups span
±0.01 … ±0.3:

| encoding | levels for the ±0.3 group | levels for the ±0.01 group |
|---|---|---|
| stored: 4-bit, per-group scale | 16 (step 0.04) | 16 (step 0.0013) |
| int8 path: 8-bit, per-row scale | 255 (step 0.0024) | ~8 usable (step 0.0024) |
| int4b path: 4-bit, per-row scale | 15 (step 0.04) | **~0.5 usable** (step 0.04) |

The int8 row's "~8 usable levels" is already the borderline case that
happened to be fine in practice; int4b pushes past it by another 16×.

## 9. Options, and the actual decision

1. **(a) Per-channel symmetric int4 sidecar.** One-time repack (cheap, same
   spirit as today's requant), then the fast int4b GEMM with the trivial
   epilogue. The open question is purely §8's quality loss. Note it might be
   smaller than the table's worst case suggests — MLP rows after training are
   often not that dynamic, and outlier-heavy rows could stay on the int8
   path per-layer. **Go/no-go = an eval**: cosine / max-abs-err of layer
   outputs vs the current path on real captured inputs (milestone-2 style),
   then an end-to-end quality check.
2. **(b) Group-chunked accumulation.** Keep group-64 scales by running the
   K reduction in 80 scaled chunks. Preserves quality by construction —
   **measured (2026-09-03, `int4b_chunked.py`) and ruled out**: at the
   M=2048 gate/up shape, C=64 chunks run at **7.0 TOPS** (14× slower than
   single-shot int4b), C=128 at 12.4, C=256 at 20.4, C=512 at 32.2, C=1024
   at 45.5. Two separate costs: the static-K descriptor + fp32 side
   accumulator alone drop a *single*-chunk run from 98 to ~70 TOPS, and each
   additional `op.run` boundary costs a near-constant ~0.56 ms (re-staging /
   simdgroup barriers inside MPP), which at 80 chunks swamps everything.
   Even the coarsest tested chunking (C=1024 = 16 groups per scale, already
   most of the way to per-row quality-wise) is ~2× slower than today's
   requant+int8 path. No viable point on this curve.
3. **(c) Status quo.** Requant to per-channel int8 (near-lossless, §6), eat
   the ~0.2 s/pass requant and the transient copies. This is what ships
   today and it already delivers +30–47% prefill.

The int4b hardware itself is proven good (§7c of README.md). Whether it ships
comes down to measuring (a)'s accuracy — arithmetic speed is no longer the
question.

### 9.1 Two more escape routes, checked and closed

**Parallelize over groups instead of serializing (option b′).** Option (b)
kept the 80 group-partials in registers and paid ~0.56 ms per sequential
`op.run`. The restructured version makes the group index a *grid* dimension:
launch 80 independent `[M,64] × [64,N]` GEMMs (each still a fat matrix at
prefill M, so tiling is fine) and combine `Σ_g s_w[n,g] · P_g[m,n]` in a
second kernel. The boundaries disappear — but now the partials materialize.
For one gate/up call at M=2048, each `P_g` is `[2048, 17408]` int32 =
143 MB; × 80 groups = **11.4 GB written + 11.4 GB read back** by the
reduction, ~42 ms at the M5 Max's ~546 GB/s — vs 3.7 ms for the single-shot
GEMM whose entire output is 71 MB. Bandwidth-bound at roughly the same
~8 TOPS-eq where option (b) is overhead-bound. (K=64 GEMMs also collapse
arithmetic intensity: the C-tile traffic a deep-K GEMM amortizes over 5120
accumulation steps gets paid every 64.) Large M doesn't help — the
intermediate scales with M·N, inflating exactly as fast as the useful work.

The full picture is symmetric. The weighted sum over 80 group-partials must
happen *somewhere*:

| where the 80 partials live | binding cost | throughput |
|---|---|---|
| registers, sequential (b) | 80 × ~0.56 ms `op.run` restarts | 7 TOPS (measured) |
| DRAM, parallel (b′) | 22.8 GB round-trip per layer call | ~8 TOPS-eq (bandwidth math) |
| inside the MMA datapath | — | NVFP4/Blackwell (§10); no such port on M5 |

**Fold the group scales into the activations?** This would be the free
lunch, and it fails for one precise reason. If the group scales were shared
by all output channels — `s[g]`, no n index — they *could* be folded into
the left operand: pre-multiply `x[m,k]` by `s[g(k)]` (one cheap pass over
the small M×K activation matrix, before activation quantization), and the
plain unscaled int4 GEMM plus the per-row epilogue would be exact. The whole
group structure would ride in through the activations for free.

But the scales are `s_w[n, g]` — **each of the 17,408 output channels has
its own 80** (they were chosen per 64-weight group *of that row*, §3). A
product term `x_q[m,k] · w_q[n,k]` therefore needs the factor
`s_w[n, g(k)]`, which depends on both the summation index k and the output
index n. One physical activation value `x[m,k]` is multiplied against
17,408 different channels, each wanting a different scale on that term — a
single pre-scaled copy of x can't wear 17,408 scales. (The reverse fold
fails symmetrically: `s_x[m]` is per-token and one weight serves every
token, so weights can't absorb activation scales either.)

General rule, worth stating once: **per-row × per-column is the largest
scale structure a scale-blind integer GEMM can honor** — anything finer
varies along K for someone, gets trapped inside the accumulator (§7), and
can only be honored inside the datapath (§10).

## 10. Postscript: NVFP4 in this framework (what Blackwell does about all of the above)

The vLLM sibling project ([erokhins/vllm-qwen3.6-27b-nvfp4](https://github.com/erokhins/vllm-qwen3.6-27b-nvfp4),
commit b85e97d) got W4A4 prefill by flipping 192 config labels — no kernels.
NVFP4 is worth restating in this document's terms, because it is *exactly*
this document's problem with every hard part moved into silicon.

**The code grid (§2), but non-uniform.** NVFP4's 4-bit code is not an
integer 0..15 — it's **e2m1**, a miniature float: 1 sign bit, 2 exponent
bits, 1 mantissa bit. Its sixteen representable values are

```
±{0, 0.5, 1, 1.5, 2, 3, 4, 6}
```

— spacing 0.5 near zero, 2.0 out at the tail. Where our affine int grid
spends its 16 levels evenly, e2m1 spends them like weights are actually
distributed: densely near zero, sparsely on the rare large values. For
bell-shaped weight groups that's a better use of 4 bits than a uniform grid
(it's a hardware-friendly approximation of the same idea behind NF4).

**Groups (§3), but 16 instead of 64, and two-level scales.** Reconstruction
is `w_real ≈ code · s_group · S_tensor`: a per-**16**-element FP8 (E4M3)
scale, times one fp32 scale for the whole tensor. Two levels because an FP8
scale alone has too little range/precision to both size a group and span the
tensor's dynamic range; the fp32 global scale re-centers the FP8 scales into
their sweet spot. Note the granularity: 4× finer groups than our group-64
affine. Storage math (§4): 4 bits + 8 bits/16 = **4.5 bits/weight — the
same budget as MLX's group-64 affine** (which spends the overhead on
coarser-grouped but higher-precision bf16 scale+bias instead). No bias:
e2m1 is symmetric by construction, like our int8/int4 re-encodings.

**Activations quantized the same way.** W4A4 means the activations also get
e2m1 codes with per-16-group scales, computed on the fly (plus a
*calibrated* per-tensor global scale, the `input_scale` the weight-only path
loads and throws away). Compare our per-row int8: NVFP4 activations are both
narrower (4 bits) and finer-grouped (16 values per scale vs 5120).

**The §7 factoring wall — solved in silicon.** This is the punchline. The
whole reason our int8 path re-encodes per-row is that M5's `matmul2d`
accumulates raw integer products into one int32 and can only absorb scales
that are constant along K. Blackwell's tensor-core MMA instead **takes the
per-group scale factors as a third operand**: alongside the tiles of 4-bit
codes, it consumes tiles of FP8 scales, and internally computes

```
acc[m,n] = Σ_groups  s_x[m,g] · s_w[n,g] · ( Σ_{k∈g} x_code · w_code )
```

with the inner 16-element dot in FP4 and the group-scaled accumulation in
higher precision — i.e., **our option (b), group-chunked scaled
accumulation, executed inside the MMA instruction at full rate**. What cost
us 7 TOPS in software (80 `op.run` boundaries at ~0.56 ms each) is the
tensor core's native dataflow there. That's why the vLLM change needed no
requant, no per-channel compromise, no quality decision: the stored
group-16 grid rides through to the ALUs untouched.

Summary table, in this document's terms:

| | MLX stored format | our int8 path | our int4b option (a) | NVFP4 W4A4 |
|---|---|---|---|---|
| code | uint4, uniform | int8, uniform | int4, uniform | e2m1 float, non-uniform |
| weight scale granularity | 64 (scale+bias, bf16) | per-row | per-row | 16 (FP8) × per-tensor (fp32) |
| activations | bf16 (untouched) | int8, per-row | int8, per-row | e2m1, per-16-group |
| who applies group scales | dequant before multiply (float path) | nobody (collapsed to per-row) | nobody (collapsed to per-row) | **the MMA instruction** |
| quality cost of the fast path | — | ~none (§6.3) | open question (§8) | ~none by construction |

So NVFP4 isn't a cleverer answer to the scale-factoring problem — it's the
same answer we couldn't afford (option b), made free by hardware co-design.
The M5's NAX is one hardware generation behind that idea: it has the 2×
low-precision datapath but not the scale-aware accumulation.
