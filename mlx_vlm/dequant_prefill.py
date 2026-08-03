"""Dequantize-on-the-fly prefill patch.

MLX's quantized matmul kernels (qmm/gather_qmm) are tuned for decode-sized
batches; at prefill sizes they run ~2x slower than bf16 GEMMs (see
qmm_bitsweep.py). This patch keeps quantized weights resident but, for
large-token calls only, dequantizes the layer's weights to bf16 transiently
and uses the plain GEMM path. Decode (small batches) keeps the quantized
kernels. Bit-exact output is not guaranteed (different accumulation order),
but the math is the same dequantized weights.

Usage: call apply() before loading the model. The server applies it at
startup when started with --dequant-prefill (MLX_VLM_DEQUANT_PREFILL=1).
"""

import mlx.core as mx
import mlx.nn as nn

from .models import switch_layers

# per-call thresholds, in rows of the matmul (tokens, or token-expert
# assignments for the switch layers); below these the quantized kernels win
SWITCH_MIN_ROWS = 4096  # ~410 prompt tokens at top-10 routing
LINEAR_MIN_ROWS = 512
# skip huge output dims (lm_head): dequantizing 100k x 3072 per chunk is waste
LINEAR_MAX_OUT = 32768


def _dequant(m: nn.Module) -> mx.array:
    return mx.dequantize(
        m["weight"],
        m["scales"],
        m.get("biases"),
        group_size=m.group_size,
        bits=m.bits,
        mode=getattr(m, "mode", "affine"),
    )


def apply():
    qsl_orig = switch_layers.QuantizedSwitchLinear.__call__

    def qsl_call(self, x, indices, sorted_indices=False):
        if x.size // x.shape[-1] < SWITCH_MIN_ROWS:
            return qsl_orig(self, x, indices, sorted_indices)
        w = _dequant(self)
        out = mx.gather_mm(
            x,
            w.swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            out = out + mx.expand_dims(self["bias"][indices], -2)
        return out

    switch_layers.QuantizedSwitchLinear.__call__ = qsl_call

    ql_orig = nn.QuantizedLinear.__call__

    def ql_call(self, x):
        if (
            x.size // x.shape[-1] < LINEAR_MIN_ROWS
            or self["weight"].shape[0] > LINEAR_MAX_OUT
        ):
            return ql_orig(self, x)
        out = x @ _dequant(self).T
        if "bias" in self:
            out = out + self["bias"]
        return out

    nn.QuantizedLinear.__call__ = ql_call
