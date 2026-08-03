"""Hybrid fp16 compute for M1-family GPUs (no native bfloat -> bf16 is
emulated; fp16 GEMMs are ~1.1x and SDPA ~1.3x faster, see kernel_probe.py).

Full-model fp16 overflows: Laguna's residual stream carries massive
activations peaking at ~9.4e5 (>> fp16 max 65504), created by the last
layers' block outputs. So:
  - the residual stream stays bf16 end-to-end (norms, embed, lm_head too);
  - attention/MoE block internals run fp16 for the first `num_fp16_layers`
    layers only — their inputs are RMSNorm-bounded, and measured bf16 block
    outputs there stay ~1e3 (60x fp16 headroom);
  - the tail layers (whose outputs create the 1e5+ spikes) stay fully bf16.

Composes with dequant_patch/overlap_patch (dequantize follows the scales
dtype, so fp16 layers dequantize straight to fp16).

Usage: call apply(model) AFTER loading (needs the model instance). The
server applies it when started with --hybrid-fp16 (MLX_VLM_HYBRID_FP16=1);
Laguna models only.
"""

import mlx.core as mx

from .models.laguna import language as laguna

NUM_FP16_LAYERS = 42  # layers 0..41 measured peak ~1.1e3; 43+ approach 6e4

_class_patched = False


def apply(model, num_fp16_layers: int = NUM_FP16_LAYERS):
    layers = model.language_model.model.layers
    for layer in layers[:num_fp16_layers]:
        layer.self_attn.set_dtype(mx.float16)
        layer.mlp.set_dtype(mx.float16)
        layer._fp16_blocks = True

    global _class_patched
    if _class_patched:
        return
    _class_patched = True

    dl_orig = laguna.DecoderLayer.__call__

    def dl_call(self, x, mask=None, cache=None):
        if not getattr(self, "_fp16_blocks", False):
            return dl_orig(self, x, mask, cache)
        r = self.self_attn(self.input_layernorm(x).astype(mx.float16), mask, cache)
        h = x + r.astype(x.dtype)
        r = self.mlp(self.post_attention_layernorm(h).astype(mx.float16))
        return h + r.astype(x.dtype)

    laguna.DecoderLayer.__call__ = dl_call
