"""Attribute the MTP verify-forward premium for Qwen3.6-27B (B=1).

Splits the cost of a 3-token verify forward into:
  - plain S=1 decode forward (reference)
  - plain S=3 forward, no GDN capture (T=3 matmul cost)
  - S=3 forward with GDN capture (speculative_verify_hidden — the real verify)
  - argmax head over 3 positions vs 1

Plus a microbench of one big QuantizedLinear and the lm_head at T=1 / T=3
via the stock path vs the custom target-verify kernel (which is currently
bypassed for B=1).
"""

import json
import os
import time

os.environ.setdefault(
    "HF_HUB_CACHE", "/Users/stanislav.erokhin/.local/share/junie-local/models"
)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx

from mlx_vlm.models import cache as cache_mod
from mlx_vlm.models.qwen3_5 import language as q35
from mlx_vlm.utils import load

MODEL = "mlx-community/Qwen3.6-27B-4bit"
REQ = "/tmp/req_small.json"
ITERS = 30


def timeit(fn, iters=ITERS, warmup=3):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return 1000.0 * (time.perf_counter() - start) / iters


def main():
    model, processor = load(MODEL)
    lm = model.language_model
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    msgs = [m for m in json.load(open(REQ))["messages"] if isinstance(m.get("content"), str)]
    ids = tok.encode(
        tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    )[:4096]
    pc = cache_mod.make_prompt_cache(lm)
    x = mx.array([ids], dtype=mx.int32)
    step = 2048
    while x.shape[1] > 0:
        n = min(step, x.shape[1])
        lm(x[:, :n], cache=pc, skip_logits=True)
        mx.eval([c.state for c in pc])
        x = x[:, n:]
    print(f"prefilled {len(ids)} tokens")

    x1 = mx.array([[42]], dtype=mx.int32)
    x3 = mx.array([[42, 43, 44]], dtype=mx.int32)
    x4 = mx.array([[42, 43, 44, 45]], dtype=mx.int32)

    def fwd_plain(xx):
        out = lm(xx, cache=pc, skip_logits=True, return_hidden=True)
        mx.eval(out.hidden_states[-1])
        return out.hidden_states[-1]

    def fwd_verify(xx):
        h, kv, gdn = lm.speculative_verify_hidden(xx, pc)
        mx.eval(h)
        return h

    t_s1 = timeit(lambda: fwd_plain(x1))
    t_s3 = timeit(lambda: fwd_plain(x3))
    t_s4 = timeit(lambda: fwd_plain(x4))
    t_v3 = timeit(lambda: fwd_verify(x3))
    t_v4 = timeit(lambda: fwd_verify(x4))

    h1 = fwd_plain(x1)
    h3 = fwd_plain(x3)

    def head_argmax(h):
        out = lm.speculative_argmax_from_hidden(h)
        mx.eval(out)

    def head_logits(h):
        out = lm.speculative_logits_from_hidden(h)
        mx.eval(out)

    t_am1 = timeit(lambda: head_argmax(h1))
    t_am3 = timeit(lambda: head_argmax(h3))
    t_lg1 = timeit(lambda: head_logits(h1))
    t_lg3 = timeit(lambda: head_logits(h3))

    print(f"forward S=1 plain            : {t_s1:7.2f} ms")
    print(f"forward S=3 plain            : {t_s3:7.2f} ms")
    print(f"forward S=4 plain            : {t_s4:7.2f} ms")
    print(f"forward S=3 verify (capture) : {t_v3:7.2f} ms")
    print(f"forward S=4 verify (capture) : {t_v4:7.2f} ms")
    print(f"argmax head T=1              : {t_am1:7.2f} ms")
    print(f"argmax head T=3              : {t_am3:7.2f} ms")
    print(f"logits head T=1              : {t_lg1:7.2f} ms")
    print(f"logits head T=3              : {t_lg3:7.2f} ms")

    # ---- projection microbench: stock vs custom verify kernel at B=1 ----
    layer0 = lm.model.layers[0]
    mlp_down = layer0.mlp.down_proj
    K = mlp_down.weight.shape[1] * 32 // mlp_down.bits
    for name, lin in (("mlp.down_proj", mlp_down), ("lm_head", lm.lm_head)):
        Kd = lin.weight.shape[1] * 32 // lin.bits
        for T in (1, 2, 3, 4):
            xt = mx.random.normal((1, T, Kd)).astype(lin.scales.dtype)
            mx.eval(xt)
            t_stock = timeit(lambda: mx.eval(lin(xt)), iters=50)
            out = q35._target_verify_quantized_linear(lin, xt)
            if out is not None:
                t_custom = timeit(
                    lambda: mx.eval(q35._target_verify_quantized_linear(lin, xt)),
                    iters=50,
                )
                custom = f"{t_custom:6.3f} ms"
            else:
                custom = "  n/a"
            print(f"{name:14s} T={T}: stock {t_stock:6.3f} ms | custom {custom}")


if __name__ == "__main__":
    main()
