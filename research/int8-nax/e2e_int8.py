"""End-to-end A/B of the int8 NAX prefill patch on the real served model.

Loads Qwen3.6-27B-4bit, measures prefill tok/s and greedy output on a long
prompt, applies mlx_vlm.int8_prefill, and measures again. Greedy decoding
(temperature 0) makes the generated text directly comparable.
"""

import os
import sys

os.environ["HF_HUB_CACHE"] = (
    "/Users/stanislav.erokhin/.local/share/junie-local/models"
)
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, "/Users/stanislav.erokhin/IdeaProjects/mlx-vlm")

from mlx_vlm import apply_chat_template, generate, load  # noqa: E402

model, processor = load("mlx-community/Qwen3.6-27B-4bit")
config = model.config

text = (
    "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
    * 400
)
prompt = apply_chat_template(
    processor, config, f"Summarize the following in one sentence:\n{text}",
    num_images=0,
)


def run(tag, max_tokens=48):
    r = generate(model, processor, prompt, max_tokens=max_tokens, verbose=False)
    print(
        f"{tag}: prompt_tokens={r.prompt_tokens}  "
        f"prefill={r.prompt_tps:.1f} tok/s  decode={r.generation_tps:.1f} tok/s",
        flush=True,
    )
    return r


run("warmup     ", max_tokens=4)
a1 = run("baseline  1")
a2 = run("baseline  2")

from mlx_vlm.int8_prefill import apply, warmup  # noqa: E402

apply()
warmup(model)
run("patch warm ", max_tokens=4)
b1 = run("int8      1")
b2 = run("int8      2")

base = max(a1.prompt_tps, a2.prompt_tps)
pat = max(b1.prompt_tps, b2.prompt_tps)
print(f"\nprefill: {base:.1f} -> {pat:.1f} tok/s  ({(pat / base - 1) * 100:+.1f}%)")
print(f"\nbaseline output:\n{a1.text}")
print(f"\nint8 output:\n{b1.text}")
print(f"\noutputs identical: {a1.text == b1.text}")
