import os, sys
os.environ["HF_HUB_CACHE"] = "/Users/stanislav.erokhin/.local/share/junie-local/models"
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, "/Users/stanislav.erokhin/IdeaProjects/mlx-vlm")

from mlx_vlm import load, generate, apply_chat_template

model, processor = load("mlx-community/Qwen3.6-27B-4bit")
config = model.config

text = ("The quick brown fox jumps over the lazy dog near the riverbank at dawn. " * 400)
prompt = apply_chat_template(processor, config, f"Summarize in one word:\n{text}", num_images=0)

def run(tag):
    r = generate(model, processor, prompt, max_tokens=4, verbose=False)
    print(f"{tag}: prompt_tokens={r.prompt_tokens}  prefill={r.prompt_tps:.1f} tok/s  decode={r.generation_tps:.1f} tok/s", flush=True)
    return r.prompt_tps

run("warmup     ")
a1 = run("unpatched 1")
a2 = run("unpatched 2")

from mlx_vlm.dequant_prefill import apply
apply()
run("patch warm ")
b1 = run("patched   1")
b2 = run("patched   2")

base, pat = max(a1, a2), max(b1, b2)
print(f"\nbest unpatched {base:.1f} tok/s -> best patched {pat:.1f} tok/s  ({(pat/base-1)*100:+.1f}%)")
