import os, sys
os.environ["HF_HUB_CACHE"] = "/Users/stanislav.erokhin/.local/share/junie-local/models"
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, "/Users/stanislav.erokhin/IdeaProjects/mlx-vlm")

from mlx_vlm import apply_chat_template, generate, load

model, processor = load("mlx-community/Qwen3.6-27B-4bit")
config = model.config

doc = open("/Users/stanislav.erokhin/IdeaProjects/mlx-vlm/research/int8-nax/README.md").read()
prompts = {
    "filler": "Summarize the following in one sentence:\n" + ("The quick brown fox jumps over the lazy dog near the riverbank at dawn. " * 400),
    "doc": "Read this engineering document and list its three most important conclusions, one line each:\n\n" + doc + doc,
}
prompts = {k: apply_chat_template(processor, config, v, num_images=0) for k, v in prompts.items()}

def run(tag, p, n=96):
    r = generate(model, processor, p, max_tokens=n, verbose=False)
    print(f"{tag}: tokens={r.prompt_tokens} prefill={r.prompt_tps:.1f} decode={r.generation_tps:.1f}", flush=True)
    return r

base = {}
generate(model, processor, prompts["filler"], max_tokens=2, verbose=False)  # warmup
for k, p in prompts.items():
    run(f"base warm {k}", p, n=2)
    base[k] = run(f"baseline  {k}", p)

from mlx_vlm.int8_prefill import apply, warmup, SCOPE
print(f"\napplying int8 patch, scope={SCOPE}")
apply(); warmup(model)

pat = {}
for k, p in prompts.items():
    run(f"int8 warm {k}", p, n=2)
    pat[k] = run(f"int8      {k}", p)

print()
for k in prompts:
    b, q = base[k], pat[k]
    same = b.text == q.text
    print(f"[{k}] prefill {b.prompt_tps:.0f} -> {q.prompt_tps:.0f} tok/s ({q.prompt_tps/b.prompt_tps-1:+.1%}), outputs identical: {same}")
    if not same:
        print(f"  baseline: {b.text[:300]}")
        print(f"  int8    : {q.text[:300]}")
