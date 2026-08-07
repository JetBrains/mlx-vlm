"""Cold-prefill step-size sweep on a long prompt (dequant patch applied).

Isolates what the session replay cannot: a single uncached 14.4k-token
prefill at each --prefill-step-size, with peak memory. Speed is flat across
1024-4096 on M4 while peak memory is not, which is why the shipped
prefill_step_size stays at the low end of that plateau.

Loads the model from the same place the server does (the config's
models_dir), so run it with the server stopped.
"""

import os
import sys

sys.path.insert(0, ".")

from mlx_vlm_shared.server_settings import DEFAULT_CONFIG, load_config

_cfg = load_config()
os.environ.setdefault(
    "HF_HUB_CACHE",
    os.path.expanduser(_cfg.get("models_dir") or DEFAULT_CONFIG["models_dir"]),
)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mlx.core as mx  # noqa: E402

from mlx_vlm import apply_chat_template, generate, load  # noqa: E402
from mlx_vlm.dequant_prefill import apply  # noqa: E402

apply()
model, processor = load(_cfg.get("model_name") or DEFAULT_CONFIG["model_name"])
config = model.config

text = "The quick brown fox jumps over the lazy dog near the riverbank at dawn. " * 900
prompt = apply_chat_template(
    processor, config, f"Summarize in one word:\n{text}", num_images=0
)


def run(step):
    r = generate(
        model, processor, prompt, max_tokens=2, verbose=False, prefill_step_size=step
    )
    mx.clear_cache()
    return r.prompt_tokens, r.prompt_tps, mx.get_peak_memory() / 1e9


run(4096)  # warmup (kernel compiles, page-in)
mx.reset_peak_memory()
print(f"device: {mx.device_info()['device_name']}")
for step in (512, 1024, 2048, 4096, 8192, 16384):
    mx.reset_peak_memory()
    toks, tps, peak = run(step)
    print(
        f"step {step:6d}: {toks} tokens  prefill {tps:6.1f} tok/s  peak {peak:.1f} GB",
        flush=True,
    )
# repeat best candidates to gauge noise
for step in (2048, 4096, 8192):
    toks, tps, _ = run(step)
    print(f"step {step:6d} (repeat): {tps:6.1f} tok/s", flush=True)
