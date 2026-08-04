#!/usr/bin/env bash
set -euo pipefail

# Serve the OpenAI-compatible chat/completions endpoint using the mlx-vlm
# sources in this repo (not any pip-installed copy), with a pre-downloaded
# local model.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=8085
MODEL_ID="mlx-community/Qwen3.6-27B-4bit"
# Multi-token-prediction speculative-decoding drafter for the model above.
# It has no standalone language_model head, so it must be passed as
# --draft-model, never requested directly as a chat "model".
# Measured ~1.45-1.6x decode speedup at 2.24 accepted tokens/round; the
# drafter itself costs only ~9% of a round — the rest is the 3-token verify
# forward. Do not add --draft-block-size: the sweep (research/mtp-overhead)
# showed the configured depth 3 is optimal (2/4/5/6 are all slower) and the
# adaptive controller already handles bursts. Per-request acceptance shows
# up in the log as "Speculative decode: ... accepted_tokens_per_round=".
DRAFT_MODEL_ID="mlx-community/Qwen3.6-27B-MTP-4bit"

# The model is already downloaded into a Hugging Face hub-style cache dir
# (models--org--name/snapshots/...) that lives outside the default HF cache
# location. Point the HF cache at it and load by repo id, offline, so
# "/v1/models" reports a clean id instead of a raw filesystem path.
export HF_HUB_CACHE="/Users/stanislav.erokhin/.local/share/junie-local/models"
export HF_HUB_OFFLINE=1

# Make sure "import mlx_vlm" resolves to this checkout's sources, ahead of
# any installed package.
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

# Cross-request KV cache reuse (Automatic Prefix Caching). Qwen3.6 is a
# hybrid linear-attention model, so APC uses session storage: ONE shared
# full-attention KV set per conversation (~64 KiB/token, e.g. ~9 GiB at 150k)
# plus small recurrent-state checkpoints at the last few prefix lengths, so
# a warm hit can resume from any recent checkpoint -- including one before a
# mid-history edit. Verify via "APC enabled (...)" and GET /v1/cache/stats.
# Note: --preserve-thinking (below) is required for warm hits to survive new
# user turns -- without it the Qwen3.6 chat template re-renders older
# assistant turns (drops their <think> blocks) whenever a new user message
# arrives, which changes the token stream mid-history and misses the cache.
export APC_ENABLED=1
export APC_EXACT_SESSIONS=2        # concurrent conversations kept warm
export APC_SESSION_CHECKPOINTS=8   # resumable positions per conversation
# Persist APC snapshots on SSD so warm prefixes survive restarts.
export APC_DISK_PATH="$HOME/.local/share/junie-local/apc-cache"

# Stable cross-session prompt prefix (Junie system message + tool schemas +
# first user message; byte-identical across sessions). Prefilled once at
# startup, pinned in APC (never evicted, doesn't count against
# APC_EXACT_SESSIONS), and persisted via APC_DISK_PATH — so the FIRST
# request of a brand-new Junie session already warm-starts. Watch for
# "Seed prefix warmed and pinned" in the log.
SEED_REQUEST="$SCRIPT_DIR/research/junie.json"

# W8A8 int8 prefill on the M5 neural accelerators (see
# research/int8-nax/README.md). int8 weight tensors are built per layer by a
# fused kernel and freed right after use (MLX_VLM_INT8_CACHE=none default),
# so peak memory overhead is ~one layer, not a 24 GB copy; the larger
# prefill step amortizes the per-chunk rebuild (4096 measured best:
# ~1000 tok/s at 24.8 GB peak on a 12.6k prompt). If a quality issue shows
# up on real workloads, first try MLX_VLM_INT8_SCOPE=mlp (keeps attention
# numerics untouched), then drop --int8-prefill entirely.
exec "$PYTHON_BIN" -m mlx_vlm.server \
  --host 0.0.0.0 \
  --port "$PORT" \
  --model "$MODEL_ID" \
  --draft-model "$DRAFT_MODEL_ID" \
  --draft-kind mtp \
  --int8-prefill \
  --prefill-step-size 4096 \
  --preserve-thinking \
  --seed-request "$SEED_REQUEST"
