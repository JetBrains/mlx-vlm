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
# hybrid linear-attention model, so APC runs in "exact" whole-prefix-snapshot
# mode: a warm hit needs the new prompt to extend a previously seen prompt,
# which is the normal multi-turn chat pattern. Default keeps only 2 in-memory
# snapshots; raise it a bit so a few parallel conversations stay warm.
# Verify via server log line "APC enabled (...)" and GET /v1/cache/stats.
# Note: --preserve-thinking (below) is required for warm hits to survive new
# user turns -- without it the Qwen3.6 chat template re-renders older
# assistant turns (drops their <think> blocks) whenever a new user message
# arrives, which changes the token stream mid-history and misses the cache.
export APC_ENABLED=1
export APC_EXACT_CACHE_ENTRIES=4

# W8A8 int8 prefill on the M5 neural accelerators (+30-47% prefill measured,
# decode untouched -- see research/int8-nax/README.md). If a quality issue
# shows up on real workloads, first try MLX_VLM_INT8_SCOPE=mlp (keeps
# attention numerics untouched), then drop --int8-prefill entirely.
exec "$PYTHON_BIN" -m mlx_vlm.server \
  --host 0.0.0.0 \
  --port "$PORT" \
  --model "$MODEL_ID" \
  --draft-model "$DRAFT_MODEL_ID" \
  --draft-kind mtp \
  --int8-prefill \
  --preserve-thinking
