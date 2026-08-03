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

exec "$PYTHON_BIN" -m mlx_vlm.server \
  --host 0.0.0.0 \
  --port "$PORT" \
  --model "$MODEL_ID" \
  --draft-model "$DRAFT_MODEL_ID" \
  --draft-kind mtp
