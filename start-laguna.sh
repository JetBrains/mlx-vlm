#!/usr/bin/env bash
set -euo pipefail

# Serve the OpenAI-compatible chat/completions endpoint using the mlx-vlm
# sources in this repo (not any pip-installed copy), with the local Laguna
# model. Same port as start.sh / start-mxfp4.sh -- run only one of them.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=8085
MODEL_ID="mlx-community/Laguna-S-2.1-oQ2e-fast"

# The model lives as a plain directory under MODELS_ROOT, already laid out
# as <org>/<name>. Resolving MODEL_ID as a relative path against MODELS_ROOT
# makes get_model_path() load it directly (no download), while "/v1/models"
# reports the clean id "mlx-community/Laguna-S-2.1-oQ2e-fast" (the server
# uses the exact --model string passed here as that id).
MODELS_ROOT="/Users/stanislav.erokhin/.omlx/models"

export HF_HUB_OFFLINE=1

# Make sure "import mlx_vlm" resolves to this checkout's sources, ahead of
# any installed package.
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

cd "$MODELS_ROOT"

# --hybrid-fp16: fp16 block internals for the early layers, bf16 residual
# stream (faster on M1-family GPUs, which emulate bfloat).
# --dequant-prefill (bf16 GEMMs for prefill-sized matmuls) is available too
# but disabled for now.
exec "$PYTHON_BIN" -m mlx_vlm.server \
  --host 0.0.0.0 \
  --port "$PORT" \
  --model "$MODEL_ID"
