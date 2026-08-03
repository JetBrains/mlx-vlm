#!/usr/bin/env bash
set -euo pipefail

# Serve the OpenAI-compatible chat/completions endpoint using the mlx-vlm
# sources in this repo (not any pip-installed copy), with the experimental
# mxfp4-quantized conversion of the model (see start.sh for the current
# affine-int4 production model). Same port as start.sh -- run one or the
# other, not both.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=8085
MODEL_ID="mlx-community/Qwen3.6-27B-mxfp4-test"
# Multi-token-prediction speculative-decoding drafter, carried over from the
# affine-int4 setup. NOTE: its speculative-decoding acceptance rate against
# this mxfp4 target has not been verified -- it was tuned as a drafter for
# the affine-int4 model.
DRAFT_MODEL_ID="mlx-community/Qwen3.6-27B-MTP-4bit"

# MODEL_ID above is not a real HF repo id -- it's resolved as a plain
# relative path against ALIAS_ROOT below, which holds a symlink
# (mlx-community/Qwen3.6-27B-mxfp4-test -> the actual converted model
# directory). This makes get_model_path() pick it up directly, with no
# download and no duplicated weights, while still reporting the clean id
# "mlx-community/Qwen3.6-27B-mxfp4-test" in "/v1/models" (the server uses
# the exact --model string passed here as that id).
ALIAS_ROOT="/Users/stanislav.erokhin/.exo/models/_hf_repo_alias"

# HF_HUB_CACHE is still needed to resolve the draft model above by repo id.
export HF_HUB_CACHE="/Users/stanislav.erokhin/.local/share/junie-local/models"
export HF_HUB_OFFLINE=1

# Make sure "import mlx_vlm" resolves to this checkout's sources, ahead of
# any installed package.
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

cd "$ALIAS_ROOT"

exec "$PYTHON_BIN" -m mlx_vlm.server \
  --host 0.0.0.0 \
  --port "$PORT" \
  --model "$MODEL_ID" \
  --draft-model "$DRAFT_MODEL_ID" \
  --draft-kind mtp
