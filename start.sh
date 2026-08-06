#!/usr/bin/env bash
set -euo pipefail

# One-command install + serve for the Junie local server.
#
# On every run this script makes sure the pieces are in place, then starts
# the OpenAI-compatible server from this repo's sources:
#   1) model weights   -> downloaded/verified into ~/.local/share/junie-local
#   2) python venv      -> created at ./.venv on first run
#   3) Junie descriptor -> written to ~/.junie/models
#   4) gateway          -> public API on host/port from server-config.json
#      worker           -> private inference process on port 8086
#
# Steps 1-3 are no-ops when already done, so this is also the everyday
# start command.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Everything below goes both to the screen and to mlx_server.log in the
# repo dir (gitignored; truncated on each start).
exec > >(tee "$SCRIPT_DIR/mlx_server.log") 2>&1

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
if [ "$OS_NAME" != "Darwin" ] || [ "$ARCH_NAME" != "arm64" ]; then
  echo "ERROR: This server requires macOS on Apple Silicon (arm64)." >&2
  echo "Detected: $OS_NAME $ARCH_NAME. Use a native arm64 terminal on Apple Silicon." >&2
  exit 1
fi

HOST=0.0.0.0
PORT=19239
WORKER_PORT=8086
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

# ---------------------------------------------------------------------------
# 1) Model weights (download style borrowed from junie-local's install.sh:
#    resumable curl with retry/backoff, SHA256 verification, HF-hub-layout
#    zips extracted with completion markers so interrupted installs redo
#    cleanly).
# ---------------------------------------------------------------------------
BASE_URL="https://download.jetbrains.com/resources/junie-local"
BASE_DIR="$HOME/.local/share/junie-local"
MODELS_DIR="$BASE_DIR/models"
DOWNLOAD_DIR="$BASE_DIR/incomplete_downloads"
export JUNIE_SERVER_CONFIG="$BASE_DIR/server-config.json"

MODEL_ZIP_1="models--mlx-community--Qwen3.6-27B-4bit.zip"
MODEL_SHA256_1="adf7f8d832ed994dcc6d09372036b4d12f49a4ccda066179cc64dc2dd113f91d"
MODEL_DIR_ID_1="mlx-community--Qwen3.6-27B-4bit"
MODEL_ZIP_2="models--mlx-community--Qwen3.6-27B-MTP-4bit.zip"
MODEL_SHA256_2="9266c1ba244ec6176fc82474bbfd20614969eb28c4cfa24301e515fbd1f5a525"
MODEL_DIR_ID_2="mlx-community--Qwen3.6-27B-MTP-4bit"

download_with_retry() {
  url="$1"
  output_file="$2"
  max_retries="${3:-3}"
  attempt=1
  delay=2

  while [ "$attempt" -le "$max_retries" ]; do
    echo "  Attempt $attempt of $max_retries..."
    if curl --progress-bar -SL -C - -o "$output_file" "$url"; then
      return 0
    fi

    if [ "$attempt" -lt "$max_retries" ]; then
      echo "  Download failed. Retrying in ${delay}s..."
      sleep "$delay"
      delay=$((delay * 2))
    fi
    attempt=$((attempt + 1))
  done

  echo "  ERROR: Download failed after $max_retries attempts."
  return 1
}

download_and_verify() {
  archive="$1"
  expected_sha256="$2"

  echo "Downloading $archive..."
  download_with_retry "$BASE_URL/$archive" "$DOWNLOAD_DIR/$archive"
  echo "  Download complete. Checking SHA256..."

  actual=$(shasum -a 256 "$DOWNLOAD_DIR/$archive" | awk '{print $1}')
  if [ "$actual" != "$expected_sha256" ]; then
    echo "  ERROR: SHA256 mismatch for $archive"
    echo "    Expected: $expected_sha256"
    echo "    Actual:   $actual"
    rm -f "$DOWNLOAD_DIR/$archive"
    exit 1
  fi
  echo "  SHA256 verified: $actual"
}

model_completion_marker() {
  echo "$MODELS_DIR/.models--$1.installed"
}

model_installed() {
  model_dir_id="$1"
  [ -d "$MODELS_DIR/models--$model_dir_id" ] \
    && [ -f "$(model_completion_marker "$model_dir_id")" ]
}

install_model_if_needed() {
  zip_file="$1"
  sha256_sum="$2"
  model_dir_id="$3"

  if model_installed "$model_dir_id"; then
    return 0
  fi

  echo "Model $model_dir_id is not installed. Downloading..."
  mkdir -p "$MODELS_DIR" "$DOWNLOAD_DIR"
  download_and_verify "$zip_file" "$sha256_sum"
  echo "Extracting $zip_file to $MODELS_DIR..."
  # Remove leftovers from a previously interrupted extraction
  rm -rf "$MODELS_DIR/models--$model_dir_id"
  unzip -q "$DOWNLOAD_DIR/$zip_file" -d "$MODELS_DIR"
  touch "$(model_completion_marker "$model_dir_id")"
  rm -f "$DOWNLOAD_DIR/$zip_file"
  echo "  Extraction complete."
}

install_model_if_needed "$MODEL_ZIP_1" "$MODEL_SHA256_1" "$MODEL_DIR_ID_1"
install_model_if_needed "$MODEL_ZIP_2" "$MODEL_SHA256_2" "$MODEL_DIR_ID_2"
rmdir "$DOWNLOAD_DIR" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 2) Managed Python environment.
#
# mlx>=0.32 ships wheels for CPython 3.10-3.14 only, while the stock
# /usr/bin/python3 from the Xcode Command Line Tools is still 3.9 -- so the
# venv is built with uv, which downloads a suitable managed CPython on its
# own. uv itself is auto-installed into <repo>/.uv when not already
# present, so the script has no prerequisites at all and everything it
# bootstraps stays inside the repo dir (gitignored).
# ---------------------------------------------------------------------------
VENV="$SCRIPT_DIR/.venv"
VENV_PYTHON_VERSION=3.13
UV_DIR="$SCRIPT_DIR/.uv"

# Managed-CPython downloads also go inside the repo dir (uv's default is
# ~/.local/share/uv/python).
export UV_PYTHON_INSTALL_DIR="$UV_DIR/python"

find_uv() {
  command -v uv 2>/dev/null && return 0
  for cand in "$UV_DIR/bin/uv" "$HOME/.local/bin/uv"; do
    if [ -x "$cand" ]; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

if ! UV_BIN="$(find_uv)"; then
  echo "Installing uv (Python package manager) to $UV_DIR/bin ..."
  curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR="$UV_DIR/bin" INSTALLER_NO_MODIFY_PATH=1 sh
  UV_BIN="$UV_DIR/bin/uv"
fi

# Keep the project environment on the same Python version used by the lock.
if [ -x "$VENV/bin/python" ] \
   && ! "$VENV/bin/python" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 13) else 1)' \
        >/dev/null 2>&1; then
  echo "Existing virtualenv does not use Python $VENV_PYTHON_VERSION; recreating it..."
  rm -rf "$VENV"
fi

# This creates the venv and managed Python when missing. On later starts it is
# a fast no-op unless uv.lock changed. --locked prevents silent version drift.
echo "Syncing locked Python dependencies..."
"$UV_BIN" sync --locked --python "$VENV_PYTHON_VERSION"

PYTHON_BIN="$VENV/bin/python"

# ---------------------------------------------------------------------------
# 3) Gateway address and Junie model descriptor.
#
# Read the address with the managed Python created above. A missing or broken
# config uses the defaults and will be normalized when the gateway starts.
# ---------------------------------------------------------------------------
CONFIG_ADDRESS="$("$PYTHON_BIN" - "$JUNIE_SERVER_CONFIG" <<'PY'
import json
import sys

from mlx_vlm_shared.server_settings import normalize_config

try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raw = {}
except (OSError, ValueError):
    raw = {}

config, _ = normalize_config(raw)
print(f'{config["host"]}\t{config["port"]}')
PY
)"
IFS=$'\t' read -r HOST PORT <<< "$CONFIG_ADDRESS"

# The id must be the real HF repo id because Junie sends it as the "model".
JUNIE_MODELS_DIR="$HOME/.junie/models"
JUNIE_MODEL_NAME="local-qwen3.6-27b-4bit-vlm"
JUNIE_MODEL_FILE="$JUNIE_MODELS_DIR/$JUNIE_MODEL_NAME.json"
mkdir -p "$JUNIE_MODELS_DIR"
cat > "$JUNIE_MODEL_FILE" <<EOF
{
  "id": "$MODEL_ID",
  "baseUrl": "http://localhost:$PORT/v1/chat/completions",
  "apiType": "OpenAICompletion",
  "temperature": 0.6,
  "maxContextLength": 150000,
  "extraBody": {
    "enable_thinking": false
  }
}
EOF
echo "Junie model descriptor: $JUNIE_MODEL_FILE"

# Set this model as Junie's default (same mechanism as junie-local's
# install.sh: descriptor-file models are addressed as "custom:<file stem>").
JUNIE_SETTINGS="$HOME/.junie/settings.json"
if [ -f "$JUNIE_SETTINGS" ]; then
  plutil -replace "modelForLaunch" -string "custom:$JUNIE_MODEL_NAME" \
    "$JUNIE_SETTINGS"
  echo "Junie default model set to $JUNIE_MODEL_NAME (restart Junie to apply)."
else
  echo "WARNING: Junie settings not found at $JUNIE_SETTINGS;"
  echo "         select the $MODEL_ID model in Junie manually."
fi

# ---------------------------------------------------------------------------
# 4) Gateway and inference worker.
# ---------------------------------------------------------------------------

# The models live in a Hugging Face hub-style cache dir (models--org--name/
# snapshots/...) outside the default HF cache location. Point the HF cache
# at it and load by repo id, offline, so "/v1/models" reports a clean id
# instead of a raw filesystem path.
export HF_HUB_CACHE="$MODELS_DIR"
export HF_HUB_OFFLINE=1

# Make sure "import mlx_vlm" resolves to this checkout's sources, ahead of
# any installed package.
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ "$PORT" = "$WORKER_PORT" ]; then
  echo "ERROR: Public gateway port $PORT conflicts with private worker port $WORKER_PORT." >&2
  exit 1
fi

check_port_available() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind((host, port))
except OSError as exc:
    raise SystemExit(f"ERROR: Cannot listen on {host}:{port}: {exc}")
finally:
    sock.close()
PY
}

check_port_available "$HOST" "$PORT"
check_port_available 127.0.0.1 "$WORKER_PORT"

# Model, drafter, prefill, prompt/KV cache and warmup settings come from
# JUNIE_SERVER_CONFIG through mlx_vlm.server.junie. The gateway keeps the
# worker transport private on localhost port 8086.

# Stop an individual non-streaming request cleanly before Junie's five-minute
# retry window. The gateway has a separate 275-second hard limit for workers
# that cannot acknowledge cancellation.
export MLX_VLM_SOFT_REQUEST_TIMEOUT=270

exec "$PYTHON_BIN" -m mlx_vlm_gateway \
  --host "$HOST" \
  --port "$PORT" \
  --worker-url "http://127.0.0.1:$WORKER_PORT" \
  --startup-timeout 120 \
  --request-timeout 275 \
  -- \
  "$PYTHON_BIN" -m mlx_vlm.server.junie \
  --host 127.0.0.1 \
  --port "$WORKER_PORT"
