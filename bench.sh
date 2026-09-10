#!/usr/bin/env bash
set -euo pipefail

# Benchmark the running server by replaying a captured real Junie session
# (research/junie-replay). Prints per-request serving stats — KV cached,
# prefill speed over new tokens, generation speed, speculative acceptance —
# and their means. Stats come from the response "timings" blocks, so this
# works however the server was started.
#
# Usage:
#   ./bench.sh                              # use model from request files
#   ./bench.sh --model Qwen3.8-27B-MLX-4bit  # override model for benchmarking
#   ./bench.sh --reasoning-effort low       # set reasoning_effort (auto-enables thinking)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "ERROR: Python environment is missing; run ./init_dev.sh first." >&2
    exit 1
  fi
fi

CONFIG_PATH="$HOME/.local/share/junie-local/server-config.json"
# Port on the first line, api_key (empty when the config has none) on the
# second: one interpreter start for both.
CONFIG_VALUES=$(PYTHONPATH="$SCRIPT_DIR" "$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
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
print(config["port"])
print(config["api_key"] or "")
PY
)
PORT="${PORT:-$(printf '%s\n' "$CONFIG_VALUES" | sed -n 1p)}"
# Both the daemon and the worker reject requests without this key when the
# config has one; replay.py picks it up from the environment.
export MLX_VLM_SERVER_API_KEY="${MLX_VLM_SERVER_API_KEY:-$(printf '%s\n' "$CONFIG_VALUES" | sed -n 2p)}"

HEALTH_CURL=(curl -sf -m 5)
if [ -n "$MLX_VLM_SERVER_API_KEY" ]; then
  HEALTH_CURL+=(-H "Authorization: Bearer $MLX_VLM_SERVER_API_KEY")
fi
if ! "${HEALTH_CURL[@]}" "http://localhost:$PORT/health" > /dev/null 2>&1; then
  echo "Server is not running on port $PORT."
  echo "Start it first:  ./serverctl.sh start"
  exit 1
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/research/junie-replay/replay.py" \
  --url "http://localhost:$PORT" "$@"
