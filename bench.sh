#!/usr/bin/env bash
set -euo pipefail

# Benchmark the running server by replaying a captured real Junie session
# (research/junie-replay). Prints per-request serving stats — KV cached,
# prefill speed over new tokens, generation speed, speculative acceptance —
# and their means. Stats come from the response "timings" blocks, so this
# works however the server was started.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "ERROR: Python environment is missing; run ./start.sh first." >&2
    exit 1
  fi
fi

CONFIG_PATH="$HOME/.local/share/junie-local/server-config.json"
CONFIG_PORT=$(PYTHONPATH="$SCRIPT_DIR" "$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
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
PY
)
PORT="${PORT:-$CONFIG_PORT}"

if ! curl -sf -m 5 "http://localhost:$PORT/health" > /dev/null 2>&1; then
  echo "Server is not running on port $PORT."
  echo "Start it first:  ./start.sh"
  exit 1
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/research/junie-replay/replay.py" \
  --url "http://localhost:$PORT" "$@"
