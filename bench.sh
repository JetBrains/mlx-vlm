#!/usr/bin/env bash
set -euo pipefail

# Benchmark the running server by replaying a captured real Junie session
# (research/junie-replay). Prints per-request serving stats — KV cached,
# prefill speed over new tokens, generation speed, speculative acceptance —
# and their means. Stats come from the response "timings" blocks, so this
# works however the server was started.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8085

if ! curl -sf -m 5 "http://localhost:$PORT/health" > /dev/null 2>&1; then
  echo "Server is not running on port $PORT."
  echo "Start it first:  ./start.sh"
  exit 1
fi

PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/research/junie-replay/replay.py" \
  --url "http://localhost:$PORT" "$@"
