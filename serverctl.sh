#!/usr/bin/env bash
set -euo pipefail

# Small command-line client for the gateway control API.
#
# Examples:
#   ./serverctl.sh start
#   ./serverctl.sh status
#   ./serverctl.sh settings
#   ./serverctl.sh apply max_context_length=150000
#   ./serverctl.sh apply auto_unload_time=600
#   ./serverctl.sh apply kv_quantization=true force=true
#   ./serverctl.sh wait
#   ./serverctl.sh models
#   ./serverctl.sh unload
#   ./serverctl.sh stop

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  PYTHON="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "ERROR: Python environment is missing; run ./start.sh first." >&2
  exit 1
fi

CONFIG_PATH="$HOME/.local/share/junie-local/server-config.json"
CONFIG_PORT=$(PYTHONPATH="$SCRIPT_DIR" "$PYTHON" - "$CONFIG_PATH" <<'PY'
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
BASE="http://127.0.0.1:$PORT"
CURL=(curl -sS --fail-with-body -m 30)

usage() {
  sed -n '/^# Examples:/,/^$/{s/^# \{0,1\}//p;}' "${BASH_SOURCE[0]}"
  exit 1
}

pretty() { "$PYTHON" -m json.tool; }
get() { "${CURL[@]}" "$BASE$1" | pretty; }
post() {
  if [ $# -eq 2 ]; then
    "${CURL[@]}" -X POST -H 'Content-Type: application/json' \
      -d "$2" "$BASE$1" | pretty
  else
    "${CURL[@]}" -X POST "$BASE$1" | pretty
  fi
}

kv_to_json() {
  "$PYTHON" - "$@" <<'PY'
import json
import sys

body = {}
for pair in sys.argv[1:]:
    if "=" not in pair:
        raise SystemExit(f"ERROR: expected key=value, got {pair!r}")
    key, value = pair.split("=", 1)
    if value == "true":
        value = True
    elif value == "false":
        value = False
    elif value == "null":
        value = None
    else:
        try:
            value = int(value)
        except ValueError:
            pass
    body[key] = value
print(json.dumps(body))
PY
}

apply_payload() {
  payload="$1"
  busy=$(curl -sS -m 5 "$BASE/status" 2>/dev/null \
    | "$PYTHON" -c \
      'import json,sys; print(str(json.load(sys.stdin)["inference"]["in_progress"]).lower())' \
    2>/dev/null || echo false)
  forced=$(printf '%s' "$payload" \
    | "$PYTHON" -c \
      'import json,sys; print(str(json.load(sys.stdin).get("force") is True).lower())')

  if [ "$busy" = true ] && [ "$forced" != true ]; then
    if [ ! -t 0 ]; then
      echo "ERROR: inference is active; retry with force=true to stop it." >&2
      exit 1
    fi
    read -r -p "Inference is active. Stop it and apply settings now? [y/N] " answer
    case "$answer" in
      y | Y | yes | YES)
        payload=$(printf '%s' "$payload" \
          | "$PYTHON" -c \
            'import json,sys; value=json.load(sys.stdin); value["force"]=True; print(json.dumps(value))')
        ;;
      *) echo "Settings were not changed."; exit 1 ;;
    esac
  fi
  post /apply_settings "$payload"
}

wait_ready() {
  while :; do
    phase=$(curl -sS -m 5 "$BASE/status" 2>/dev/null \
      | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["phase"])' \
      2>/dev/null || echo unreachable)
    echo "phase: $phase"
    case "$phase" in
      ready) return 0 ;;
      error) get /status; return 1 ;;
    esac
    sleep 2
  done
}

command="${1:-}"
[ $# -gt 0 ] && shift

case "$command" in
  start) exec "$SCRIPT_DIR/start.sh" ;;
  status) get /status ;;
  wait) wait_ready ;;
  settings) get /current_settings ;;
  apply)
    [ $# -gt 0 ] || usage
    apply_payload "$(kv_to_json "$@")"
    ;;
  apply-json)
    [ $# -eq 1 ] || usage
    apply_payload "$1"
    ;;
  stop) post /shutdown ;;
  health) get /health ;;
  metrics) get /metrics ;;
  cache-stats) get /v1/cache/stats ;;
  cache-reset) post /v1/cache/reset ;;
  models) get /v1/models ;;
  unload) post /unload ;;
  *) usage ;;
esac
