#!/usr/bin/env bash
set -euo pipefail

# Thin curl wrapper around the local server's HTTP API (see JUNIE_API.md).
#
# Usage:
#   ./serverctl.sh start                    launch the server (background, silent)
#   ./serverctl.sh status                   lifecycle phase + inference progress
#   ./serverctl.sh wait                     poll status until phase is "ready"
#   ./serverctl.sh settings                 current serving settings
#   ./serverctl.sh apply key=value [...]    apply settings, restarting the worker
#                                           when the setting requires it:
#                                             ./serverctl.sh apply max_context_length=150000
#                                             ./serverctl.sh apply auto_unload_time=600
#                                             ./serverctl.sh apply kv_quantization=true force=true
#                                           numbers/true/false/null are sent as-is,
#                                           anything else as a JSON string
#   ./serverctl.sh apply-json '{"max_context_length": 150000}'
#   ./serverctl.sh stop                     POST /shutdown (graceful)
#   ./serverctl.sh health | models | metrics | cache-stats | unload
#
# Everything but "start" is plain HTTP, so this drives a checkout and the
# frozen junie-mlx-vlm alike. PORT overrides the port read from the config.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The port the daemon serves on, from the same file the daemon reads, with the
# same fallback as DEFAULT_CONFIG["port"] in mlx_vlm_shared/server_settings.py
# for when that file has not been written yet. plutil parses JSON, so no
# interpreter is needed; it fails alike on a missing file, a missing key and
# unparsable contents, and any of those means "use the default".
DEFAULT_PORT=19239
CONFIG_PATH="${JUNIE_SERVER_CONFIG:-$HOME/.local/share/junie-local/server-config.json}"
PORT="${PORT:-$(plutil -extract port raw -o - -- "$CONFIG_PATH" 2>/dev/null || true)}"
case "$PORT" in
  '' | *[!0-9]*) PORT="$DEFAULT_PORT" ;;
esac
BASE="http://localhost:$PORT"

usage() {
  sed -n '/^# Usage:/,/^$/{s/^# \{0,1\}//p;}' "${BASH_SOURCE[0]}"
  exit 1
}

CURL=(curl -sS --fail-with-body -m 30)

# plutil reprints JSON but sorts the keys, and refuses anything that is not a
# plist or JSON -- so fall back to the raw body, since an error page is still
# worth reading.
pretty() {
  body="$(cat)"
  if formatted="$(printf '%s' "$body" | plutil -convert json -r -o - -- - 2>/dev/null)"; then
    printf '%s\n' "$formatted"
  else
    printf '%s\n' "$body"
  fi
}

get() { "${CURL[@]}" "$BASE$1" | pretty; }
post() {
  if [ $# -ge 2 ]; then
    "${CURL[@]}" -X POST -H 'Content-Type: application/json' -d "$2" "$BASE$1" | pretty
  else
    "${CURL[@]}" -X POST "$BASE$1" | pretty
  fi
}

kv_to_json() {
  json="{"
  sep=""
  for pair in "$@"; do
    case "$pair" in
      *=*) ;;
      *) echo "ERROR: expected key=value, got '$pair'" >&2; exit 1 ;;
    esac
    key="${pair%%=*}"
    value="${pair#*=}"
    case "$value" in
      true | false | null) ;;
      *)
        if ! [[ "$value" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
          value="\"$value\""
        fi
        ;;
    esac
    json="$json$sep\"$key\": $value"
    sep=", "
  done
  echo "$json}"
}

# The one command that cannot be binary agnostic, because it has to know what
# to run: a checkout has start_dev.sh, an unpacked tarball has the
# junie-mlx-vlm binary either beside this script or on PATH.
start_server() {
  if "${CURL[@]}" -o /dev/null -m 2 "$BASE/health" >/dev/null 2>&1; then
    echo "Already serving on port $PORT."
    return 0
  fi

  server=""
  for candidate in "$SCRIPT_DIR/start_dev.sh" "$SCRIPT_DIR/junie-mlx-vlm"; do
    if [ -x "$candidate" ]; then
      server="$candidate"
      break
    fi
  done
  if [ -z "$server" ]; then
    server="$(command -v junie-mlx-vlm || true)"
  fi
  if [ -z "$server" ]; then
    echo "ERROR: found neither $SCRIPT_DIR/start_dev.sh nor a junie-mlx-vlm" >&2
    echo "       binary beside this script or on PATH." >&2
    exit 1
  fi

  # Detached and quiet: the server outlives this shell and prints nothing
  # here. start_dev.sh still writes mlx_server.log; the binary logs nowhere.
  nohup "$server" >/dev/null 2>&1 &
  echo "Started $(basename "$server") (pid $!); follow it with ./serverctl.sh wait"
}

wait_ready() {
  while :; do
    phase="$("${CURL[@]}" -m 5 "$BASE/status" 2>/dev/null \
      | plutil -extract phase raw -o - -- - 2>/dev/null || true)"
    echo "phase: ${phase:-unreachable}"
    case "$phase" in
      ready) return 0 ;;
      error)
        get /status
        return 1
        ;;
    esac
    sleep 2
  done
}

cmd="${1:-}"
[ $# -gt 0 ] && shift

case "$cmd" in
  start) start_server ;;
  status) get /status ;;
  wait) wait_ready ;;
  settings) get /current_settings ;;
  apply)
    [ $# -gt 0 ] || usage
    post /apply_settings "$(kv_to_json "$@")"
    ;;
  apply-json)
    [ $# -eq 1 ] || usage
    post /apply_settings "$1"
    ;;
  stop) post /shutdown ;;
  health) get /health ;;
  models) get /v1/models ;;
  metrics) get /metrics ;;
  cache-stats) get /v1/cache/stats ;;
  unload) post /unload ;;
  *) usage ;;
esac
