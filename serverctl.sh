#!/usr/bin/env bash
set -euo pipefail

# Thin curl wrapper around the local server's HTTP API (see JUNIE_API.md).
#
# Usage:
#   ./serverctl.sh start                    launch the server (background, silent);
#                                           from a checkout, ./init_dev.sh first
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
#   ./serverctl.sh uninstall                stop the engine and remove everything
#                                           install.sh set up: the install
#                                           directory (engine, models, logs,
#                                           config) and the Junie model config
#                                           it wrote; only the default install
#                                           path is supported
#   ./serverctl.sh health | models | metrics | cache-stats | unload
#
# Everything but "start" and "uninstall" is plain HTTP, so this drives a
# checkout and the frozen junie-mlx-vlm alike. PORT overrides the port read
# from the config, API_KEY the api_key read from it.

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

# The bearer token both the daemon and the worker require on every request,
# read from the same file the same way -- install.sh generates one per
# machine and writes it there. Empty covers all the cases that mean "send no
# header": a config with "api_key": null (a checkout that never ran
# install.sh, whose API is open), a config without the key at all, and no
# config yet.
API_KEY="${API_KEY:-$(plutil -extract api_key raw -o - -- "$CONFIG_PATH" 2>/dev/null || true)}"

# The daemon's own output, beside the worker log it writes itself.
DAEMON_LOG="${CONFIG_PATH%/*}/junie-mlx-vlm-daemon.log"

usage() {
  sed -n '/^# Usage:/,/^$/{s/^# \{0,1\}//p;}' "${BASH_SOURCE[0]}"
  exit 1
}

CURL=(curl -sS --fail-with-body -m 30)
if [ -n "$API_KEY" ]; then
  CURL+=(-H "Authorization: Bearer $API_KEY")
fi

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
# to run. It is the same command either way -- an unpacked tarball has the
# junie-mlx-vlm binary beside this script, a checkout has it in the venv that
# init_dev.sh builds, and it may simply be on PATH.
start_server() {
  if "${CURL[@]}" -o /dev/null -m 2 "$BASE/health" >/dev/null 2>&1; then
    echo "Already serving on port $PORT."
    return 0
  fi

  server=""
  for candidate in \
    "$SCRIPT_DIR/junie-mlx-vlm" \
    "$SCRIPT_DIR/.venv/bin/junie-mlx-vlm"; do
    if [ -x "$candidate" ]; then
      server="$candidate"
      break
    fi
  done
  if [ -z "$server" ]; then
    server="$(command -v junie-mlx-vlm || true)"
  fi
  if [ -z "$server" ]; then
    echo "ERROR: no junie-mlx-vlm beside this script, in ./.venv/bin or on" >&2
    echo "       PATH. From a checkout, run ./init_dev.sh first." >&2
    exit 1
  fi

  # The daemon writes the worker's log itself; this is its own output --
  # its startup lines, uvicorn's, and anything that dies before logging
  # exists. Kept one run back, the same way the daemon keeps the worker's.
  mkdir -p "$(dirname "$DAEMON_LOG")"
  if [ -f "$DAEMON_LOG" ]; then
    mv -f "$DAEMON_LOG" "$DAEMON_LOG.0"
  fi

  # Detached: the server outlives this shell, and says nothing here.
  nohup "$server" >"$DAEMON_LOG" 2>&1 &
  echo "Started $(basename "$server") (pid $!); logging to $DAEMON_LOG"
  echo "Follow it with ./serverctl.sh wait"
}

# Undo what install.sh set up: stop the engine, remove the Junie model config
# it wrote (clearing the default-model setting if it names that model), then
# delete the install directory — engine versions, the current
# symlink, models, logs and server-config.json. Deleting the tree this script
# runs from is fine: the shell keeps its open file, and nothing here executes
# from the tree afterwards.
#
# The rm -rf target is deliberately the spelled-out default install path and
# nothing else — never a variable derived from the environment. With
# JUNIE_SERVER_CONFIG pointing anywhere else this refuses to run rather than
# guess which directory to delete.
uninstall_all() {
  if [ "$CONFIG_PATH" != "$HOME/.local/share/junie-local/server-config.json" ]; then
    echo "ERROR: uninstall only supports the default install path." >&2
    echo "       JUNIE_SERVER_CONFIG points at $CONFIG_PATH — unset it and" >&2
    echo "       re-run, or remove that installation manually." >&2
    exit 1
  fi

  # The exact model id install.sh writes the Junie config under.
  junie_model_id="local-qwen3.6-27b-4bit"
  junie_model_config="$HOME/.junie/models/$junie_model_id.json"
  junie_settings="$HOME/.junie/settings.json"

  # Stop the engine first, so nothing holds the port or writes to the tree.
  if "${CURL[@]}" -o /dev/null -m 2 "$BASE/health" >/dev/null 2>&1; then
    echo "Stopping the engine on port $PORT..."
    "${CURL[@]}" -X POST -m 10 "$BASE/shutdown" >/dev/null 2>&1 || true
  fi
  waited=0
  while [ "$waited" -lt 10 ] && pgrep -f junie-mlx-vlm >/dev/null 2>&1; do
    sleep 1
    waited=$((waited + 1))
  done
  if pgrep -f junie-mlx-vlm >/dev/null 2>&1; then
    echo "The engine did not stop in time; killing it."
    pkill -f junie-mlx-vlm || true
  fi

  if [ -f "$junie_model_config" ]; then
    echo "Removing Junie model config $junie_model_config"
    rm -f "$junie_model_config"
  fi
  launch_model="$(plutil -extract modelForLaunch raw -o - -- "$junie_settings" 2>/dev/null || true)"
  if [ "$launch_model" = "custom:$junie_model_id" ]; then
    echo "Clearing default model custom:$junie_model_id in $junie_settings"
    plutil -remove modelForLaunch "$junie_settings" 2>/dev/null || true
  fi

  if [ -d "$HOME/.local/share/junie-local" ]; then
    echo "Removing $HOME/.local/share/junie-local (engine, models, logs, config)..."
    rm -rf "$HOME/.local/share/junie-local"
  else
    echo "Nothing installed at $HOME/.local/share/junie-local."
  fi

  echo "Uninstall complete. Restart Junie to apply the changes."
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
  uninstall) uninstall_all ;;
  health) get /health ;;
  models) get /v1/models ;;
  metrics) get /metrics ;;
  cache-stats) get /v1/cache/stats ;;
  unload) post /unload ;;
  *) usage ;;
esac
