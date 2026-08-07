#!/usr/bin/env bash
set -euo pipefail

# One-command serve for the Junie local server.
#
# This script only prepares the Python environment and starts the daemon:
#   1) python venv -> created at ./.venv on first run
#   2) daemon      -> python -m mlx_vlm_gateway
#
# The daemon takes no arguments: it reads every setting from the config
# file (JUNIE_SERVER_CONFIG, default
# ~/.local/share/junie-local/server-config.json), serves the public API,
# and spawns the inference worker itself. Model weights and the Junie
# model descriptor are installed separately.

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

# ---------------------------------------------------------------------------
# 1) Managed Python environment.
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

# Make sure "import mlx_vlm" resolves to this checkout's sources, ahead of
# any installed package.
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ---------------------------------------------------------------------------
# 2) Daemon.
# ---------------------------------------------------------------------------
exec "$VENV/bin/python" -m mlx_vlm_gateway
