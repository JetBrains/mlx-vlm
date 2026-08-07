#!/usr/bin/env bash
set -euo pipefail

# Prepare this checkout to serve: create ./.venv and install the locked
# dependencies. It starts nothing -- ./serverctl.sh start does that, for a
# checkout and an unpacked tarball alike.
#
# Run it once, and again whenever uv.lock changes. The project is installed
# editable, so ./.venv/bin/junie-mlx-vlm runs these sources, not a copy.
#
# Shipped machines need none of this: the frozen junie-mlx-vlm from
# build_cli_tarball.sh carries its own interpreter and dependencies.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
if [ "$OS_NAME" != "Darwin" ] || [ "$ARCH_NAME" != "arm64" ]; then
  echo "ERROR: This server requires macOS on Apple Silicon (arm64)." >&2
  echo "Detected: $OS_NAME $ARCH_NAME. Use a native arm64 terminal on Apple Silicon." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Managed Python environment.
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

echo
echo "Ready. Start the server with:  ./serverctl.sh start"
