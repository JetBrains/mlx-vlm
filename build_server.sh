#!/usr/bin/env bash
set -euo pipefail

# Freeze mlx_vlm_gateway/cli.py into a self-contained tar.gz.
#
#   Output:  dist/junie-mlx-vlm-<version>-macos-arm64.tar.gz
#   Run:     junie-mlx-vlm/junie-mlx-vlm <mlx_vlm_gateway.cli args>
#
# Neither subcommand takes options of its own: every setting comes from
# JUNIE_SERVER_CONFIG. One executable serves both, because the daemon
# re-invokes itself as `junie-mlx-vlm worker`.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NAME="junie-mlx-vlm"
# This binary's own version, not the upstream mlx-vlm one in mlx_vlm/version.py
# that the Python package still carries.
VERSION="$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' mlx_vlm_gateway/version.py)"
ARCHIVE="$SCRIPT_DIR/dist/$NAME-$VERSION-macos-arm64.tar.gz"

# Deleted after the archive is built; a failed build leaves it for inspection.
BUILD_DIR="$(mktemp -d -t "build-$NAME")"
PYTHON="$BUILD_DIR/.venv/bin/python"

# uv installs a managed CPython: mlx needs 3.10+.
UV_PYTHON_INSTALL_DIR="$BUILD_DIR/.python" \
  uv venv --python 3.13 "$BUILD_DIR/.venv"

# Install the checkout as a package: the frozen server imports mlx_vlm, and
# pyproject.toml takes its dependency list from requirements.txt.
uv pip install --python "$PYTHON" . pyinstaller

# mlx.core is a compiled extension; ask it for its lib directory.
MLX_LIB_DIR="$("$PYTHON" -c 'import mlx.core, pathlib; print(pathlib.Path(mlx.core.__file__).resolve().parent / "lib")')"
# PyInstaller follows every import statement, including those inside function
# bodies. These are the dependencies it cannot see that way.
HIDDEN_DEPS=(
  # mlx_vlm (this repo): architectures, drafters and tool parsers -- importlib
  # names built from config.json and from the chat template.
  --collect-submodules mlx_vlm.models
  --collect-submodules mlx_vlm.speculative.drafters
  --collect-submodules mlx_vlm.tool_parsers
  # mlx: imported from native code.
  --hidden-import mlx._reprlib_fix
  # mlx: 162 MB shader library, opened at run time as data.
  --add-data "$MLX_LIB_DIR/mlx.metallib:mlx/lib"
  # mlx-lm: importlib names from config keys; only reached for text-only models.
  --collect-submodules mlx_lm.models
  --collect-submodules mlx_lm.tool_parsers
  --collect-submodules mlx_lm.chat_templates
  # llguidance: llguidance/__init__.py does `__version__ = version("llguidance")`
  # at import time, so it needs its .dist-info, not only its code.
  --copy-metadata llguidance
)

# --noupx: UPX breaks macOS binaries; PyInstaller uses it if present (upx#601).
# pushd: PyInstaller writes dist/, build/ and the spec into the cwd.
pushd "$BUILD_DIR"
"$PYTHON" -m PyInstaller \
  --onedir --name "$NAME" \
  --noupx \
  --console --target-arch arm64 \
  "${HIDDEN_DEPS[@]}" \
  "$SCRIPT_DIR/mlx_vlm_gateway/cli.py"
popd

"$BUILD_DIR/dist/$NAME/$NAME" --help
"$BUILD_DIR/dist/$NAME/$NAME" daemon --help
"$BUILD_DIR/dist/$NAME/$NAME" worker --help

# serverctl.sh drives the server over HTTP and looks for the binary beside
# itself, so the archive root is where it belongs.
install -m 755 "$SCRIPT_DIR/serverctl.sh" "$BUILD_DIR/dist/$NAME/serverctl.sh"

# Read the output, not the status: with no command serverctl.sh prints usage
# and exits 1.
CTL_USAGE="$("$BUILD_DIR/dist/$NAME/serverctl.sh" 2>&1 || true)"
case "$CTL_USAGE" in
  *"./serverctl.sh start"*) ;;
  *) echo "ERROR: serverctl.sh in the archive does not run" >&2; exit 1 ;;
esac

mkdir -p "$(dirname "$ARCHIVE")"
tar -czf "$ARCHIVE" -C "$BUILD_DIR/dist" "$NAME"

# pushd: keeps the path inside the .sha256 file relative.
pushd "$(dirname "$ARCHIVE")"
shasum -a 256 "$(basename "$ARCHIVE")" | tee "$(basename "$ARCHIVE").sha256"
popd

rm -rf "$BUILD_DIR"
echo "Built $ARCHIVE"
