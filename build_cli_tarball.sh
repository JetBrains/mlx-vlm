#!/usr/bin/env bash
set -euo pipefail

# Freeze the junie-mlx-vlm dispatcher into a self-contained tar.gz, from
# this checkout's own mlx_vlm_gateway/cli.py -- no wrapper script.
#
#   Output:  dist/junie-mlx-vlm-<version>-macos-<arch>.tar.gz
#   Run:     junie-mlx-vlm/junie-mlx-vlm
#
# The program takes no options. Both the daemon and the inference worker it
# spawns read every setting from JUNIE_SERVER_CONFIG (default
# ~/.local/share/junie-local/server-config.json), and one executable serves
# both because the daemon re-invokes it as `junie-mlx-vlm worker` -- a
# frozen build has no interpreter to hand `-m` to.
#
# That entry file works here because its imports are absolute. PyInstaller
# runs its entry script in script mode, where __package__ is empty and a
# relative import (`from .app import main`, as the two __main__.py files
# have it) raises "attempted relative import with no known parent package";
# `python -m` avoids that only because runpy imports the package and sets
# __package__ first.
#
# CPython, the build environment, PyInstaller's scratch space and the frozen
# tree all live in one mktemp directory that is deleted on exit, so the only
# thing this leaves behind is the tarball, and it reads nothing a previous
# build left behind -- the repo's .venv and .uv are neither used nor touched.
# The price: every run re-installs the interpreter (~3 s from uv's cache) and
# re-resolves requirements.txt, with no incremental PyInstaller cache, so a
# build is a full ~2 minutes; and two tarballs built weeks apart may not hold
# identical dependency versions.
#
# The only thing taken from outside is the `uv` binary itself, if the machine
# already has one; otherwise that is fetched into the temp dir too.
#
# Nothing is excluded, so the archive also carries OpenCV and SciPy (~150 MB)
# for video inputs and mlx_audio's resampler.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

APP_NAME="junie-mlx-vlm"
# This binary's own version, not the upstream mlx-vlm one in
# mlx_vlm/version.py that the Python package still carries.
VERSION="$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' mlx_vlm_gateway/version.py)"
TARBALL="$SCRIPT_DIR/dist/$APP_NAME-$VERSION-macos-$(uname -m).tar.gz"
PYTHON_VERSION=3.13

TMP_ROOT="$(mktemp -d -t mlx-vlm-build)"
trap 'rm -rf "$TMP_ROOT"' EXIT
BUILD_VENV="$TMP_ROOT/venv"
PY="$BUILD_VENV/bin/python"
STAGE="$TMP_ROOT/dist"
export UV_PYTHON_INSTALL_DIR="$TMP_ROOT/python"
echo "Building in $TMP_ROOT (removed on exit)"

find_uv() {
  command -v uv 2>/dev/null && return 0
  for candidate in "$SCRIPT_DIR/.uv/bin/uv" "$HOME/.local/bin/uv"; do
    if [ -x "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

if ! UV_BIN="$(find_uv)"; then
  echo "Installing uv into the temp dir ..."
  curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR="$TMP_ROOT/uv/bin" INSTALLER_NO_MODIFY_PATH=1 sh
  UV_BIN="$TMP_ROOT/uv/bin/uv"
fi

# uv downloads a managed CPython: stock macOS python3 is 3.9 and mlx>=0.32
# ships wheels for 3.10+ only.
echo "Creating a throwaway build environment (Python $PYTHON_VERSION) ..."
"$UV_BIN" venv --python "$PYTHON_VERSION" "$BUILD_VENV"
"$UV_BIN" pip install --quiet --python "$PY" -r requirements.txt pyinstaller

# mlx.core is a compiled extension; ask it where its lib/ lives rather than
# assuming a site-packages layout.
MLX_LIB_DIR="$("$PY" -c 'import mlx.core, pathlib; print(pathlib.Path(mlx.core.__file__).resolve().parent / "lib")')"
MLX_JACCL="$MLX_LIB_DIR/libjaccl.dylib"
if [ ! -f "$MLX_JACCL" ]; then
  echo "ERROR: $MLX_JACCL is missing; mlx's library layout changed." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Freeze. PyInstaller finds dependencies by following import statements from
# the entry script -- including ones nested inside functions -- so cv2, scipy,
# transformers, tokenizers, miniaudio and the rest need no flags at all, and
# libmlx.dylib arrives too because it scans Mach-O load commands. The
# dispatcher leans on the nested case: it imports the daemon and the worker
# only inside the branch that runs them, and both halves still land here.
# What it cannot see, and each flag below covers:
#
#   mlx._reprlib_fix is imported by mlx.core's C++ extension, from machine code
#   where there is no bytecode to analyse. (The extension names five modules;
#   this is the only one that exists in the macOS wheel -- mlx.cuda, mlx.metal
#   and mlx.gc_func do not. mlx.nn and mlx.utils are reached normally, through
#   mlx_vlm's own imports.)
#
#   mlx.metallib is a 156 MB Metal shader library that libmlx.dylib opens at
#   runtime by looking next to itself, so it must land in mlx/lib/ -- and being
#   a data file, no import points at it.
#
#   libjaccl.dylib is a load command of libmlx.dylib, but written as
#   @rpath/libjaccl.dylib while libmlx.dylib carries no LC_RPATH at all: the
#   search path comes from core.cpython-313-darwin.so, one level up, whose
#   @loader_path/lib covers both. Scanning libmlx.dylib on its own, PyInstaller
#   cannot resolve that and leaves the library out, so `import mlx.core` in the
#   bundle dies with "Library not loaded: @rpath/libjaccl.dylib". It goes to
#   _internal/ rather than next to libmlx.dylib because that is where every
#   rpath PyInstaller rewrote points.
#
#   Architectures, drafters, tool parsers and chat templates are resolved as
#   importlib.import_module(f"...{name}") from strings that exist only once the
#   model's config.json has been read (utils.get_model_and_args). Sweeping the
#   packages is not laziness here: model_type is data that ships with the
#   weights, so at build time the right module genuinely cannot be known.
#
#   llguidance reads version("llguidance") while importing, and without its
#   .dist-info the PackageNotFoundError -- a subclass of ImportError -- makes
#   the server answer 400 "llguidance is required" to structured requests.
#
# The three mlx_lm sweeps are the only lines here that Qwen3.6 does not need;
# they cost 0.3 MB and cover text-only models routed through mlx_lm's own
# resolver, and tokenizer configs that set chat_template_type.
# ---------------------------------------------------------------------------
"$PY" -m PyInstaller --noconfirm --noupx \
  --onedir --console --target-arch arm64 \
  --name "$APP_NAME" \
  --distpath "$STAGE" \
  --workpath "$TMP_ROOT/pyinstaller" \
  --specpath "$TMP_ROOT" \
  --paths "$SCRIPT_DIR" \
  --hidden-import mlx._reprlib_fix \
  --add-data "$MLX_LIB_DIR/mlx.metallib:mlx/lib" \
  --add-binary "$MLX_JACCL:." \
  --collect-submodules mlx_vlm.models \
  --collect-submodules mlx_vlm.speculative \
  --collect-submodules mlx_vlm.tool_parsers \
  --collect-submodules mlx_lm.models \
  --collect-submodules mlx_lm.tool_parsers \
  --collect-submodules mlx_lm.chat_templates \
  --copy-metadata llguidance \
  mlx_vlm_gateway/cli.py

# ---------------------------------------------------------------------------
# Pack with tar, not zip: the tree contains symlinked dylibs that `zip -r`
# would replace with copies. COPYFILE_DISABLE keeps xattrs out of the archive
# so extractions carry no ._* members; the ad-hoc signatures PyInstaller
# applies live inside the Mach-O headers and survive either way.
# ---------------------------------------------------------------------------
mkdir -p "$(dirname "$TARBALL")"
echo
echo "Packing $TARBALL ..."
COPYFILE_DISABLE=1 tar -czf "$TARBALL" -C "$STAGE" "$APP_NAME"

# ---------------------------------------------------------------------------
# Verify the archive that ships, unpacked somewhere else.
# ---------------------------------------------------------------------------
CHECK_DIR="$TMP_ROOT/verify"
mkdir -p "$CHECK_DIR"
tar -xzf "$TARBALL" -C "$CHECK_DIR"
APP="$CHECK_DIR/$APP_NAME"

# Python itself: the interpreter shared library plus a zipped stdlib. uv's
# managed CPython is a framework build, so the library arrives as
# Python.framework/Versions/<x.y>/Python; a plain build would leave a
# libpython<x.y>.dylib instead, and either is a working interpreter.
INTERPRETER=""
for candidate in \
  "$APP/_internal/Python.framework/Versions/$PYTHON_VERSION/Python" \
  "$APP/_internal/libpython$PYTHON_VERSION.dylib"; do
  if [ -f "$candidate" ]; then
    INTERPRETER="$candidate"
    break
  fi
done
if [ -z "$INTERPRETER" ]; then
  echo "ERROR: no interpreter library in the archive ($APP/_internal)" >&2
  exit 1
fi
for required in "$INTERPRETER" "$APP/_internal/base_library.zip"; do
  [ -f "$required" ] || { echo "ERROR: $required missing from the archive" >&2; exit 1; }
  echo "Interpreter: ${required#"$APP/_internal/"} ($(du -h "$required" | awk '{print $1}'))"
done

# Dependencies that only load lazily never show up in a --help run, so check
# that their files are in the bundle at all.
for package in cv2 scipy llguidance PIL numpy transformers tokenizers hf_xet; do
  [ -e "$APP/_internal/$package" ] \
    || { echo "ERROR: $package missing from the archive" >&2; exit 1; }
done
echo "Bundled: cv2 scipy llguidance PIL numpy transformers tokenizers hf_xet"

# One --help per subcommand, because they import different halves of the
# bundle: the daemon pulls fastapi, starlette, uvicorn and httpx, and only
# the worker pulls mlx, mlx_lm, transformers and the tokenizer. Checking one
# would let the other ship broken -- and the bare --help, answered by the
# dispatcher itself, imports neither, so it only proves the entry point runs.
"$APP/$APP_NAME" --help >/dev/null
echo "Ran: $APP_NAME --help (dispatcher)"
"$APP/$APP_NAME" daemon --help >/dev/null
echo "Ran: $APP_NAME daemon --help (fastapi, uvicorn)"
"$APP/$APP_NAME" worker --help >/dev/null
echo "Ran: $APP_NAME worker --help (mlx, transformers, tokenizer)"

echo
echo "Built $TARBALL"
echo "  $(du -sh "$TARBALL" | awk '{print $1}') packed, \
$(du -sh "$STAGE/$APP_NAME" | awk '{print $1}') unpacked"
shasum -a 256 "$TARBALL"
echo
echo "On the target Mac:"
echo "  curl -fsSL <url>/$(basename "$TARBALL") | tar -xz -C <dir>"
echo "  <dir>/$APP_NAME/$APP_NAME"
echo "It takes no options; settings come from JUNIE_SERVER_CONFIG (default"
echo "~/.local/share/junie-local/server-config.json), and the model weights"
echo "named there must already be installed."
echo "Browser-downloaded instead of curl'd? macOS SIGKILLs quarantined binaries:"
echo "  xattr -dr com.apple.quarantine <dir>/$APP_NAME"
