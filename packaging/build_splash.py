#!/usr/bin/env python3
"""Build a relocatable Junie gateway with independently pinned MLX and Splash workers.

Requires uv on the build host; the installed archive needs neither uv nor a
system Python. Model weights are distributed separately by the installer.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path)
    parser.add_argument("--mlx-archive", type=Path)
    args = parser.parse_args()
    pin = json.loads((ROOT / "packaging/splash-runtime.json").read_text())
    mlx_pin = json.loads((ROOT / "packaging/mlx-runtime.json").read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="junie-splash-build-") as temp:
        stage = Path(temp)
        archive = args.runtime_archive or stage / "runtime.tar.gz"
        if not args.runtime_archive:
            urllib.request.urlretrieve(pin["url"], archive)
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != pin["sha256"]:
            raise ValueError("Splash runtime checksum mismatch")
        extracted = stage / "upstream"
        with tarfile.open(archive) as tar:
            tar.extractall(extracted, filter="data")
        root = stage / "junie-mlx-vlm"
        root.mkdir()
        shutil.move(str(next(extracted.iterdir())), root / "splash")
        mlx_archive = args.mlx_archive or stage / "mlx.tar.gz"
        if not args.mlx_archive:
            urllib.request.urlretrieve(mlx_pin["url"], mlx_archive)
        with mlx_archive.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != mlx_pin["sha256"]:
                raise ValueError("MLX runtime checksum mismatch")
        with tarfile.open(mlx_archive) as tar:
            tar.extractall(root / "runtime", filter="data")
        (root / "runtime/junie-mlx-vlm").rename(root / "runtime/mlx")
        python = root / "splash/python/bin/python3"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--require-hashes",
                "--target",
                str(root / "gateway-deps"),
                "--python",
                str(python),
                "-r",
                str(ROOT / "packaging/gateway-requirements.lock"),
            ],
            check=True,
        )
        for name in ("mlx_vlm_gateway", "mlx_vlm_shared"):
            shutil.copytree(
                ROOT / name,
                root / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        shutil.copy2(ROOT / "serverctl.sh", root / "serverctl.sh")
        launcher = root / "junie-mlx-vlm"
        launcher.write_text("""#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$ROOT:$ROOT/gateway-deps"
export JUNIE_SPLASH_RUNTIME="$ROOT/splash"
export JUNIE_MLX_WORKER="$ROOT/runtime/mlx/junie-mlx-vlm"
exec "$ROOT/splash/python/bin/python3" -m mlx_vlm_gateway.cli "$@"
""")
        launcher.chmod(0o755)
        (root / "splash-runtime.json").write_text(json.dumps(pin, indent=2) + "\n")
        (root / "build-info.json").write_text(
            json.dumps(
                {
                    "gateway_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                    ).strip(),
                    "gateway_dirty": bool(
                        subprocess.check_output(
                            ["git", "status", "--porcelain"], cwd=ROOT, text=True
                        ).strip()
                    ),
                    "gateway_dependencies_sha256": hashlib.sha256(
                        (ROOT / "packaging/gateway-requirements.lock").read_bytes()
                    ).hexdigest(),
                    "runtimes": {"mlx": mlx_pin, "splash": pin},
                },
                indent=2,
            )
            + "\n"
        )
        subprocess.run(
            [str(root / "runtime/mlx/junie-mlx-vlm"), "worker", "--help"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run([str(launcher), "--help"], check=True, stdout=subprocess.DEVNULL)
        with tarfile.open(args.output, "w:gz") as tar:
            tar.add(root, arcname=root.name)
    print(args.output)


if __name__ == "__main__":
    main()
