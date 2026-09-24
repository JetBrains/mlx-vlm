"""Splash worker owned by the Junie gateway, with a bundled offline runtime."""

import argparse
import json
import os
import signal
import subprocess
from pathlib import Path


def runtime_paths(config):
    """Installed bundles resolve beside the launcher, never through PATH."""
    bundled = os.environ.get("JUNIE_SPLASH_RUNTIME")
    if bundled:
        root = Path(bundled).resolve()
        return root / "python/bin/python3", root, root / "engine/splash"
    source = Path(config.get("splash_source", "")).expanduser().resolve()
    return (
        Path(config.get("splash_python", "")).expanduser(),
        source,
        source / "build/splash",
    )


def package_model(config):
    package = Path(config["splash_package"]).expanduser().resolve()
    manifest = json.loads((package / "manifest.json").read_text())
    model = manifest.get("model")
    if not isinstance(model, str) or len(model.split("/")) != 2:
        raise ValueError("Splash manifest must name an owner/repo model")
    return model


def validate_config(config):
    if config.get("worker_backend", "mlx") != "splash":
        return
    for key in (
        ("splash_package",)
        if os.environ.get("JUNIE_SPLASH_RUNTIME")
        else ("splash_python", "splash_source", "splash_package")
    ):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Splash requires {key}")
    python, source, binary = runtime_paths(config)
    package = Path(config["splash_package"]).expanduser()
    for path in (
        python,
        source / "server/server.py",
        binary,
        package / "manifest.json",
        package / "target",
        package / "draft",
        package / "vision",
        package / "tokenizer",
    ):
        if not path.exists():
            raise ValueError(f"Missing Splash runtime/package path: {path}")
    package_model(config)
    limit = config.get("splash_max_memory_bytes")
    if limit is not None and (type(limit) is not int or not 0 < limit <= 2**63 - 1):
        raise ValueError("splash_max_memory_bytes must be a positive integer byte count")
    # The benchmark-pinned f58d36d runtime has fixed INT8 KV. Never claim
    # that the UI's BF16 setting was applied while silently retaining INT8.
    if config.get("kv_quantization") is not True:
        raise ValueError("This Splash adapter requires kv_quantization=true (INT8)")


def command(config):
    validate_config(config)
    python, source, binary = runtime_paths(config)
    package = Path(config["splash_package"]).expanduser().resolve()
    args = [
        str(python.absolute()),
        str(source / "server/server.py"),
        str(package / "target"),
        str(package / "draft"),
        "--tokenizer",
        str(package / "tokenizer"),
        "--model",
        package_model(config),
        "--binary",
        str(binary),
        "--host",
        "127.0.0.1",
        "--port",
        str(config["worker_port"]),
        "--no-webui",
    ]
    if config.get("splash_max_memory_bytes") is not None:
        args += ["--max-memory", str(config["splash_max_memory_bytes"])]
    if config.get("max_context_length") is not None:
        args += ["--max-context", str(config["max_context_length"])]
    if config.get("soft_request_timeout") is not None:
        args += ["--request-timeout", str(config["soft_request_timeout"])]
    return args


def supervise(args, env, parent_pid, grace_s=1.0):
    """Own the Splash HTTP/native process group, including on gateway death."""
    child = subprocess.Popen(args, env=env, start_new_session=True)
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    previous = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        while not stopping and os.getppid() == parent_pid:
            try:
                return child.wait(timeout=min(0.1, grace_s))
            except subprocess.TimeoutExpired:
                pass
        return 0
    finally:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            pass
        # Also reap any native descendant left behind by an HTTP crash.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    from mlx_vlm_shared.server_settings import load_config

    argparse.ArgumentParser(description=__doc__).parse_args()
    config = load_config()
    env = dict(os.environ)
    # Keep the credential out of argv, logs and the process listing.
    if config.get("api_key"):
        env["SPLASH_API_KEY"] = config["api_key"]
    else:
        env.pop("SPLASH_API_KEY", None)
    env["HF_HUB_OFFLINE"] = "1"
    parent_pid = int(env.get("MLX_VLM_GATEWAY_PID", os.getppid()))
    raise SystemExit(
        supervise(
            command(config), env, parent_pid, min(1.0, config["shutdown_timeout_s"] / 4)
        )
    )
