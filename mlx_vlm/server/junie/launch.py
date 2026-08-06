"""Launch the server from the persistent config file.

``python -m mlx_vlm.server.junie`` reads ``JUNIE_SERVER_CONFIG`` once and
turns it into the stock server's inputs: the launch-time settings
(host/port, prefill tuning, seed request, ...) become the equivalent
``mlx_vlm.server`` command line, and the runtime settings (model/drafter,
context length, KV quantization, ...) are exported as env vars — so
start.sh stays a dumb bootstrapper and every serving setting lives in one
file.

Extra command-line arguments are appended after the config-derived ones
(argparse last-wins), so ad-hoc overrides still work:

    python -m mlx_vlm.server.junie --log-level DEBUG
"""

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .config import DEFAULT_CONFIG, config_path, load_config
from .parent_watchdog import start_parent_watchdog


logger = logging.getLogger("mlx_vlm.server")

DEFAULT_KV_QUANT_BITS = 8


def _apple_chip_generation() -> Optional[int]:
    """Return the Apple M-series generation reported by macOS."""
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"\bApple M(\d+)\b", result.stdout)
    return int(match.group(1)) if match else None


def _default_seed_request() -> Optional[str]:
    """The repo's bundled Junie seed prompt, when running from a checkout."""
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "research" / "junie.json"
    return str(path) if path.is_file() else None


def apply_config_to_env(cfg: dict) -> None:
    """Translate config values into the environment read by the worker."""

    def set_or_unset(name, value):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)

    set_or_unset("MLX_VLM_PRELOAD_MODEL", cfg.get("model_name"))
    set_or_unset("MLX_VLM_DRAFT_MODEL", cfg.get("draft_model"))
    set_or_unset(
        "MLX_VLM_DRAFT_KIND",
        cfg.get("draft_kind") if cfg.get("draft_model") else None,
    )
    set_or_unset("MAX_KV_SIZE", cfg.get("max_context_length"))
    set_or_unset(
        "KV_BITS",
        DEFAULT_KV_QUANT_BITS if cfg.get("kv_quantization") else None,
    )
    # The gateway owns idle timing and stops the whole worker process.
    os.environ.pop("MLX_VLM_AUTO_UNLOAD_TIME", None)


def initialize_from_config(cfg: dict) -> None:
    """Export the runtime settings from the config to the env.

    Called before the server starts, so the config file — not command-line
    flags — decides which model to serve and with which settings.
    """
    logger.info(
        "Config: %s -> model=%s draft=%s max_context_length=%s "
        "kv_quantization=%s auto_unload_time=%s",
        config_path(),
        cfg.get("model_name"),
        cfg.get("draft_model"),
        cfg.get("max_context_length"),
        cfg.get("kv_quantization"),
        cfg.get("auto_unload_time"),
    )
    apply_config_to_env(cfg)


def apply_inference_env(cfg: dict) -> None:
    """Export the inference env vars (APC, ngram cap) from the config.

    These are read by the runtime at model-load / draft time, not parsed
    as server flags, so the launcher sets them before the server starts.
    """

    def set_or_unset(name, value):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)

    set_or_unset("APC_ENABLED", "1" if cfg.get("apc_enabled") else "0")
    set_or_unset("APC_EXACT_SESSIONS", cfg.get("apc_exact_sessions"))
    set_or_unset("APC_SESSION_CHECKPOINTS", cfg.get("apc_session_checkpoints"))
    disk_path = cfg.get("apc_disk_path")
    if disk_path is None:
        base = config_path()
        disk_path = (
            os.path.join(os.path.dirname(base), "apc-cache") if base else None
        )
    set_or_unset(
        "APC_DISK_PATH", os.path.expanduser(disk_path) if disk_path else None
    )
    set_or_unset("MLX_VLM_NGRAM_MAX", cfg.get("ngram_max"))
    # Always serialize request processing: the multi-request batching paths
    # are undertested with kv-bits, and Junie's traffic is sequential anyway.
    # Deliberately not configurable — hardcoded, ignoring any config value.
    os.environ["MLX_VLM_MAX_CONCURRENT_REQUESTS"] = "1"


def build_argv(cfg: dict) -> List[str]:
    argv = [
        "--host",
        str(cfg.get("host") or DEFAULT_CONFIG["host"]),
        "--port",
        str(cfg.get("port") or DEFAULT_CONFIG["port"]),
        "--prefill-step-size",
        str(cfg.get("prefill_step_size") or DEFAULT_CONFIG["prefill_step_size"]),
        # Quantize the KV cache from token 0 when kv_quantization is on
        # (the stock default, 5000, keeps contexts below that threshold
        # fp16 — pointless for Junie, whose requests are all 10k+, and it
        # creates a second, undertested fp16/quantized regime). Ignored
        # when KV_BITS is unset. Must be a flag, not an env export:
        # cli.py writes QUANTIZED_KV_START from the flag unconditionally.
        "--quantized-kv-start",
        "0",
    ]
    if cfg.get("int8_prefill") and (_apple_chip_generation() or 0) >= 5:
        argv.append("--int8-prefill")
    elif cfg.get("int8_prefill"):
        logger.warning(
            "int8 prefill requires Apple M5 or newer; using standard prefill."
        )
    if cfg.get("preserve_thinking"):
        argv.append("--preserve-thinking")
    if cfg.get("log_raw_tokens"):
        argv.append("--log-raw-tokens")
    seed = cfg.get("seed_request")
    if seed is None:
        seed = _default_seed_request()
    if seed:
        argv.extend(["--seed-request", str(seed)])
    return argv


def main() -> None:
    start_parent_watchdog()
    from ..cli import main as cli_main

    cfg = load_config()
    if cfg is None:
        # JUNIE_SERVER_CONFIG not set: fall back to stock flag behavior.
        cli_main()
        return
    initialize_from_config(cfg)
    apply_inference_env(cfg)
    sys.argv = [sys.argv[0], *build_argv(cfg), *sys.argv[1:]]
    cli_main()
