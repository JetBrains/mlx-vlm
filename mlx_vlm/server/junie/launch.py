"""Launch the server from the persistent config file.

``python -m mlx_vlm.server.junie`` takes no arguments. It reads the
config file once (see :mod:`mlx_vlm_shared.server_settings`) and turns it
into the stock server's inputs: the launch-time settings (host/worker
port, prefill tuning, seed request, ...) become the equivalent
``mlx_vlm.server`` command line, and the runtime settings (model/drafter,
context length, KV quantization, ...) are exported as env vars — so every
serving setting lives in one file.

Normally the daemon (``python -m mlx_vlm_gateway``) spawns this; it also
exports the Hugging Face cache location, which has to be in place before
this process imports anything.
"""

import argparse
import logging
import os
import re
import subprocess
import sys
from typing import List, Optional

from mlx_vlm_shared.server_settings import (
    CONFIG_PATH_ENV,
    DEFAULT_CONFIG,
    DEFAULT_CONFIG_PATH,
    config_path,
    load_config,
)

from .parent_watchdog import start_parent_watchdog


logger = logging.getLogger("mlx_vlm.server")

DEFAULT_KV_QUANT_BITS = 8
# Must match mlx_vlm.server.cli, whose own basicConfig call is a no-op once
# this module has installed a root handler.
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def configure_logging() -> None:
    """Install the root log handler before anything here writes a record.

    ``mlx_vlm.server.cli`` configures logging too, but only after argparse
    has run — by which point the launcher has already reported the config
    it resolved, and with no handler installed those INFO records go to
    Python's last-resort handler, which drops anything below WARNING.
    """
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


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
    set_or_unset("MLX_VLM_SOFT_REQUEST_TIMEOUT", cfg.get("soft_request_timeout"))
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
    disk_path = cfg.get("apc_disk_path") or os.path.join(
        os.path.dirname(config_path()), "apc-cache"
    )
    set_or_unset("APC_DISK_PATH", os.path.expanduser(disk_path))
    set_or_unset("MLX_VLM_NGRAM_MAX", cfg.get("ngram_max"))
    # Always serialize request processing: the multi-request batching paths
    # are undertested with kv-bits, and Junie's traffic is sequential anyway.
    # Deliberately not configurable — hardcoded, ignoring any config value.
    os.environ["MLX_VLM_MAX_CONCURRENT_REQUESTS"] = "1"


def build_argv(cfg: dict) -> List[str]:
    argv = [
        # Same host as the daemon; the worker's own port keeps the private
        # inference API off the public one.
        "--host",
        str(cfg.get("host") or DEFAULT_CONFIG["host"]),
        "--port",
        str(cfg.get("worker_port") or DEFAULT_CONFIG["worker_port"]),
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
    # No implicit seed: the launcher used to fall back to the checkout's
    # research/junie.json, which a frozen bundle has no path to. Only an
    # explicit setting warms a prefix, and the default leaves it off until
    # seeding is reworked.
    if seed := cfg.get("seed_request"):
        argv.extend(["--seed-request", str(seed)])
    return argv


def main() -> None:
    argparse.ArgumentParser(
        description=(
            "MLX-VLM inference worker. Takes no arguments; every setting "
            f"comes from the config file (${CONFIG_PATH_ENV}, default "
            f"{DEFAULT_CONFIG_PATH})."
        )
    ).parse_args()
    configure_logging()
    start_parent_watchdog()
    from ..cli import main as cli_main

    cfg = load_config()
    initialize_from_config(cfg)
    apply_inference_env(cfg)
    sys.argv = [sys.argv[0], *build_argv(cfg)]
    cli_main()
