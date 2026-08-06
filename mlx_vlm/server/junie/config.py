"""Load the single persistent config used by the Junie gateway and worker."""

import json
import logging
import os
from typing import Optional

from mlx_vlm_shared.server_settings import DEFAULT_CONFIG, normalize_config


logger = logging.getLogger("mlx_vlm.server")

CONFIG_PATH_ENV = "JUNIE_SERVER_CONFIG"
DEFAULT_KV_QUANT_BITS = 8


def config_path() -> Optional[str]:
    return os.environ.get(CONFIG_PATH_ENV) or None


def _sanitize(raw: dict) -> dict:
    """Apply the shared defaults and validation rules."""
    config, invalid = normalize_config(raw)
    for key, value in invalid.items():
        logger.warning(
            "Config: ignoring invalid %r value %r (using %r)",
            key,
            value,
            config[key],
        )
    return config


def _write(path: str, config: dict) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = os.path.join(directory, f".{os.path.basename(path)}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def load_config() -> Optional[dict]:
    """Read and normalize the config, creating it when it is missing."""
    path = config_path()
    if not path:
        return None

    try:
        with open(path, encoding="utf-8") as stream:
            raw = json.load(stream)
        if not isinstance(raw, dict):
            raise ValueError("config root must be a JSON object")
    except FileNotFoundError:
        logger.info("Config: %s not found; creating it with defaults.", path)
        config = dict(DEFAULT_CONFIG)
        _write(path, config)
        return config
    except (OSError, ValueError) as exc:
        logger.warning(
            "Config: cannot read %s (%s); replacing it with defaults.",
            path,
            exc,
        )
        config = dict(DEFAULT_CONFIG)
        _write(path, config)
        return config

    config = _sanitize(raw)
    if config != raw:
        try:
            _write(path, config)
            logger.info("Config: normalized invalid or missing settings in %s", path)
        except OSError as exc:
            logger.warning("Config: failed to normalize %s: %s", path, exc)
    return config


def save_settings(updates: dict) -> None:
    """Merge settings into the config file when called by a standalone worker."""
    path = config_path()
    if not path:
        return
    config = load_config() or dict(DEFAULT_CONFIG)
    config = _sanitize({**config, **updates})
    try:
        _write(path, config)
        logger.info("Config: saved %s to %s", sorted(updates), path)
    except OSError as exc:
        logger.warning("Config: failed to save settings to %s: %s", path, exc)


def apply_config_to_env(config: dict) -> None:
    """Translate config values into the environment read by the worker."""

    def set_or_unset(name, value):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)

    set_or_unset("MLX_VLM_PRELOAD_MODEL", config.get("model_name"))
    set_or_unset("MLX_VLM_DRAFT_MODEL", config.get("draft_model"))
    set_or_unset(
        "MLX_VLM_DRAFT_KIND",
        config.get("draft_kind") if config.get("draft_model") else None,
    )
    set_or_unset("MAX_KV_SIZE", config.get("max_context_length"))
    set_or_unset(
        "KV_BITS",
        DEFAULT_KV_QUANT_BITS if config.get("kv_quantization") else None,
    )
    # The gateway owns idle timing and stops the whole worker process.
    os.environ.pop("MLX_VLM_AUTO_UNLOAD_TIME", None)


def initialize_from_config() -> None:
    """Load the config once before model preload and export it to the worker."""
    config = load_config()
    if config is None:
        return
    logger.info(
        "Config: %s -> model=%s draft=%s max_context_length=%s "
        "kv_quantization=%s auto_unload_time=%s",
        config_path(),
        config.get("model_name"),
        config.get("draft_model"),
        config.get("max_context_length"),
        config.get("kv_quantization"),
        config.get("auto_unload_time"),
    )
    apply_config_to_env(config)
