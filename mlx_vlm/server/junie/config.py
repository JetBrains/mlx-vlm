"""Load the single persistent config used by the Junie gateway and worker."""

import json
import logging
import os
from typing import Optional

from mlx_vlm_shared.server_settings import DEFAULT_CONFIG, normalize_config


logger = logging.getLogger("mlx_vlm.server")

CONFIG_PATH_ENV = "JUNIE_SERVER_CONFIG"


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


def load_config() -> Optional[dict]:
    """Read and normalize the gateway-owned config without changing it."""
    path = config_path()
    if not path:
        return None

    try:
        with open(path, encoding="utf-8") as stream:
            raw = json.load(stream)
        if not isinstance(raw, dict):
            raise ValueError("config root must be a JSON object")
    except FileNotFoundError:
        logger.warning("Config: %s not found; using defaults in memory.", path)
        return dict(DEFAULT_CONFIG)
    except (OSError, ValueError) as exc:
        logger.warning(
            "Config: cannot read %s (%s); using defaults in memory.",
            path,
            exc,
        )
        return dict(DEFAULT_CONFIG)

    return _sanitize(raw)

