"""The single Junie server config file: location, defaults, validation.

Both processes are configured only by this file: the daemon
(``python -m mlx_vlm_gateway``) and the inference worker it spawns
(``python -m mlx_vlm.server.junie``) take no command-line arguments and
resolve the same path through :func:`config_path`.
"""

import json
import logging
import os
from typing import Optional


logger = logging.getLogger("mlx_vlm.config")

CONFIG_PATH_ENV = "JUNIE_SERVER_CONFIG"
DEFAULT_CONFIG_PATH = "~/.local/share/junie-local/server-config.json"

PUBLIC_SETTING_KEYS = (
    "model_name",
    "max_context_length",
    "kv_quantization",
    "auto_unload_time",
)
RESTART_SETTING_KEYS = {
    "model_name",
    "max_context_length",
    "kv_quantization",
}

DEFAULT_CONFIG = {
    "model_name": "mlx-community/Qwen3.6-27B-4bit",
    # Multi-token-prediction speculative-decoding drafter for the model
    # above. It has no standalone language_model head, so it is only ever
    # served as the drafter, never requested directly as a chat "model".
    # Measured ~1.45-1.6x decode speedup at 2.24 accepted tokens/round; the
    # drafter itself costs only ~9% of a round — the rest is the 3-token
    # verify forward. Do not add a draft block size: the sweep
    # (research/mtp-overhead) showed the configured depth 3 is optimal
    # (2/4/5/6 are all slower) and the adaptive controller already handles
    # bursts. Per-request acceptance shows up in the log as
    # "Speculative decode: ... accepted_tokens_per_round=".
    "draft_model": "mlx-community/Qwen3.6-27B-MTP-4bit",
    "draft_kind": "mtp",
    # Where both models are installed: a Hugging Face hub-style cache dir
    # (models--org--name/snapshots/...) outside the default HF location.
    # The worker loads by repo id from here, so "/v1/models" reports a
    # clean id instead of a raw filesystem path.
    "models_dir": "~/.local/share/junie-local/models",
    "max_context_length": None,
    "kv_quantization": True,
    "auto_unload_time": 600,
    # Seconds after which the worker ends a single request cleanly, before
    # Junie's five-minute retry window. Keep it under the daemon's 275s hard
    # limit for workers that cannot acknowledge cancellation; null disables
    # the soft stop and leaves only that hard limit.
    "soft_request_timeout": 270,
    # One address for both processes: the daemon serves the public API on
    # "port", and the worker it spawns serves the private inference API on
    # "worker_port".
    "host": "0.0.0.0",
    "port": 19239,
    "worker_port": 19240,
    "int8_prefill": True,
    "prefill_step_size": 1024,
    "preserve_thinking": True,
    # Path to a chat-completions body whose prompt prefix is prefilled and
    # pinned at startup. Off by default: seeding is being reworked, and the
    # checkout-relative research/junie.json it used to default to does not
    # exist in a packaged build.
    "seed_request": None,
    "log_raw_tokens": True,
    "apc_enabled": True,
    "apc_exact_sessions": 4,
    "apc_session_checkpoints": 15,
    "apc_disk_path": None,
    "ngram_max": 8,
}

DEFAULT_PUBLIC_SETTINGS = {key: DEFAULT_CONFIG[key] for key in PUBLIC_SETTING_KEYS}
DEFAULT_DRAFT_MODEL = DEFAULT_CONFIG["draft_model"]


def _is_positive_int_or_none(value) -> bool:
    return value is None or (
        isinstance(value, int) and not isinstance(value, bool) and value > 0
    )


def _is_int_in(low, high):
    def check(value):
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and low <= value <= high
        )

    return check


_VALIDATORS = {
    "model_name": lambda value: (
        value is None or (isinstance(value, str) and bool(value.strip()))
    ),
    "draft_model": lambda value: (
        value is None or (isinstance(value, str) and bool(value.strip()))
    ),
    "draft_kind": lambda value: value is None or value in ("dflash", "eagle3", "mtp"),
    "models_dir": lambda value: isinstance(value, str) and bool(value.strip()),
    "max_context_length": _is_positive_int_or_none,
    "kv_quantization": lambda value: isinstance(value, bool),
    "auto_unload_time": _is_positive_int_or_none,
    "soft_request_timeout": _is_positive_int_or_none,
    "host": lambda value: isinstance(value, str) and bool(value.strip()),
    "port": _is_int_in(1, 65535),
    "worker_port": _is_int_in(1, 65535),
    "int8_prefill": lambda value: isinstance(value, bool),
    "prefill_step_size": _is_int_in(1, 1 << 20),
    "preserve_thinking": lambda value: isinstance(value, bool),
    "seed_request": lambda value: value is None or isinstance(value, str),
    "log_raw_tokens": lambda value: isinstance(value, bool),
    "apc_enabled": lambda value: isinstance(value, bool),
    "apc_exact_sessions": _is_int_in(0, 64),
    "apc_session_checkpoints": _is_int_in(1, 64),
    "apc_disk_path": lambda value: value is None or isinstance(value, str),
    "ngram_max": _is_int_in(1, 1024),
}


def is_valid_setting(key: str, value) -> bool:
    """Return whether a known setting has a valid value."""
    validator = _VALIDATORS.get(key)
    return validator is not None and bool(validator(value))


def normalize_config(raw: dict) -> tuple[dict, dict]:
    """Return a safe full config and the invalid known values that were dropped."""
    normalized = dict(DEFAULT_CONFIG)
    invalid = {}
    for key, value in raw.items():
        validator = _VALIDATORS.get(key)
        if validator is None:
            normalized[key] = value
        elif validator(value):
            normalized[key] = value
        else:
            invalid[key] = value
    return normalized, invalid


def config_path() -> str:
    """The config file both processes read, overridable by the environment."""
    raw = os.environ.get(CONFIG_PATH_ENV) or DEFAULT_CONFIG_PATH
    # Absolute, so the daemon and the worker it spawns agree on the file and
    # on the APC cache directory derived from the directory holding it.
    return os.path.abspath(os.path.expanduser(raw))


def load_config(path: Optional[str] = None) -> dict:
    """Read and normalize the config without changing it on disk.

    Any problem — missing file, unreadable file, bad values — degrades to
    the defaults for the affected keys, so a broken config can never keep
    the server from starting.
    """
    path = path or config_path()
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
            "Config: cannot read %s (%s); using defaults in memory.", path, exc
        )
        return dict(DEFAULT_CONFIG)

    config, invalid = normalize_config(raw)
    for key, value in invalid.items():
        logger.warning(
            "Config: ignoring invalid %r value %r (using %r)", key, value, config[key]
        )
    return config
