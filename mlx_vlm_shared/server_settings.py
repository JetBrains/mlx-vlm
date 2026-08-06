"""Pure validation rules for the single Junie server config file."""

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
    "draft_model": "mlx-community/Qwen3.6-27B-MTP-4bit",
    "draft_kind": "mtp",
    "max_context_length": None,
    "kv_quantization": True,
    "auto_unload_time": 600,
    "host": "0.0.0.0",
    "port": 19239,
    "int8_prefill": True,
    "prefill_step_size": 1024,
    "preserve_thinking": True,
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
    "max_context_length": _is_positive_int_or_none,
    "kv_quantization": lambda value: isinstance(value, bool),
    "auto_unload_time": _is_positive_int_or_none,
    "host": lambda value: isinstance(value, str) and bool(value.strip()),
    "port": _is_int_in(1, 65535),
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
