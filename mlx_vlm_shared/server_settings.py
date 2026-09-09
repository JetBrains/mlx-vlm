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
    "model_name": "Qwen3.6-27B-MLX-4bit",
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
    "draft_model": "Qwen3.6-27B-MTP-MLX-4bit",
    "draft_kind": "mtp",
    # Where both models are installed, outside the default HF location.
    # Either layout works: a plain directory named after the model (the
    # Qwen3.8+ install layout, resolved by get_model_path via HF_HUB_CACHE)
    # or a Hugging Face hub-style cache dir (models--org--name/snapshots/...)
    # loaded by repo id. Both keep "/v1/models" reporting a clean id instead
    # of a raw filesystem path.
    "models_dir": "~/.local/share/junie-local/models",
    "max_context_length": None,
    "kv_quantization": False,
    "auto_unload_time": 60,
    # Seconds after which the worker ends a single request cleanly, before
    # Junie's five-minute retry window. Keep it under the daemon's 275s hard
    # limit for workers that cannot acknowledge cancellation; null disables
    # the soft stop and leaves only that hard limit.
    "soft_request_timeout": 270,
    # One address for both processes: the daemon serves the public API on
    # "port", and the worker it spawns serves the private inference API on
    # "worker_port".
    "host": "127.0.0.1",
    "port": 19239,
    "worker_port": 19240,
    # The bearer token the daemon requires on every request, and hands to
    # the worker it spawns (MLX_VLM_SERVER_API_KEY, which the worker
    # already requires on every endpoint): install.sh generates one per
    # machine and writes it here. null leaves both APIs open, which is what
    # a checkout that never ran install.sh gets. Changing it needs a
    # restart, like "host" and "port", so it is not an /apply_settings
    # field.
    "api_key": None,
    "int8_prefill": True,
    "prefill_step_size": 1024,
    # Serve chat requests that name a model the gateway does not know with
    # the configured model instead of answering 404 model_not_found. Coding
    # agents send whatever id sits in their profile (Junie, Continue,
    # Cursor) or their built-in defaults (sub-agents ask for their vendor's
    # cloud model ids), and a 404 leaves them retrying.
    # Supported model names still switch the worker as before. false keeps
    # the strict 404.
    "model_alias": True,
    # Cap on the MLX allocator's freed-buffer cache in GB
    # (MLX_VLM_CACHE_LIMIT_GB). Long prefills leave multi-GB chunk logits
    # and superseded batch caches in that pool; a small cap trims the
    # process peak by ~15 GB on a 27B model at no measurable speed cost.
    # null keeps the MLX default (no cap).
    "mlx_cache_limit_gb": 3,
    "preserve_thinking": True,
    # How many stable Junie prompt prefixes (the messages before the
    # issue-description message) to keep pinned: each one's KV is
    # snapshotted during a request's own prefill and persisted to the APC
    # disk tier, so new sessions — including right after a restart — start
    # warm at that boundary instead of re-prefilling ~15k tokens. A prompt
    # change simply pins a new snapshot while the least recently used one
    # ages out. 0 disables pinning.
    "pin_stable_prefix": 5,
    "log_raw_tokens": False,
    "apc_enabled": True,
    # Hybrid APC (mlx_vlm/apc_hybrid.py): attention K/V in shared 256-token
    # blocks plus a ladder of recurrent-state checkpoints from the end of
    # the prompt, instead of whole-cache exact snapshots. Continued
    # conversations recompute only their tail and the disk tier holds
    # blocks once. Needs apc_enabled; bypasses pin_stable_prefix.
    "apc_hybrid": False,
    "apc_exact_sessions": 2,
    "apc_session_checkpoints": 4,
    # How many growing conversations (each request a superset prefix of the
    # last) keep an exact-cache snapshot on disk at once. Each chain's older
    # snapshots are superseded by its newest one as it grows, so this caps
    # distinct conversations, not total snapshot files. Pooled separately
    # from "pin_stable_prefix" above, so a burst of session churn can't
    # evict the shared warm-start prefix (or vice versa). 0 disables writing
    # growing-session snapshots to disk at all.
    "apc_max_growing_sessions": 5,
    # null does not mean "no disk cache" — launch.py falls back to
    # "apc-cache" next to this config file (see config_path()), so the
    # disk tier is on by default. Set this only to relocate it, e.g. to a
    # faster disk or a separate volume.
    "apc_disk_path": None,
    "ngram_max": 8,
    # Daemon-side supervisor tuning: how long to wait for the worker to
    # start, how long a single proxied request may run before the daemon
    # gives up and restarts the worker, and the health-probe cadence. Keep
    # "request_timeout_s" above "soft_request_timeout" above, since that is
    # the worker's own softer per-request limit that should fire first.
    "startup_timeout_s": 120.0,
    "request_timeout_s": 275.0,
    "startup_probe_interval_s": 1.0,
    "probe_interval_s": 5.0,
    "probe_timeout_s": 2.0,
    "probe_failures_before_restart": 3,
    "max_start_failures": 3,
    "startup_retry_cooldown_s": 30.0,
    "restart_delay_s": 2.0,
    "shutdown_timeout_s": 5.0,
    "idle_check_interval_s": 1.0,
}

DEFAULT_PUBLIC_SETTINGS = {key: DEFAULT_CONFIG[key] for key in PUBLIC_SETTING_KEYS}
DEFAULT_DRAFT_MODEL = DEFAULT_CONFIG["draft_model"]

# The models the server can serve, each paired with its MTP drafter. Both
# live in "models_dir" with the same layout. A chat request naming one of
# these gets the worker (re)loaded with it; any other name is rejected.
SUPPORTED_MODELS = {
    "Qwen3.6-27B-MLX-4bit": "Qwen3.6-27B-MTP-MLX-4bit",
    "Qwen3.8-27B-MLX-4bit": "Qwen3.8-27B-MTP-MLX-4bit",
}


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


def _is_positive_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _is_positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


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
    "api_key": lambda value: (
        value is None or (isinstance(value, str) and bool(value.strip()))
    ),
    "int8_prefill": lambda value: isinstance(value, bool),
    "model_alias": lambda value: isinstance(value, bool),
    "prefill_step_size": _is_int_in(1, 1 << 20),
    "mlx_cache_limit_gb": lambda value: value is None or (
        isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
    ),
    "preserve_thinking": lambda value: isinstance(value, bool),
    "pin_stable_prefix": _is_int_in(0, 64),
    "log_raw_tokens": lambda value: isinstance(value, bool),
    "apc_enabled": lambda value: isinstance(value, bool),
    "apc_hybrid": lambda value: isinstance(value, bool),
    "apc_exact_sessions": _is_int_in(0, 64),
    "apc_session_checkpoints": _is_int_in(1, 64),
    "apc_max_growing_sessions": _is_int_in(0, 64),
    "apc_disk_path": lambda value: value is None or isinstance(value, str),
    "ngram_max": _is_int_in(1, 1024),
    "startup_timeout_s": _is_positive_number,
    "request_timeout_s": _is_positive_number,
    "startup_probe_interval_s": _is_positive_number,
    "probe_interval_s": _is_positive_number,
    "probe_timeout_s": _is_positive_number,
    "probe_failures_before_restart": _is_positive_int,
    "max_start_failures": _is_positive_int,
    "startup_retry_cooldown_s": _is_positive_number,
    "restart_delay_s": _is_positive_number,
    "shutdown_timeout_s": _is_positive_number,
    "idle_check_interval_s": _is_positive_number,
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
