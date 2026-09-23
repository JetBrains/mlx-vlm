import json
import logging
import os
from pathlib import Path
from typing import Optional

from mlx_vlm_shared.server_settings import (
    DEFAULT_CONFIG,
    DEFAULT_DRAFT_MODEL,
    DEFAULT_PUBLIC_SETTINGS,
    PUBLIC_SETTING_KEYS,
    RESTART_SETTING_KEYS,
    discover_supported_models,
    is_valid_setting,
    normalize_config,
)

logger = logging.getLogger("mlx_vlm.gateway")

__all__ = [
    "DEFAULT_PUBLIC_SETTINGS",
    "RESTART_SETTING_KEYS",
    "SettingsStore",
    "SettingsValidationError",
]


class SettingsValidationError(ValueError):
    pass


class SettingsStore:
    """Validated in-memory view of the single persistent JSON config."""

    def __init__(self, path: Optional[str]):
        self.path = Path(path).expanduser() if path else None
        self._config = self._load_once()

    def _load_once(self) -> dict:
        if self.path is None:
            return dict(DEFAULT_CONFIG)

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("config root must be a JSON object")
        except FileNotFoundError:
            config = dict(DEFAULT_CONFIG)
            self._write(config)
            return config
        except (OSError, ValueError) as exc:
            logger.warning(
                "Cannot read config %s (%s); replacing it with defaults.",
                self.path,
                exc,
            )
            config = dict(DEFAULT_CONFIG)
            self._write(config)
            return config

        config, invalid = normalize_config(raw)
        for key, value in invalid.items():
            logger.warning(
                "Invalid config value %s=%r; using default %r.",
                key,
                value,
                config[key],
            )
        if config != raw:
            try:
                self._write(config)
            except OSError as exc:
                logger.warning("Failed to normalize config %s: %s", self.path, exc)
        return config

    def _write(self, config: dict) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(config, stream, indent=2)
                stream.write("\n")
            os.replace(temporary, self.path)
        except BaseException:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise

    def current(self) -> dict:
        return {key: self._config[key] for key in PUBLIC_SETTING_KEYS}

    def draft_model(self) -> Optional[str]:
        return self._config.get("draft_model", DEFAULT_DRAFT_MODEL)

    def is_splash(self) -> bool:
        return self._config.get("worker_backend", "mlx") == "splash"

    def inference_payload(self, payload: dict) -> dict:
        if not self.is_splash():
            return payload
        from mlx_vlm_gateway.splash import package_model

        return {**payload, "model": package_model(self._config)}

    def supported_models(self) -> dict:
        if self.is_splash():
            return {
                self._config["model_name"]: {
                    "draft_model": self.draft_model(),
                    "draft_kind": "dflash",
                }
            }
        return discover_supported_models(self.models_dir())

    def models_dir(self) -> str:
        return self._config.get("models_dir", DEFAULT_CONFIG["models_dir"])

    def validate(self, body) -> tuple[dict, bool]:
        if not isinstance(body, dict):
            raise SettingsValidationError("Request body must be a JSON object.")
        unknown = set(body) - set(PUBLIC_SETTING_KEYS) - {"force"}
        if unknown:
            raise SettingsValidationError(f"Unknown settings: {sorted(unknown)}")
        force = body.get("force", False)
        if not isinstance(force, bool):
            raise SettingsValidationError('"force" must be a boolean.')

        updates = {key: body[key] for key in PUBLIC_SETTING_KEYS if key in body}
        if not updates:
            raise SettingsValidationError("No settings provided.")

        if "model_name" in updates:
            value = updates["model_name"]
            if not isinstance(value, str) or not value.strip():
                raise SettingsValidationError(
                    '"model_name" must be a non-empty string.'
                )
            updates["model_name"] = value.strip()
            supported = self.supported_models()
            if updates["model_name"] not in supported:
                raise SettingsValidationError(
                    "Unsupported model. Available models: " f"{', '.join(supported)}"
                )

        for key in ("max_context_length", "auto_unload_time"):
            if key in updates and not is_valid_setting(key, updates[key]):
                raise SettingsValidationError(
                    f'"{key}" must be a positive integer or null.'
                )

        if "kv_quantization" in updates and not is_valid_setting(
            "kv_quantization", updates["kv_quantization"]
        ):
            raise SettingsValidationError('"kv_quantization" must be a boolean.')
        if self.is_splash() and updates.get("kv_quantization") is False:
            raise SettingsValidationError("This Splash runtime supports INT8 KV only.")
        return updates, force

    def save(self, updates: dict) -> dict:
        # A supported model always runs with its paired drafter and draft
        # kind; switching model without them would feed the verify pass a
        # mismatched MTP head, or run the wrong speculative-decode path
        # against the new drafter.
        supported = self.supported_models().get(updates.get("model_name"))
        if supported is not None:
            updates = dict(updates)
            updates.setdefault("draft_model", supported["draft_model"])
            if supported["draft_kind"] is not None:
                updates.setdefault("draft_kind", supported["draft_kind"])
        config, invalid = normalize_config({**self._config, **updates})
        if invalid:
            raise SettingsValidationError(f"Invalid settings: {sorted(invalid)}")
        self._write(config)
        self._config = config
        return self.current()
