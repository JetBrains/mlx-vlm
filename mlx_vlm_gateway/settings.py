import json
import os
import tempfile
from pathlib import Path
from typing import Optional


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
DEFAULT_PUBLIC_SETTINGS = {
    "model_name": "mlx-community/Qwen3.6-27B-4bit",
    "max_context_length": None,
    "kv_quantization": False,
    "auto_unload_time": None,
}
DEFAULT_DRAFT_MODEL = "mlx-community/Qwen3.6-27B-MTP-4bit"


class SettingsValidationError(ValueError):
    pass


class SettingsStore:
    """Lightweight access to the worker's persistent JSON configuration."""

    def __init__(self, path: Optional[str]):
        self.path = Path(path).expanduser() if path else None

    def _load_raw(self) -> dict:
        if self.path is None or not self.path.is_file():
            return {}
        try:
            value = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    def current(self) -> dict:
        raw = self._load_raw()
        return {
            key: raw.get(key, default)
            for key, default in DEFAULT_PUBLIC_SETTINGS.items()
        }

    def draft_model(self) -> Optional[str]:
        return self._load_raw().get("draft_model", DEFAULT_DRAFT_MODEL)

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

        for key in ("max_context_length", "auto_unload_time"):
            if key not in updates:
                continue
            value = updates[key]
            if value is not None and not (
                isinstance(value, int) and not isinstance(value, bool) and value > 0
            ):
                raise SettingsValidationError(
                    f'"{key}" must be a positive integer or null.'
                )

        if "kv_quantization" in updates and not isinstance(
            updates["kv_quantization"], bool
        ):
            raise SettingsValidationError('"kv_quantization" must be a boolean.')
        return updates, force

    def save(self, updates: dict) -> dict:
        current_file = self._load_raw()
        current_file.update(updates)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(
                dir=self.path.parent,
                prefix=".server-config-",
            )
            try:
                with os.fdopen(fd, "w") as stream:
                    json.dump(current_file, stream, indent=2)
                    stream.write("\n")
                os.replace(temporary, self.path)
            except BaseException:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise
        return self.current()
