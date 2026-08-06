import json

import pytest

from mlx_vlm_gateway.settings import (
    DEFAULT_PUBLIC_SETTINGS,
    SettingsStore,
    SettingsValidationError,
)


def test_settings_store_reads_and_preserves_worker_config(tmp_path):
    path = tmp_path / "server-config.json"
    path.write_text(
        json.dumps(
            {
                "model_name": "old-model",
                "draft_model": "draft-model",
                "prefill_step_size": 1024,
            }
        )
    )
    store = SettingsStore(str(path))

    settings = store.save({"model_name": "new-model", "kv_quantization": True})

    assert settings == {
        **DEFAULT_PUBLIC_SETTINGS,
        "model_name": "new-model",
        "kv_quantization": True,
    }
    persisted = json.loads(path.read_text())
    assert persisted["draft_model"] == "draft-model"
    assert persisted["prefill_step_size"] == 1024


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ([], "Request body must be a JSON object."),
        ({}, "No settings provided."),
        ({"unknown": 1}, "Unknown settings: ['unknown']"),
        ({"model_name": ""}, '"model_name" must be a non-empty string.'),
        (
            {"max_context_length": 0},
            '"max_context_length" must be a positive integer or null.',
        ),
        (
            {"auto_unload_time": True},
            '"auto_unload_time" must be a positive integer or null.',
        ),
        (
            {"kv_quantization": "yes"},
            '"kv_quantization" must be a boolean.',
        ),
        ({"model_name": "demo", "force": 1}, '"force" must be a boolean.'),
    ],
)
def test_settings_validation_errors(body, message):
    with pytest.raises(SettingsValidationError) as error:
        SettingsStore(None).validate(body)
    assert str(error.value) == message


def test_settings_validation_returns_updates_and_force():
    updates, force = SettingsStore(None).validate(
        {
            "model_name": "  demo  ",
            "max_context_length": None,
            "auto_unload_time": 600,
            "kv_quantization": False,
            "force": True,
        }
    )

    assert updates == {
        "model_name": "demo",
        "max_context_length": None,
        "kv_quantization": False,
        "auto_unload_time": 600,
    }
    assert force is True
