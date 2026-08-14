import json
import os

import pytest

import mlx_vlm_gateway.settings as settings_module
from mlx_vlm_gateway.app import apply_model_cache_env, build_settings, rotate_worker_log
from mlx_vlm_gateway.settings import (
    DEFAULT_PUBLIC_SETTINGS,
    SettingsStore,
    SettingsValidationError,
)
from mlx_vlm_gateway.supervisor import worker_command
from mlx_vlm_shared.server_settings import (
    CONFIG_PATH_ENV,
    DEFAULT_CONFIG,
    DEFAULT_CONFIG_PATH,
    config_path,
)


def test_config_path_defaults_to_the_shared_location(monkeypatch):
    monkeypatch.delenv(CONFIG_PATH_ENV, raising=False)
    assert config_path() == os.path.expanduser(DEFAULT_CONFIG_PATH)

    monkeypatch.setenv(CONFIG_PATH_ENV, "~/elsewhere.json")
    assert config_path() == os.path.expanduser("~/elsewhere.json")


def test_daemon_settings_come_from_the_config(tmp_path):
    path = str(tmp_path / "server-config.json")

    settings = build_settings(path, {**DEFAULT_CONFIG, "host": "0.0.0.0"})

    # A worker bound to 0.0.0.0 is reached over the loopback address.
    assert settings.worker_url == f"http://127.0.0.1:{DEFAULT_CONFIG['worker_port']}"
    assert settings.worker_command == worker_command()
    assert settings.config_path == path


def test_worker_log_sits_beside_the_config(tmp_path):
    path = str(tmp_path / "server-config.json")

    settings = build_settings(path, DEFAULT_CONFIG)

    assert settings.worker_log_path == str(tmp_path / "junie-mlx-vlm.log")


def test_rotation_keeps_one_previous_run(tmp_path):
    log = tmp_path / "junie-mlx-vlm.log"

    # Nothing yet: rotating is a no-op that still makes the directory usable.
    rotate_worker_log(str(log))
    assert not log.exists() and not (tmp_path / "junie-mlx-vlm.log.0").exists()

    log.write_text("first run\n")
    rotate_worker_log(str(log))
    assert (tmp_path / "junie-mlx-vlm.log.0").read_text() == "first run\n"
    assert not log.exists()

    log.write_text("second run\n")
    rotate_worker_log(str(log))

    # Only one generation is kept, so the first run is gone.
    assert (tmp_path / "junie-mlx-vlm.log.0").read_text() == "second run\n"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["junie-mlx-vlm.log.0"]


def test_rotation_creates_a_missing_directory(tmp_path):
    log = tmp_path / "fresh-machine" / "junie-mlx-vlm.log"

    rotate_worker_log(str(log))

    assert log.parent.is_dir()


def test_daemon_exports_the_configured_models_dir_to_the_worker(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", "/some/other/cache")
    # setenv-then-delenv so monkeypatch restores whatever was there.
    monkeypatch.setenv("HF_HUB_OFFLINE", "restored-on-teardown")
    monkeypatch.delenv("HF_HUB_OFFLINE")

    apply_model_cache_env({**DEFAULT_CONFIG, "models_dir": str(tmp_path)})

    # The worker inherits this environment when the daemon spawns it.
    assert os.environ["HF_HUB_CACHE"] == str(tmp_path)
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_daemon_keeps_an_explicit_online_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    apply_model_cache_env({**DEFAULT_CONFIG, "models_dir": str(tmp_path)})

    assert os.environ["HF_HUB_OFFLINE"] == "0"


def test_daemon_refuses_a_worker_port_that_collides_with_the_public_one(tmp_path):
    config = {**DEFAULT_CONFIG, "port": 19239, "worker_port": 19239}

    with pytest.raises(SystemExit, match="must differ"):
        build_settings(str(tmp_path / "server-config.json"), config)


def test_gateway_creates_missing_config_with_defaults(tmp_path):
    path = tmp_path / "server-config.json"

    store = SettingsStore(str(path))

    assert store.current() == DEFAULT_PUBLIC_SETTINGS
    assert json.loads(path.read_text()) == settings_module.DEFAULT_CONFIG


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
    assert [item.name for item in tmp_path.iterdir()] == ["server-config.json"]


def test_settings_store_keeps_old_config_when_atomic_replace_fails(
    monkeypatch, tmp_path
):
    path = tmp_path / "server-config.json"
    original = {"model_name": "old-model", "kv_quantization": False}
    path.write_text(json.dumps(original))
    store = SettingsStore(str(path))
    persisted_before_save = json.loads(path.read_text())

    def fail_replace(_source, _target):
        raise OSError("disk full")

    monkeypatch.setattr(settings_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="disk full"):
        store.save({"kv_quantization": True})

    assert json.loads(path.read_text()) == persisted_before_save
    assert [item.name for item in tmp_path.iterdir()] == ["server-config.json"]


def test_settings_are_validated_once_and_then_read_from_memory(tmp_path):
    path = tmp_path / "server-config.json"
    path.write_text(json.dumps({"model_name": "demo", "auto_unload_time": "hello"}))

    store = SettingsStore(str(path))
    path.write_text(json.dumps({"auto_unload_time": 123}))

    assert store.current()["auto_unload_time"] == 600
    assert store.current()["model_name"] == "demo"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ([], "Request body must be a JSON object."),
        ({}, "No settings provided."),
        ({"unknown": 1}, "Unknown settings: ['unknown']"),
        ({"model_name": ""}, '"model_name" must be a non-empty string.'),
        (
            {"model_name": "other-model"},
            "Model switching is not supported. Available model: "
            "Qwen3.8-27B-MLX-4bit",
        ),
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
            "model_name": "  Qwen3.8-27B-MLX-4bit  ",
            "max_context_length": None,
            "auto_unload_time": 600,
            "kv_quantization": False,
            "force": True,
        }
    )

    assert updates == {
        "model_name": "Qwen3.8-27B-MLX-4bit",
        "max_context_length": None,
        "kv_quantization": False,
        "auto_unload_time": 600,
    }
    assert force is True
