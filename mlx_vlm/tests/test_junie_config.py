import asyncio
import json
import os

from mlx_vlm.server import _app_module as server_app
from mlx_vlm.server.junie import launch
from mlx_vlm_shared.server_settings import (
    CONFIG_PATH_ENV,
    DEFAULT_CONFIG,
    load_config,
    normalize_config,
)


def test_config_file_drives_model_preload(monkeypatch, tmp_path):
    path = tmp_path / "server-config.json"
    path.write_text(
        json.dumps(
            {
                "model_name": "test-model",
                "draft_model": None,
                "kv_quantization": False,
            }
        )
    )
    monkeypatch.setenv(CONFIG_PATH_ENV, str(path))
    monkeypatch.delenv("MLX_VLM_PRELOAD_MODEL", raising=False)

    # The launcher (python -m mlx_vlm.server.junie) exports the runtime
    # settings to the env before the server starts; the stock lifespan
    # preload picks them up from there.
    launch.initialize_from_config(load_config())
    assert os.environ["MLX_VLM_PRELOAD_MODEL"] == "test-model"

    calls = []

    def fake_get_cached_model(model_path, adapter_path=None, *, model_kind="auto"):
        calls.append((model_path, adapter_path, model_kind))
        return object(), object(), object()

    monkeypatch.setattr(server_app, "get_cached_model", fake_get_cached_model)
    monkeypatch.setattr(server_app, "_start_seed_prefix_warmup", lambda: None)

    async def run_lifespan():
        async with server_app.lifespan(server_app.app):
            pass

    asyncio.run(run_lifespan())

    assert calls == [("test-model", None, "text_generation")]


def test_launcher_builds_worker_settings(monkeypatch, tmp_path):
    cfg = {
        **DEFAULT_CONFIG,
        "host": "0.0.0.0",
        # The worker serves "worker_port"; "port" belongs to the daemon.
        "port": 12345,
        "worker_port": 12346,
        "prefill_step_size": 2048,
        "seed_request": "",
        "int8_prefill": False,
        "preserve_thinking": False,
        "log_raw_tokens": False,
        "apc_enabled": True,
        "apc_disk_path": str(tmp_path / "cache"),
        "ngram_max": 6,
        # Not a supported setting: the launcher must ignore it and keep
        # request processing serialized.
        "max_concurrent_requests": 4,
    }

    assert launch.build_argv(cfg) == [
        "--host",
        "0.0.0.0",
        "--port",
        "12346",
        "--prefill-step-size",
        "2048",
        "--quantized-kv-start",
        "0",
    ]

    launch.apply_inference_env(cfg)
    assert os.environ["APC_ENABLED"] == "1"
    assert os.environ["APC_DISK_PATH"] == str(tmp_path / "cache")
    assert os.environ["MLX_VLM_NGRAM_MAX"] == "6"
    assert os.environ["MLX_VLM_MAX_CONCURRENT_REQUESTS"] == "1"


def test_pre_m5_mac_uses_standard_prefill(monkeypatch):
    monkeypatch.setattr(launch, "_apple_chip_generation", lambda: 4)

    assert "--int8-prefill" not in launch.build_argv(DEFAULT_CONFIG)


def test_m5_mac_uses_int8_prefill(monkeypatch):
    monkeypatch.setattr(launch, "_apple_chip_generation", lambda: 5)

    assert "--int8-prefill" in launch.build_argv(DEFAULT_CONFIG)


def test_gateway_owns_auto_unload(monkeypatch):
    monkeypatch.setenv("MLX_VLM_AUTO_UNLOAD_TIME", "60")

    launch.apply_config_to_env(DEFAULT_CONFIG)

    assert "MLX_VLM_AUTO_UNLOAD_TIME" not in os.environ


def test_soft_request_timeout_comes_from_the_config(monkeypatch):
    monkeypatch.setenv("MLX_VLM_SOFT_REQUEST_TIMEOUT", "9999")

    launch.apply_config_to_env({**DEFAULT_CONFIG, "soft_request_timeout": 42})
    assert os.environ["MLX_VLM_SOFT_REQUEST_TIMEOUT"] == "42"

    # null disables the soft stop, leaving only the daemon's hard limit.
    launch.apply_config_to_env({**DEFAULT_CONFIG, "soft_request_timeout": None})
    assert "MLX_VLM_SOFT_REQUEST_TIMEOUT" not in os.environ


def test_missing_worker_config_uses_defaults_without_writing(monkeypatch, tmp_path):
    path = tmp_path / "server-config.json"
    monkeypatch.setenv(CONFIG_PATH_ENV, str(path))

    loaded = load_config()

    assert loaded == DEFAULT_CONFIG
    assert not path.exists()


def test_invalid_worker_config_is_normalized_without_writing(
    monkeypatch, tmp_path
):
    path = tmp_path / "server-config.json"
    original = {
        "model_name": "test-model",
        "auto_unload_time": "hello",
        "kv_quantization": "yes",
    }
    path.write_text(json.dumps(original))
    monkeypatch.setenv(CONFIG_PATH_ENV, str(path))

    loaded = load_config()

    assert loaded["model_name"] == "test-model"
    assert loaded["auto_unload_time"] == DEFAULT_CONFIG["auto_unload_time"]
    assert loaded["kv_quantization"] is DEFAULT_CONFIG["kv_quantization"]
    assert json.loads(path.read_text()) == original
    assert [item.name for item in tmp_path.iterdir()] == ["server-config.json"]


def test_shared_validation_preserves_previously_allowed_null_model():
    loaded, invalid = normalize_config({"model_name": None})

    assert loaded["model_name"] is None
    assert invalid == {}
