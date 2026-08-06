import asyncio
import json
import os

from mlx_vlm.server import _app_module as server_app
from mlx_vlm.server.junie import config, launch


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
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(path))
    monkeypatch.delenv("MLX_VLM_PRELOAD_MODEL", raising=False)
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
        **config.DEFAULT_CONFIG,
        "host": "0.0.0.0",
        "port": 12345,
        "prefill_step_size": 2048,
        "seed_request": "",
        "int8_prefill": False,
        "preserve_thinking": False,
        "log_raw_tokens": False,
        "apc_enabled": True,
        "apc_disk_path": str(tmp_path / "cache"),
        "ngram_max": 6,
        "max_concurrent_requests": 1,
    }

    assert launch.build_argv(cfg) == [
        "--host",
        "0.0.0.0",
        "--port",
        "12345",
        "--prefill-step-size",
        "2048",
    ]

    launch.apply_inference_env(cfg)
    assert os.environ["APC_ENABLED"] == "1"
    assert os.environ["APC_DISK_PATH"] == str(tmp_path / "cache")
    assert os.environ["MLX_VLM_NGRAM_MAX"] == "6"
    assert os.environ["MLX_VLM_MAX_CONCURRENT_REQUESTS"] == "1"


def test_gateway_owns_auto_unload(monkeypatch):
    monkeypatch.setenv("MLX_VLM_AUTO_UNLOAD_TIME", "60")

    config.apply_config_to_env(config.DEFAULT_CONFIG)

    assert "MLX_VLM_AUTO_UNLOAD_TIME" not in os.environ


def test_missing_config_is_created_with_stas_defaults(monkeypatch, tmp_path):
    path = tmp_path / "server-config.json"
    monkeypatch.setenv(config.CONFIG_PATH_ENV, str(path))

    loaded = config.load_config()

    assert loaded == config.DEFAULT_CONFIG
    assert json.loads(path.read_text()) == config.DEFAULT_CONFIG
