import json
import os
import sys
import time

import pytest

from mlx_vlm_gateway.settings import SettingsStore, SettingsValidationError
from mlx_vlm_gateway.splash import command, supervise
from mlx_vlm_shared.server_settings import DEFAULT_CONFIG


@pytest.fixture
def config(tmp_path):
    source = tmp_path / "source"
    package = tmp_path / "package"
    for file in ("server/server.py", "build/splash"):
        p = source / file
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    for name in ("target", "draft", "vision", "tokenizer"):
        (package / name).mkdir(parents=True)
    (package / "manifest.json").write_text(json.dumps({"model": "local/blend-v03"}))
    return {
        **DEFAULT_CONFIG,
        "worker_backend": "splash",
        "splash_python": sys.executable,
        "splash_source": str(source),
        "splash_package": str(package),
        "kv_quantization": True,
        "model_name": "junie-blend",
        "draft_model": "v03",
        "api_key": "test-secret",
        "worker_port": 19540,
    }


def test_command_keeps_secret_out_of_argv_and_preserves_context(config):
    config["max_context_length"] = 65536
    args = command(config)
    assert "test-secret" not in " ".join(args)
    assert args[args.index("--model") + 1] == "local/blend-v03"
    assert args[args.index("--max-context") + 1] == "65536"
    assert args[args.index("--port") + 1] == "19540"
    assert "--max-memory" not in args


def test_missing_package_and_unsupported_kv_fail_before_launch(config):
    config["kv_quantization"] = False
    with pytest.raises(ValueError, match="INT8"):
        command(config)
    config["kv_quantization"] = True
    config["splash_package"] += "-absent"
    with pytest.raises(ValueError, match="Missing"):
        command(config)


def test_identity_mapping_preserves_tools_reasoning_and_sampling(config, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    store = SettingsStore(str(path))
    payload = {
        "model": "junie-blend",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "reasoning_effort": "low",
        "messages": [
            {
                "role": "assistant",
                "reasoning_content": "reasoning",
                "tool_calls": [{"id": "call1"}],
            }
        ],
    }
    mapped = store.inference_payload(payload)
    assert mapped == {**payload, "model": "local/blend-v03"}
    assert payload["model"] == "junie-blend"
    assert list(store.supported_models()) == ["junie-blend"]
    with pytest.raises(SettingsValidationError, match="INT8"):
        store.validate({"kv_quantization": False})
    with pytest.raises(SettingsValidationError, match="Unsupported model"):
        store.validate({"model_name": "another-model"})
    updates, _ = store.validate({"auto_unload_time": 30})
    store.save(updates)
    assert store.current()["auto_unload_time"] == 30
    assert store.draft_model() == "v03"


def test_supervisor_reaps_worker_when_gateway_parent_is_gone(tmp_path):
    marker = tmp_path / "started"
    script = "import pathlib,time; pathlib.Path(%r).touch(); time.sleep(60)" % str(
        marker
    )
    start = time.monotonic()
    assert supervise([sys.executable, "-c", script], dict(os.environ), -1) == 0
    assert time.monotonic() - start < 8


def test_supervisor_propagates_child_failure():
    assert (
        supervise(
            [sys.executable, "-c", "raise SystemExit(7)"],
            dict(os.environ),
            os.getppid(),
        )
        == 7
    )
