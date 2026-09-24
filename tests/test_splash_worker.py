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
    assert (path.stat().st_mode & 0o777) == 0o600
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


def test_bundled_runtime_needs_no_checkout_or_external_python(
    config, tmp_path, monkeypatch
):
    runtime = tmp_path / "relocated install" / "splash"
    for file in ("python/bin/python3", "server/server.py", "engine/splash"):
        path = runtime / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setenv("JUNIE_SPLASH_RUNTIME", str(runtime))
    del config["splash_python"]
    del config["splash_source"]
    args = command(config)
    assert args[0] == str(runtime / "python/bin/python3")
    assert args[args.index("--binary") + 1] == str(runtime / "engine/splash")


def test_installer_selects_splash_and_creates_fresh_junie_default(
    config, tmp_path, monkeypatch
):
    from mlx_vlm_gateway import cli

    root = tmp_path / "install"
    models = root / "models"
    models.mkdir(parents=True)
    package = models / "blend-package"
    package.symlink_to(config["splash_package"], target_is_directory=True)
    descriptor = {
        "id": "local-blend",
        "worker_backend": "splash",
        "splash_package": "blend-package",
        "draft_model": "v03",
        "junieConfig": {"id": "blend", "apiKey": "$AUTH_TOKEN"},
    }
    (models / "blend.json").write_text(json.dumps(descriptor))
    path = root / "server-config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("JUNIE_SERVER_CONFIG", str(path))
    junie = tmp_path / "junie"
    cli.run_junie_config([str(junie), "--model", "blend"])
    saved = json.loads(path.read_text())
    assert saved["model_name"] == "blend"
    assert saved["splash_package"] == str(package)
    assert saved["api_key"] == config["api_key"]
    assert (path.stat().st_mode & 0o777) == 0o600
    assert (
        json.loads((junie / "settings.json").read_text())["modelForLaunch"]
        == "custom:local-blend"
    )
    descriptor["splash_package"] = "../outside"
    (models / "blend.json").write_text(json.dumps(descriptor))
    with pytest.raises(SystemExit):
        cli.run_junie_config([str(junie), "--model", "blend"])


def test_explicit_memory_limit_is_forwarded_in_bytes(config):
    config['splash_max_memory_bytes'] = 80 * 1024**3
    args = command(config)
    assert args[args.index('--max-memory') + 1] == str(80 * 1024**3)
    for invalid in [0, -1, True, '80G']:
        config['splash_max_memory_bytes'] = invalid
        with pytest.raises(ValueError, match='splash_max_memory_bytes'):
            command(config)
