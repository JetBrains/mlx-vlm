import json
import sys
import types

import pytest

import mlx_vlm_gateway.cli as cli
import mlx_vlm_gateway.supervisor as supervisor


def _record(calls, name):
    def run(argv):
        calls.append((name, argv))

    return run


def test_bare_invocation_starts_the_daemon(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "run_daemon", _record(calls, "daemon"))
    monkeypatch.setattr(cli, "run_worker", _record(calls, "worker"))

    cli.main([])

    assert calls == [("daemon", [])]


def test_subcommands_dispatch_and_are_stripped(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "run_daemon", _record(calls, "daemon"))
    monkeypatch.setattr(cli, "run_worker", _record(calls, "worker"))

    # --help after a subcommand belongs to that subcommand, not to us.
    cli.main(["daemon", "--help"])
    cli.main(["worker", "--help"])

    assert calls == [("daemon", ["--help"]), ("worker", ["--help"])]


def test_bare_help_is_answered_by_the_dispatcher(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "run_daemon", _record(calls, "daemon"))
    monkeypatch.setattr(cli, "run_worker", _record(calls, "worker"))

    with pytest.raises(SystemExit) as exit_code:
        cli.main(["--help"])

    assert exit_code.value.code == 0
    help_text = capsys.readouterr().out
    assert "junie-mlx-vlm" in help_text
    assert "daemon" in help_text and "worker" in help_text
    # Answered here, so neither subcommand's stack gets imported for it.
    assert calls == []


def test_an_unknown_command_is_rejected(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_daemon", _record([], "daemon"))

    for argv in (["nonsense"], ["--nonsense"]):
        with pytest.raises(SystemExit) as exit_code:
            cli.main(argv)
        assert exit_code.value.code == 2
    assert "junie-mlx-vlm" in capsys.readouterr().err


def test_worker_runs_with_the_subcommand_removed_from_argv(monkeypatch):
    seen = {}
    stub = types.ModuleType("mlx_vlm.server.junie.launch")
    stub.main = lambda: seen.setdefault("argv", list(sys.argv))
    monkeypatch.setitem(sys.modules, "mlx_vlm.server.junie.launch", stub)
    monkeypatch.setattr(sys, "argv", ["junie-mlx-vlm", "worker"])

    cli.run_worker([])

    assert seen["argv"] == ["junie-mlx-vlm"]


def test_worker_command_from_a_checkout(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)

    assert supervisor.worker_command() == (
        sys.executable,
        "-m",
        "mlx_vlm_gateway.cli",
        "worker",
    )


def test_worker_command_in_a_frozen_bundle(monkeypatch):
    # PyInstaller builds one executable, and sys.executable is it.
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/opt/junie-mlx-vlm/junie-mlx-vlm")

    assert supervisor.worker_command() == ("/opt/junie-mlx-vlm/junie-mlx-vlm", "worker")


def test_the_hardcoded_dispatcher_path_matches_the_real_module():
    assert supervisor.CLI_MODULE == cli.__name__


# --- the Junie config generator ------------------------------------------


MODEL = "Qwen3.6-27B-MLX-4bit"

# What install.sh saves to <install dir>/models/<model>.json: the Junie id the
# config is written under, and the junieConfig template to resolve.
TEMPLATE = {
    "id": "local-qwen3.6-27b-4bit",
    "junieConfig": {
        "displayName": "Qwen 3.6",
        "id": MODEL,
        "baseUrl": "http://localhost:$ENGINE_PORT/v1/chat/completions",
        "apiKey": "$AUTH_TOKEN",
        "extraBody": {"enable_thinking": False},
        "stopStrings": ["$AUTH_TOKEN-never"],
    },
    "archives": [{"modelId": MODEL}],
}


@pytest.fixture
def install(tmp_path, monkeypatch):
    """An install directory and a Junie home, as install.sh leaves them."""
    install_dir = tmp_path / "junie-local"
    (install_dir / "models").mkdir(parents=True)
    (install_dir / "models" / f"{MODEL}.json").write_text(json.dumps(TEMPLATE))
    (install_dir / "server-config.json").write_text(
        json.dumps({"api_key": "sk-secret", "port": 12345})
    )
    monkeypatch.setenv("JUNIE_SERVER_CONFIG", str(install_dir / "server-config.json"))

    junie_home = tmp_path / "junie-home"
    junie_home.mkdir()
    return install_dir, junie_home


def test_junie_config_resolves_the_template_and_sets_the_default(install, capsys):
    install_dir, junie_home = install
    # Junie's own formatting, which the rewrite has to preserve.
    settings = junie_home / "settings.json"
    settings.write_text('{\n    "sessionCount": "48"\n}\n')

    cli.main(["--junie-config", str(junie_home), "--model", MODEL])

    generated = junie_home / "models" / "local-qwen3.6-27b-4bit.json"
    assert json.loads(generated.read_text()) == {
        "displayName": "Qwen 3.6",
        "id": MODEL,
        "baseUrl": "http://localhost:12345/v1/chat/completions",
        "apiKey": "sk-secret",
        "extraBody": {"enable_thinking": False},
        # Substitution reaches into lists, not only dicts.
        "stopStrings": ["sk-secret-never"],
    }
    # The user's settings keep their other keys, their indentation, and gain
    # only the default model.
    assert settings.read_text() == (
        '{\n    "sessionCount": "48",\n'
        '    "modelForLaunch": "custom:local-qwen3.6-27b-4bit"\n}\n'
    )
    # The template on disk is untouched, so a re-run resolves it again.
    template = install_dir / "models" / f"{MODEL}.json"
    assert json.loads(template.read_text()) == TEMPLATE
    assert "Restart Junie" in capsys.readouterr().out


def test_junie_config_warns_but_writes_when_no_token_is_configured(install, capsys):
    install_dir, junie_home = install
    # A checkout that never ran install.sh: an open API and no api_key.
    (install_dir / "server-config.json").write_text(json.dumps({"port": 12345}))

    cli.main(["--junie-config", str(junie_home), "--model", MODEL])

    generated = junie_home / "models" / "local-qwen3.6-27b-4bit.json"
    assert json.loads(generated.read_text())["apiKey"] == ""
    assert "no api_key" in capsys.readouterr().err


def test_junie_config_without_settings_still_writes_the_model_config(install, capsys):
    _, junie_home = install

    cli.main(["--junie-config", str(junie_home), "--model", MODEL])

    assert (junie_home / "models" / "local-qwen3.6-27b-4bit.json").is_file()
    assert not (junie_home / "settings.json").exists()
    assert "settings not found" in capsys.readouterr().err


def test_junie_config_rejects_an_uninstalled_model(install, capsys):
    _, junie_home = install

    with pytest.raises(SystemExit) as exit_code:
        cli.main(["--junie-config", str(junie_home), "--model", "Nonexistent"])

    assert exit_code.value.code == 1
    assert "model config not found" in capsys.readouterr().err


def test_junie_config_rejects_a_template_without_a_junie_config(install, capsys):
    install_dir, junie_home = install
    (install_dir / "models" / f"{MODEL}.json").write_text(json.dumps({"id": "x"}))

    with pytest.raises(SystemExit) as exit_code:
        cli.main(["--junie-config", str(junie_home), "--model", MODEL])

    assert exit_code.value.code == 1
    assert "no junieConfig" in capsys.readouterr().err
