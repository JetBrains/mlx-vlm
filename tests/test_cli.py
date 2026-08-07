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
