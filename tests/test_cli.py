import sys
import types

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

    cli.main(["daemon", "--help"])
    cli.main(["worker"])
    # Anything else stays an argument, so the daemon reports it as unknown.
    cli.main(["--nonsense"])

    assert calls == [
        ("daemon", ["--help"]),
        ("worker", []),
        ("daemon", ["--nonsense"]),
    ]


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
