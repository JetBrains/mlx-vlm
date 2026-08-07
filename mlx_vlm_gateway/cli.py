"""The ``junie-mlx-vlm`` entry point: the daemon, or the worker it spawns.

Both processes are reachable through one program because the frozen
bundle (build_cli_tarball.sh) holds exactly one executable, which the
daemon re-invokes with the "worker" subcommand — see
:func:`mlx_vlm_gateway.supervisor.worker_command`. One dispatch path,
frozen or not.

    junie-mlx-vlm           serve the public API (default)
    junie-mlx-vlm daemon    the same, said explicitly
    junie-mlx-vlm worker    run only the inference worker

Neither subcommand takes options of its own: every setting comes from
the config file (see :mod:`mlx_vlm_shared.server_settings`).

Imports here are absolute because PyInstaller runs this file in script
mode, where ``__package__`` is empty and a relative import raises. The
two entry functions are imported only when chosen, so starting the
daemon does not pay for importing ``mlx_vlm``.
"""

import sys

from mlx_vlm_gateway.supervisor import DAEMON_SUBCOMMAND, WORKER_SUBCOMMAND


def run_daemon(argv: list[str]) -> None:
    from mlx_vlm_gateway.app import main as daemon_main

    daemon_main(argv)


def run_worker(argv: list[str]) -> None:
    # The worker parses sys.argv itself and then rewrites it for the stock
    # server, so hand it a clean one without the subcommand.
    sys.argv = [sys.argv[0], *argv]

    from mlx_vlm.server.junie.launch import main as worker_main

    worker_main()


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == [WORKER_SUBCOMMAND]:
        run_worker(argv[1:])
        return
    if argv[:1] == [DAEMON_SUBCOMMAND]:
        argv = argv[1:]
    run_daemon(argv)


if __name__ == "__main__":
    main()
