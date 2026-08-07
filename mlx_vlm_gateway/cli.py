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
the config file (see :mod:`mlx_vlm_shared.server_settings`). ``--help``
after a subcommand belongs to that subcommand; on its own it is answered
here.

Imports here are absolute because PyInstaller runs this file in script
mode, where ``__package__`` is empty and a relative import raises. The
two entry functions are imported only when chosen, so starting the
daemon does not pay for importing ``mlx_vlm`` — and neither does
``--help``.
"""

import argparse
import sys

from mlx_vlm_gateway.supervisor import DAEMON_SUBCOMMAND, WORKER_SUBCOMMAND
from mlx_vlm_shared.server_settings import CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH

PROG = "junie-mlx-vlm"


def build_parser() -> argparse.ArgumentParser:
    """The dispatcher's own parser: it answers --help and nothing else."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Raw formatting, so the lines below are wrapped here rather than
        # rewrapped to the terminal width.
        description=(
            "The Junie local server. With no arguments it runs the daemon:\n"
            "the public OpenAI-compatible API, which starts and supervises\n"
            "the inference worker itself."
        ),
        epilog=(
            "Neither subcommand takes options of its own — every setting\n"
            f"comes from the config file (${CONFIG_PATH_ENV}), which\n"
            f"defaults to {DEFAULT_CONFIG_PATH}.\n"
            "Pass --help after a subcommand for that subcommand's help."
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        default=DAEMON_SUBCOMMAND,
        choices=(DAEMON_SUBCOMMAND, WORKER_SUBCOMMAND),
        help=(
            f"{DAEMON_SUBCOMMAND} (default): serve the public API and "
            f"supervise the worker. {WORKER_SUBCOMMAND}: run only the "
            "inference worker, which the daemon normally spawns for you."
        ),
    )
    return parser


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
    # A subcommand is dispatched with everything after it untouched, so
    # "worker --help" reaches the worker's parser rather than this one.
    if argv[:1] == [WORKER_SUBCOMMAND]:
        run_worker(argv[1:])
        return
    if argv[:1] == [DAEMON_SUBCOMMAND]:
        run_daemon(argv[1:])
        return
    # Anything else is ours to answer: --help, an unknown command, or the
    # bare invocation that means the daemon.
    build_parser().parse_args(argv)
    run_daemon(argv)


if __name__ == "__main__":
    main()
