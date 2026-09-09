"""The ``junie-mlx-vlm`` entry point: the daemon, or the worker it spawns.

Both processes are reachable through one program because the frozen
bundle (build_server.sh) holds exactly one executable, which the
daemon re-invokes with the "worker" subcommand — see
:func:`mlx_vlm_gateway.supervisor.worker_command`. One dispatch path,
frozen or not.

    junie-mlx-vlm                    serve the public API (default)
    junie-mlx-vlm daemon             the same, said explicitly
    junie-mlx-vlm worker             run only the inference worker
    junie-mlx-vlm --junie-config JUNIE_HOME --model MODEL
                                     write the Junie model config

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
import json
import os
import sys

from mlx_vlm_gateway.supervisor import DAEMON_SUBCOMMAND, WORKER_SUBCOMMAND
from mlx_vlm_shared.server_settings import (
    CONFIG_PATH_ENV,
    DEFAULT_CONFIG_PATH,
    config_path,
    load_config,
)

PROG = "junie-mlx-vlm"

# Not a subcommand: serverctl.sh's entry point into the config generator.
JUNIE_CONFIG_FLAG = "--junie-config"


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


# Junie writes settings.json with four-space indentation; the model configs
# it reads are indented the same way. Both are rewritten in that shape, so a
# generated file is indistinguishable from a hand-edited one and a diff of the
# user's settings shows only the line that changed.
JUNIE_JSON_INDENT = 4


def resolve_templates(obj, engine_port: int, auth_token: str):
    """Substitute $ENGINE_PORT and $AUTH_TOKEN in every string of a tree.

    Containers are rebuilt rather than mutated, so the caller's template
    stays untouched and no copy of it is needed.
    """
    if isinstance(obj, str):
        return obj.replace("$ENGINE_PORT", str(engine_port)).replace(
            "$AUTH_TOKEN", auth_token
        )
    if isinstance(obj, list):
        return [resolve_templates(item, engine_port, auth_token) for item in obj]
    if isinstance(obj, dict):
        return {
            key: resolve_templates(value, engine_port, auth_token)
            for key, value in obj.items()
        }
    return obj


def write_json(path: str, payload) -> None:
    """Write JSON in Junie's own shape, and atomically.

    The temporary file lives in the destination directory so os.replace is a
    rename within one filesystem: readers see either the old file or the new
    one, never a half-written settings.json.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=JUNIE_JSON_INDENT)
        stream.write("\n")
    os.replace(temp_path, path)


def run_junie_config(argv: list[str]) -> None:
    """Generate the Junie model config from the installed model template.

    install.sh downloads the model's descriptor to
    ~/.local/share/junie-local/models/<model>.json and leaves the rest to the
    engine: this reads that file, resolves the template variables
    ($ENGINE_PORT, $AUTH_TOKEN) from server-config.json, writes the finished
    junieConfig to <junie_home>/models/<id>.json under the template's own id,
    and points modelForLaunch at it.
    """
    parser = argparse.ArgumentParser(
        prog=f"{PROG} --junie-config",
        description="Generate the Junie model config file and set the default model.",
    )
    parser.add_argument(
        "junie_home",
        help="Path to the Junie configuration directory (e.g. ~/.junie).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model name (e.g. Qwen3.6-27B-MLX-4bit). "
            "Matches the filename in the install directory's models/."
        ),
    )
    args = parser.parse_args(argv)

    # Resolve paths.
    server_config_path = config_path()
    models_dir = os.path.join(os.path.dirname(server_config_path), "models")
    model_config_path = os.path.join(models_dir, f"{args.model}.json")
    junie_home = os.path.expanduser(args.junie_home)
    junie_models_dir = os.path.join(junie_home, "models")
    junie_settings_path = os.path.join(junie_home, "settings.json")

    # Read the model config template.
    if not os.path.isfile(model_config_path):
        print(
            f"ERROR: model config not found at {model_config_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(model_config_path, encoding="utf-8") as f:
        model_cfg = json.load(f)

    junie_cfg = model_cfg.get("junieConfig")
    if junie_cfg is None:
        print(
            f"ERROR: no junieConfig field in {model_config_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Read server config to resolve template variables. load_config merges the
    # defaults, so every key it is asked for is present.
    server_cfg = load_config(server_config_path)
    engine_port = server_cfg["port"]
    # An install always has a token; a checkout that never ran install.sh has
    # an open API and no api_key, and the generated config then carries an
    # empty bearer token — which the open API accepts. Say so, since against
    # an authenticated engine it would surface much later as a 401.
    auth_token = server_cfg["api_key"] or ""
    if not auth_token:
        print(
            f"WARNING: no api_key in {server_config_path}; the generated config"
            " carries an empty token.",
            file=sys.stderr,
        )

    resolved = resolve_templates(junie_cfg, engine_port, auth_token)

    # Write the Junie model config under the id the template names, the same
    # id `serverctl.sh uninstall` looks up there to clean up.
    junie_model_id = model_cfg.get("id", "local-" + args.model.lower())
    junie_config_path = os.path.join(junie_models_dir, f"{junie_model_id}.json")
    write_json(junie_config_path, resolved)
    print(f"Junie model config created at {junie_config_path}")

    # Set the default model in Junie settings. The file is the user's, so it is
    # read, one key is changed, and it is written back in Junie's own shape.
    custom_id = f"custom:{junie_model_id}"
    if os.path.isfile(junie_settings_path):
        with open(junie_settings_path, encoding="utf-8") as stream:
            settings = json.load(stream)
        settings["modelForLaunch"] = custom_id
        write_json(junie_settings_path, settings)
        print(f"Default model set to {junie_model_id} in {junie_settings_path}")
    else:
        print(
            f"WARNING: Junie settings not found at {junie_settings_path}",
            file=sys.stderr,
        )
        print(
            "The model config was created, but the default model was not set.",
            file=sys.stderr,
        )
        print(
            "Start Junie once so it creates settings.json, then re-run this command.",
            file=sys.stderr,
        )

    print("Restart Junie to apply the changes.")
    print("Control the engine with: ./serverctl.sh {start|stop|status|wait}")


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # What a first argument dispatches to. --junie-config sits here rather
    # than among the parser's choices because it is serverctl.sh's entry
    # point into the generator, not something run by hand; being a flag it
    # would never survive the subcommand parser anyway.
    commands = {
        DAEMON_SUBCOMMAND: run_daemon,
        WORKER_SUBCOMMAND: run_worker,
        JUNIE_CONFIG_FLAG: run_junie_config,
    }
    # A known first argument takes everything after it untouched, so
    # "worker --help" reaches the worker's parser rather than this one.
    command = commands.get(argv[0]) if argv else None
    if command is not None:
        command(argv[1:])
        return
    # Anything else is ours to answer: --help, an unknown command, or the
    # bare invocation that means the daemon.
    build_parser().parse_args(argv)
    run_daemon(argv)


if __name__ == "__main__":
    main()
