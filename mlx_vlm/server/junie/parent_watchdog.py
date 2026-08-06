"""Stop the inference worker when its gateway parent disappears."""

import logging
import os
import signal
import time
from threading import Thread, Timer


logger = logging.getLogger("mlx_vlm.server")

GATEWAY_PID_ENV = "MLX_VLM_GATEWAY_PID"
POLL_INTERVAL_S = 1.0
FORCE_EXIT_AFTER_S = 5.0


def _terminate_worker() -> None:
    logger.error("Gateway parent exited; stopping inference worker.")
    force_exit = Timer(FORCE_EXIT_AFTER_S, os._exit, args=(1,))
    force_exit.daemon = True
    force_exit.start()
    os.kill(os.getpid(), signal.SIGTERM)


def _watch_parent(expected_parent_pid: int) -> None:
    while os.getppid() == expected_parent_pid:
        time.sleep(POLL_INTERVAL_S)
    _terminate_worker()


def start_parent_watchdog() -> bool:
    """Start the watcher when this worker was launched by the gateway."""
    raw_pid = os.environ.get(GATEWAY_PID_ENV)
    if not raw_pid:
        return False
    try:
        parent_pid = int(raw_pid)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", GATEWAY_PID_ENV, raw_pid)
        return False

    Thread(
        target=_watch_parent,
        args=(parent_pid,),
        daemon=True,
        name="gateway-parent-watchdog",
    ).start()
    return True
