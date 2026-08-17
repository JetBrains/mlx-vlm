import asyncio
import logging
import os
import signal
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import httpx

from mlx_vlm_shared.server_settings import CONFIG_PATH_ENV, DEFAULT_CONFIG


logger = logging.getLogger("mlx_vlm.gateway")
GATEWAY_PID_ENV = "MLX_VLM_GATEWAY_PID"
# The worker already requires a bearer token on every endpoint when this is
# set (mlx_vlm.server.app), so the key from the config file only has to
# reach it as an env var, and be sent on every call the daemon makes.
WORKER_API_KEY_ENV = "MLX_VLM_SERVER_API_KEY"

DAEMON_SUBCOMMAND = "daemon"
WORKER_SUBCOMMAND = "worker"
# The dispatcher's dotted path. It cannot report its own importable name:
# both PyInstaller and "-m" run it as __main__. Kept here rather than in
# cli.py so that nothing imports the dispatcher, which would give the
# "-m" run two copies of it.
CLI_MODULE = "mlx_vlm_gateway.cli"


def auth_headers(api_key: Optional[str]) -> dict:
    """The header to send with a request, empty when no key is configured."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def worker_command() -> tuple[str, ...]:
    """The command line that starts the inference worker.

    Frozen (build_server.sh) there is a single executable and
    ``sys.executable`` is it, so the worker is this same program with the
    "worker" subcommand. From a checkout that subcommand is reached
    through ``-m`` instead, on a real interpreter.
    """
    if getattr(sys, "frozen", False):
        return (sys.executable, WORKER_SUBCOMMAND)
    return (sys.executable, "-m", CLI_MODULE, WORKER_SUBCOMMAND)


@dataclass(frozen=True)
class GatewaySettings:
    worker_url: str = "http://127.0.0.1:19240"
    worker_command: tuple[str, ...] = ()
    # File the worker's stdout and stderr go to; None leaves it inheriting
    # this process's, which is what a checkout wants.
    worker_log_path: Optional[str] = None
    # Defaults for all fields below come from DEFAULT_CONFIG, the single
    # source of truth for these numbers; build_settings() in app.py passes
    # the (possibly config-file-overridden) values through explicitly, so
    # these only take effect for callers that construct GatewaySettings
    # directly, e.g. tests.
    startup_timeout_s: float = DEFAULT_CONFIG["startup_timeout_s"]
    # Hard limit for a worker that cannot acknowledge cancellation; the
    # worker's own softer limit comes from "soft_request_timeout" in the
    # config file.
    request_timeout_s: float = DEFAULT_CONFIG["request_timeout_s"]
    startup_probe_interval_s: float = DEFAULT_CONFIG["startup_probe_interval_s"]
    probe_interval_s: float = DEFAULT_CONFIG["probe_interval_s"]
    probe_timeout_s: float = DEFAULT_CONFIG["probe_timeout_s"]
    probe_failures_before_restart: int = DEFAULT_CONFIG["probe_failures_before_restart"]
    max_start_failures: int = DEFAULT_CONFIG["max_start_failures"]
    startup_retry_cooldown_s: float = DEFAULT_CONFIG["startup_retry_cooldown_s"]
    restart_delay_s: float = DEFAULT_CONFIG["restart_delay_s"]
    shutdown_timeout_s: float = DEFAULT_CONFIG["shutdown_timeout_s"]
    idle_check_interval_s: float = DEFAULT_CONFIG["idle_check_interval_s"]
    config_path: Optional[str] = None
    # The key this daemon requires from its clients and presents to the
    # worker it spawns; None leaves both sides open.
    api_key: Optional[str] = None


class WorkerSupervisor:
    """Own one inference worker process and keep it healthy when requested."""

    def __init__(self, settings: GatewaySettings, client: httpx.AsyncClient):
        if not settings.worker_command:
            raise ValueError("worker_command must not be empty")
        self.settings = settings
        self.client = client
        self.process: Optional[asyncio.subprocess.Process] = None
        self.desired_running = True
        self._state = "stopped"
        self.state_since_unix = time.time()
        self.restart_count = 0
        self.start_failures = 0
        self.last_restart_reason: Optional[str] = None
        self.last_error: Optional[str] = None
        self.worker_health: dict = {}
        self.consecutive_500 = 0
        self.requests_forwarded = 0
        self.requests_completed = 0
        self.requests_failed = 0
        self.requests_cancelled = 0
        self.active_requests = 0
        self.generation = 0
        self.last_activity_at = time.monotonic()
        self._started_at = time.monotonic()
        self._spawned_at: Optional[float] = None
        self._startup_retry_not_before: Optional[float] = None
        self._ready_event = asyncio.Event()
        self._operation_lock = asyncio.Lock()
        self._monitor_task: Optional[asyncio.Task] = None
        self._restart_task: Optional[asyncio.Task] = None
        self._closed = False

    @property
    def state(self) -> str:
        return self._state

    @state.setter
    def state(self, value: str) -> None:
        if value != self._state:
            self._state = value
            self.state_since_unix = time.time()

    def snapshot(self) -> dict:
        return {
            "status": self.state,
            "desired_state": "running" if self.desired_running else "stopped",
            "pid": None if self.process is None else self.process.pid,
            "restart_count": self.restart_count,
            "last_restart_reason": self.last_restart_reason,
            "last_error": self.last_error,
            "active_requests": self.active_requests,
            "generation": self.generation,
            "uptime_s": max(0.0, time.monotonic() - self._started_at),
            "worker": self.worker_health,
        }

    def metrics(self) -> dict:
        return {
            **self.snapshot(),
            "requests_forwarded": self.requests_forwarded,
            "requests_completed": self.requests_completed,
            "requests_failed": self.requests_failed,
            "requests_cancelled": self.requests_cancelled,
            "consecutive_internal_errors": self.consecutive_500,
        }

    async def open(self) -> None:
        if self._monitor_task is not None:
            return
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(), name="mlx-vlm-worker-monitor"
        )
        await self._ensure_process()

    async def close(self) -> None:
        self._closed = True
        self.desired_running = False
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        await self._wait_for_restart()
        async with self._operation_lock:
            await self._terminate_locked()
            self.state = "stopped"

    async def start_worker(self, *, wait_ready: bool = True) -> dict:
        if self.state == "error":
            if not self.startup_retry_due():
                raise RuntimeError(self.last_error or "Worker is in error state")
            logger.info("Retrying inference worker after startup cooldown")
            self._startup_retry_not_before = None
            self.state = "stopped"
        if not self.desired_running or self.state == "stopped":
            self.start_failures = 0
        self.desired_running = True
        await self._ensure_process()
        if wait_ready:
            await self.wait_until_ready()
        return self.snapshot()

    def startup_retry_due(self) -> bool:
        return (
            self.state == "error"
            and self._startup_retry_not_before is not None
            and time.monotonic() >= self._startup_retry_not_before
        )

    async def wait_until_ready(self) -> None:
        if self.state == "ready":
            return
        if not self.desired_running or self.state in {"stopped", "stopping", "error"}:
            raise RuntimeError(self.last_error or "Worker is not running")
        try:
            await asyncio.wait_for(
                self._ready_event.wait(), timeout=self.settings.startup_timeout_s
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Worker did not become ready in time") from exc
        if self.state != "ready":
            raise RuntimeError(self.last_error or "Worker failed to start")

    async def stop_worker(self) -> dict:
        self.desired_running = False
        self.state = "stopping"
        self._ready_event.set()
        await self._wait_for_restart()
        async with self._operation_lock:
            await self._terminate_locked()
            self.worker_health = {}
            self.state = "stopped"
        return self.snapshot()

    async def _ensure_process(self) -> None:
        async with self._operation_lock:
            if not self.desired_running or self._closed:
                return
            if self.process is not None and self.process.returncode is None:
                return
            await self._spawn_locked()

    def _open_worker_log(self):
        """Where the worker's output goes, or nothing to inherit ours.

        Opened per spawn and in append mode, so every worker of this run
        writes to the same file, one after another. Rotation happens once,
        when the daemon starts.
        """
        path = self.settings.worker_log_path
        return open(path, "a", encoding="utf-8") if path else nullcontext(None)

    async def _spawn_locked(self) -> None:
        command = self.settings.worker_command
        logger.info("Starting inference worker: %s", " ".join(command))
        kwargs = {}
        worker_env = os.environ.copy()
        worker_env[GATEWAY_PID_ENV] = str(os.getpid())
        # Every other setting the worker needs is in the config file it
        # reads for itself.
        if self.settings.config_path:
            worker_env[CONFIG_PATH_ENV] = self.settings.config_path
        # The worker enforces the same key on its private API, and reads it
        # from the env; a config without one leaves that API open, so an
        # inherited value must not linger.
        if self.settings.api_key:
            worker_env[WORKER_API_KEY_ENV] = self.settings.api_key
        else:
            worker_env.pop(WORKER_API_KEY_ENV, None)
        kwargs["env"] = worker_env
        if os.name != "nt":
            kwargs["start_new_session"] = True
        self._spawned_at = time.monotonic()
        self._ready_event.clear()
        self.worker_health = {}
        self.state = "starting"
        self.last_error = None
        try:
            # The child dups whatever it is handed, so this copy is only
            # needed for the length of the spawn.
            with self._open_worker_log() as log:
                if log is not None:
                    kwargs["stdout"] = log
                    kwargs["stderr"] = log
                self.process = await asyncio.create_subprocess_exec(*command, **kwargs)
        except OSError as exc:
            self.process = None
            self.last_error = f"Failed to start inference worker: {exc}"
            logger.error(self.last_error)
            return
        self.generation += 1
        self.consecutive_500 = 0

    async def _terminate_locked(self) -> None:
        process = self.process
        self.process = None
        self._ready_event.clear()
        if process is None or process.returncode is not None:
            return
        logger.info("Stopping inference worker pid=%s", process.pid)
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(
                process.wait(), timeout=self.settings.shutdown_timeout_s
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Worker did not stop gracefully; killing pid=%s", process.pid
            )
            try:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            await process.wait()

    async def _probe(self) -> bool:
        try:
            response = await self.client.get(
                f"{self.settings.worker_url}/ready",
                timeout=self.settings.probe_timeout_s,
                headers=auth_headers(self.settings.api_key),
            )
        except httpx.RequestError as exc:
            self.last_error = str(exc)
            return False
        try:
            self.worker_health = response.json()
        except ValueError:
            self.worker_health = {"status_code": response.status_code}
        if response.status_code == 200:
            self.last_error = None
            return True
        self.last_error = f"Worker readiness returned HTTP {response.status_code}"
        return False

    async def _monitor_loop(self) -> None:
        failed_probes = 0
        while True:
            try:
                await asyncio.sleep(
                    self.settings.startup_probe_interval_s
                    if self.state == "starting"
                    else self.settings.probe_interval_s
                )
                if not self.desired_running or self._closed:
                    continue
                if self.state == "error":
                    continue
                process = self.process
                if process is None or process.returncode is not None:
                    reason = (
                        self.last_error or "worker process missing"
                        if process is None
                        else f"worker exited with code {process.returncode}"
                    )
                    self.schedule_restart(
                        reason,
                        startup_failure=self.state in {"starting", "restarting"},
                    )
                    continue

                ready = await self._probe()
                if ready:
                    failed_probes = 0
                    self.start_failures = 0
                    if self.state != "ready":
                        self.last_activity_at = time.monotonic()
                    self.state = "ready"
                    self._ready_event.set()
                    continue

                failed_probes += 1
                if self.state == "starting":
                    elapsed = time.monotonic() - (self._spawned_at or time.monotonic())
                    if elapsed >= self.settings.startup_timeout_s:
                        self.schedule_restart(
                            "worker startup timeout", startup_failure=True
                        )
                    continue
                if failed_probes >= self.settings.probe_failures_before_restart:
                    failed_probes = 0
                    self.schedule_restart("worker readiness failed repeatedly")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Worker monitor failed")

    def schedule_restart(
        self,
        reason: str,
        *,
        generation: Optional[int] = None,
        startup_failure: bool = False,
    ) -> None:
        if generation is not None and generation != self.generation:
            return
        if not self.desired_running or self._closed:
            return
        if self._restart_task is not None and not self._restart_task.done():
            return
        self.last_restart_reason = reason
        self.last_error = reason
        self.restart_count += 1
        self.state = "restarting"
        self._ready_event.clear()
        self._restart_task = asyncio.create_task(
            self._restart(reason, startup_failure=startup_failure),
            name="mlx-vlm-worker-restart",
        )

    async def _wait_for_restart(self) -> None:
        task = self._restart_task
        if task is None or task is asyncio.current_task():
            return
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            if self._restart_task is task:
                self._restart_task = None

    async def _restart(self, reason: str, *, startup_failure: bool = False) -> None:
        if startup_failure:
            self.start_failures += 1
        if self.desired_running and not self._closed:
            self.state = "restarting"
            self._ready_event.clear()
        if self.start_failures >= self.settings.max_start_failures:
            logger.error(
                "Inference worker failed to start %d times; stopping restart loop: %s",
                self.start_failures,
                reason,
            )
            async with self._operation_lock:
                await self._terminate_locked()
                self.state = "error" if self.desired_running else "stopped"
                if self.state == "error":
                    self._startup_retry_not_before = (
                        time.monotonic() + self.settings.startup_retry_cooldown_s
                    )
                self._ready_event.set()
            return
        logger.warning("Restarting inference worker: %s", reason)
        async with self._operation_lock:
            await self._terminate_locked()
            if not self.desired_running or self._closed:
                self.state = "stopped"
                return
        await asyncio.sleep(self.settings.restart_delay_s)
        async with self._operation_lock:
            if not self.desired_running or self._closed:
                self.state = "stopped"
                return
            if self.process is not None and self.process.returncode is None:
                return
            await self._spawn_locked()

    def record_worker_response(self, status_code: int, *, generation: int) -> bool:
        """Return True when this response reaches the restart threshold."""
        if generation != self.generation:
            return False
        if status_code == 500:
            self.consecutive_500 += 1
        else:
            self.consecutive_500 = 0
        if self.consecutive_500 < 2:
            return False
        self.consecutive_500 = 0
        self.schedule_restart(
            "two consecutive worker HTTP 500 responses",
            generation=generation,
        )
        return True
