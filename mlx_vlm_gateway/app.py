import argparse
import asyncio
import logging
import os
import signal
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response


logger = logging.getLogger("mlx_vlm.gateway")


@dataclass(frozen=True)
class GatewaySettings:
    worker_url: str = "http://127.0.0.1:8086"
    worker_command: tuple[str, ...] = ()
    startup_timeout_s: float = 120.0
    request_timeout_s: float = 275.0
    startup_probe_interval_s: float = 1.0
    probe_interval_s: float = 5.0
    probe_timeout_s: float = 2.0
    probe_failures_before_restart: int = 3
    restart_delay_s: float = 2.0
    shutdown_timeout_s: float = 5.0


class WorkerSupervisor:
    """Own one inference worker process and keep it healthy when requested."""

    def __init__(self, settings: GatewaySettings, client: httpx.AsyncClient):
        if not settings.worker_command:
            raise ValueError("worker_command must not be empty")
        self.settings = settings
        self.client = client
        self.process: Optional[asyncio.subprocess.Process] = None
        self.desired_running = True
        self.state = "stopped"
        self.restart_count = 0
        self.last_restart_reason: Optional[str] = None
        self.last_error: Optional[str] = None
        self.worker_health: dict = {}
        self.consecutive_500 = 0
        self.requests_forwarded = 0
        self.requests_completed = 0
        self.requests_failed = 0
        self.active_requests = 0
        self._started_at = time.monotonic()
        self._spawned_at: Optional[float] = None
        self._ready_event = asyncio.Event()
        self._operation_lock = asyncio.Lock()
        self._monitor_task: Optional[asyncio.Task] = None
        self._restart_task: Optional[asyncio.Task] = None
        self._closed = False

    def snapshot(self) -> dict:
        return {
            "status": self.state,
            "desired_state": "running" if self.desired_running else "stopped",
            "pid": None if self.process is None else self.process.pid,
            "restart_count": self.restart_count,
            "last_restart_reason": self.last_restart_reason,
            "last_error": self.last_error,
            "active_requests": self.active_requests,
            "uptime_s": max(0.0, time.monotonic() - self._started_at),
            "worker": self.worker_health,
        }

    def metrics(self) -> dict:
        return {
            **self.snapshot(),
            "requests_forwarded": self.requests_forwarded,
            "requests_completed": self.requests_completed,
            "requests_failed": self.requests_failed,
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
        self.desired_running = True
        await self._ensure_process()
        if wait_ready and self.state != "ready":
            try:
                await asyncio.wait_for(
                    self._ready_event.wait(), timeout=self.settings.startup_timeout_s
                )
            except asyncio.TimeoutError as exc:
                raise RuntimeError("Worker did not become ready in time") from exc
        return self.snapshot()

    async def stop_worker(self) -> dict:
        self.desired_running = False
        self._ready_event.clear()
        await self._wait_for_restart()
        async with self._operation_lock:
            self.state = "stopping"
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

    async def _spawn_locked(self) -> None:
        command = self.settings.worker_command
        logger.info("Starting inference worker: %s", " ".join(command))
        kwargs = {}
        if os.name != "nt":
            kwargs["start_new_session"] = True
        self.process = await asyncio.create_subprocess_exec(*command, **kwargs)
        self._spawned_at = time.monotonic()
        self._ready_event.clear()
        self.worker_health = {}
        self.state = "starting"
        self.last_error = None

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
            logger.warning("Worker did not stop gracefully; killing pid=%s", process.pid)
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
                process = self.process
                if process is None or process.returncode is not None:
                    reason = (
                        "worker process missing"
                        if process is None
                        else f"worker exited with code {process.returncode}"
                    )
                    self.schedule_restart(reason)
                    continue

                ready = await self._probe()
                if ready:
                    failed_probes = 0
                    self.state = "ready"
                    self._ready_event.set()
                    continue

                failed_probes += 1
                if self.state == "starting":
                    elapsed = time.monotonic() - (self._spawned_at or time.monotonic())
                    if elapsed >= self.settings.startup_timeout_s:
                        self.schedule_restart("worker startup timeout")
                    continue
                if failed_probes >= self.settings.probe_failures_before_restart:
                    failed_probes = 0
                    self.schedule_restart("worker readiness failed repeatedly")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Worker monitor failed")

    def schedule_restart(self, reason: str) -> None:
        if not self.desired_running or self._closed:
            return
        if self._restart_task is not None and not self._restart_task.done():
            return
        self._restart_task = asyncio.create_task(
            self._restart(reason), name="mlx-vlm-worker-restart"
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

    async def _restart(self, reason: str) -> None:
        self.last_restart_reason = reason
        self.last_error = reason
        self.restart_count += 1
        self.state = "restarting"
        self._ready_event.clear()
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

    def record_worker_response(self, status_code: int) -> bool:
        """Return True when this response reaches the restart threshold."""
        if status_code == 500:
            self.consecutive_500 += 1
        else:
            self.consecutive_500 = 0
        if self.consecutive_500 < 2:
            return False
        self.consecutive_500 = 0
        self.schedule_restart("two consecutive worker HTTP 500 responses")
        return True


def _proxy_response(response: httpx.Response) -> Response:
    headers = {}
    content_type = response.headers.get("content-type")
    if content_type:
        headers["content-type"] = content_type
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=headers,
    )


def create_app(
    settings: GatewaySettings,
    client_factory: Optional[Callable[[httpx.Timeout], httpx.AsyncClient]] = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        timeout = httpx.Timeout(
            connect=2.0,
            read=settings.request_timeout_s,
            write=30.0,
            pool=5.0,
        )
        client = (
            client_factory(timeout)
            if client_factory is not None
            else httpx.AsyncClient(timeout=timeout)
        )
        supervisor = WorkerSupervisor(settings, client)
        app.state.client = client
        app.state.supervisor = supervisor
        await supervisor.open()
        try:
            yield
        finally:
            await supervisor.close()
            await client.aclose()

    app = FastAPI(title="MLX-VLM Gateway", lifespan=lifespan)

    def supervisor(request: Request) -> WorkerSupervisor:
        return request.app.state.supervisor

    @app.get("/health")
    async def health(request: Request):
        return {"status": "healthy", "worker": supervisor(request).snapshot()}

    @app.get("/ready")
    async def ready(request: Request):
        state = supervisor(request).snapshot()
        status_code = 200 if state["status"] == "ready" else 503
        return JSONResponse({"status": state["status"], "worker": state}, status_code)

    @app.post("/start_worker")
    async def start_worker(request: Request):
        try:
            state = await supervisor(request).start_worker(wait_ready=True)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"status": "ready", "worker": state}

    @app.post("/stop_worker")
    async def stop_worker(request: Request):
        state = await supervisor(request).stop_worker()
        return {"status": "stopped", "worker": state}

    async def management_proxy(request: Request, method: str, path: str):
        sup = supervisor(request)
        if sup.state != "ready":
            raise HTTPException(status_code=503, detail="Inference worker is not ready")
        try:
            response = await request.app.state.client.request(
                method, f"{settings.worker_url}{path}"
            )
        except httpx.RequestError as exc:
            sup.schedule_restart(f"worker connection failed: {exc}")
            raise HTTPException(
                status_code=503,
                detail="Inference worker restarted; please retry",
            ) from exc
        return response

    @app.get("/metrics")
    async def metrics(request: Request):
        response = await management_proxy(request, "GET", "/metrics")
        try:
            payload = response.json()
        except ValueError:
            return _proxy_response(response)
        payload["gateway"] = supervisor(request).metrics()
        return JSONResponse(payload, status_code=response.status_code)

    @app.get("/cache/stats")
    async def cache_stats(request: Request):
        return _proxy_response(
            await management_proxy(request, "GET", "/cache/stats")
        )

    @app.post("/cache/reset")
    async def cache_reset(request: Request):
        return _proxy_response(
            await management_proxy(request, "POST", "/cache/reset")
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        sup = supervisor(request)
        if sup.state != "ready":
            raise HTTPException(
                status_code=503,
                detail="Inference worker is not ready; call /start_worker if it was stopped",
                headers={"Retry-After": "5"},
            )
        try:
            payload = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Request body must be JSON") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")
        payload["stream"] = False
        headers = {"content-type": "application/json"}
        for name in ("x-apc-tenant", "x-tenant-id"):
            if value := request.headers.get(name):
                headers[name] = value

        sup.requests_forwarded += 1
        sup.active_requests += 1
        try:
            response = await request.app.state.client.post(
                f"{settings.worker_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=settings.request_timeout_s,
            )
        except httpx.TimeoutException as exc:
            sup.requests_failed += 1
            sup.schedule_restart("worker exceeded hard request timeout")
            raise HTTPException(
                status_code=504, detail="Inference worker did not stop in time"
            ) from exc
        except httpx.RequestError as exc:
            sup.requests_failed += 1
            sup.schedule_restart(f"worker connection failed during inference: {exc}")
            raise HTTPException(
                status_code=503,
                detail="Inference worker restarted; please retry",
            ) from exc
        finally:
            sup.active_requests = max(0, sup.active_requests - 1)

        if sup.record_worker_response(response.status_code):
            sup.requests_failed += 1
            raise HTTPException(
                status_code=503,
                detail="Inference worker restarted after repeated internal errors; please retry",
            )
        if response.status_code >= 400:
            sup.requests_failed += 1
        else:
            sup.requests_completed += 1
        return _proxy_response(response)

    return app


def _parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="MLX-VLM gateway and worker supervisor")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument("--worker-url", default="http://127.0.0.1:8086")
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--request-timeout", type=float, default=275.0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("worker_command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.worker_command and args.worker_command[0] == "--":
        args.worker_command = args.worker_command[1:]
    if not args.worker_command:
        parser.error("worker command is required after --")
    return args


def main(argv: Optional[Sequence[str]] = None):
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    settings = GatewaySettings(
        worker_url=args.worker_url.rstrip("/"),
        worker_command=tuple(args.worker_command),
        startup_timeout_s=args.startup_timeout,
        request_timeout_s=args.request_timeout,
    )
    uvicorn.run(
        create_app(settings),
        host=args.host,
        port=args.port,
        workers=1,
        server_header=False,
        log_level=args.log_level.lower(),
    )
