import asyncio
import json
import signal
import sys
import threading
import time

import httpx
from fastapi.testclient import TestClient

import mlx_vlm_gateway.app as gateway_module
from mlx_vlm_gateway.app import GatewaySettings, create_app


class FakeProcess:
    _next_pid = 41000

    def __init__(self):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.returncode = None
        self._done = asyncio.Event()

    def terminate(self):
        self.returncode = 0
        self._done.set()

    def kill(self):
        self.returncode = -9
        self._done.set()

    async def wait(self):
        await self._done.wait()
        return self.returncode


def _gateway(monkeypatch, handler):
    processes = []

    async def fake_create_subprocess(*_args, **_kwargs):
        process = FakeProcess()
        processes.append(process)
        return process

    def fake_killpg(pid, sig):
        process = next(process for process in processes if process.pid == pid)
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()

    monkeypatch.setattr(
        gateway_module.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess,
    )
    monkeypatch.setattr(gateway_module.os, "killpg", fake_killpg, raising=False)

    transport = httpx.MockTransport(handler)

    def client_factory(timeout):
        return httpx.AsyncClient(transport=transport, timeout=timeout)

    settings = GatewaySettings(
        worker_command=(sys.executable, "-c", "pass"),
        startup_timeout_s=1.0,
        request_timeout_s=0.5,
        startup_probe_interval_s=0.01,
        probe_interval_s=0.02,
        probe_timeout_s=0.1,
        restart_delay_s=0.01,
        shutdown_timeout_s=0.1,
    )
    return create_app(settings, client_factory=client_factory), processes


def _wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def test_gateway_forwards_batch_requests_and_controls_worker(monkeypatch):
    captured = []

    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(
                200,
                json={"status": "ready", "loaded_model": "demo"},
            )
        if request.url.path == "/v1/chat/completions":
            captured.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "timings": {"generation_tps": 42.0},
                },
            )
        if request.url.path == "/metrics":
            return httpx.Response(200, json={"requests": {"completed": 1}})
        if request.url.path == "/cache/stats":
            return httpx.Response(200, json={"enabled": True})
        if request.url.path == "/cache/reset":
            return httpx.Response(200, json={"status": "cleared"})
        raise AssertionError(request.url.path)

    app, processes = _gateway(monkeypatch, handler)
    with TestClient(app) as client:
        assert client.post("/start_worker").status_code == 200
        response = client.post(
            "/v1/chat/completions",
            json={"model": "any-model", "messages": [], "stream": True},
        )
        assert response.status_code == 200
        assert response.json()["timings"]["generation_tps"] == 42.0
        assert captured == [
            {"model": "any-model", "messages": [], "stream": False}
        ]

        metrics = client.get("/metrics").json()
        assert metrics["requests"]["completed"] == 1
        assert metrics["gateway"]["requests_completed"] == 1
        assert client.get("/cache/stats").json() == {"enabled": True}
        assert client.post("/cache/reset").json() == {"status": "cleared"}

        assert client.post("/stop_worker").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/v1/chat/completions", json={}).status_code == 503
        process_count = len(processes)
        time.sleep(0.06)
        assert len(processes) == process_count

        assert client.post("/start_worker").status_code == 200
        assert len(processes) == process_count + 1


def test_second_consecutive_500_restarts_worker(monkeypatch):
    inference_calls = 0

    def handler(request):
        nonlocal inference_calls
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/chat/completions":
            inference_calls += 1
            return httpx.Response(500, json={"detail": "generation failed"})
        raise AssertionError(request.url.path)

    app, processes = _gateway(monkeypatch, handler)
    with TestClient(app) as client:
        assert client.post("/start_worker").status_code == 200
        assert client.post("/v1/chat/completions", json={}).status_code == 500
        assert len(processes) == 1
        second = client.post("/v1/chat/completions", json={})
        assert second.status_code == 503
        assert inference_calls == 2
        _wait_until(lambda: len(processes) == 2)


def test_422_between_500_responses_resets_restart_counter(monkeypatch):
    statuses = iter((500, 422, 500))

    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(next(statuses), json={"detail": "test"})
        raise AssertionError(request.url.path)

    app, processes = _gateway(monkeypatch, handler)
    with TestClient(app) as client:
        assert client.post("/start_worker").status_code == 200
        assert client.post("/v1/chat/completions", json={}).status_code == 500
        assert client.post("/v1/chat/completions", json={}).status_code == 422
        assert client.post("/v1/chat/completions", json={}).status_code == 500
        time.sleep(0.06)
        assert len(processes) == 1


def test_worker_connection_failure_returns_503_and_restarts(monkeypatch):
    fail_inference = True

    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/chat/completions" and fail_inference:
            raise httpx.ConnectError("worker exited", request=request)
        raise AssertionError(request.url.path)

    app, processes = _gateway(monkeypatch, handler)
    with TestClient(app) as client:
        assert client.post("/start_worker").status_code == 200
        response = client.post("/v1/chat/completions", json={})
        assert response.status_code == 503
        assert response.json()["detail"] == "Inference worker restarted; please retry"
        _wait_until(lambda: len(processes) == 2)


def test_hard_timeout_returns_504_and_restarts_worker(monkeypatch):
    def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/chat/completions":
            raise httpx.ReadTimeout("generation did not stop", request=request)
        raise AssertionError(request.url.path)

    app, processes = _gateway(monkeypatch, handler)
    with TestClient(app) as client:
        assert client.post("/start_worker").status_code == 200
        response = client.post("/v1/chat/completions", json={})
        assert response.status_code == 504
        assert response.json()["detail"] == "Inference worker did not stop in time"
        _wait_until(lambda: len(processes) == 2)


def test_manual_stop_interrupts_active_request_without_restart(monkeypatch):
    inference_started = threading.Event()
    processes = None

    async def handler(request):
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path == "/v1/chat/completions":
            inference_started.set()
            while processes[0].returncode is None:
                await asyncio.sleep(0.005)
            raise httpx.ConnectError("worker stopped", request=request)
        raise AssertionError(request.url.path)

    app, processes = _gateway(monkeypatch, handler)
    with TestClient(app) as client:
        assert client.post("/start_worker").status_code == 200
        result = {}

        def send_request():
            result["response"] = client.post("/v1/chat/completions", json={})

        request_thread = threading.Thread(target=send_request)
        request_thread.start()
        assert inference_started.wait(timeout=1.0)

        assert client.post("/stop_worker").status_code == 200
        request_thread.join(timeout=1.0)
        assert not request_thread.is_alive()
        assert result["response"].status_code == 503

        process_count = len(processes)
        time.sleep(0.06)
        assert len(processes) == process_count
        assert client.get("/ready").status_code == 503
