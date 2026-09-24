"""Bounded Responses streaming proxy; retries never replay visible output."""

import asyncio
import copy
import json
import time
import uuid

import httpx


class RecoverableWorkerError(Exception):
    pass


async def worker_events(client, url, payload, headers, timeout):
    async with client.stream(
        "POST", url, json=payload, headers=headers, timeout=timeout
    ) as response:
        if response.status_code >= 400:
            await response.aread()
            try:
                error = response.json().get("error", {})
            except ValueError:
                error = {}
            if (
                response.status_code == 503
                and error.get("code") == "runtime_unavailable"
            ):
                raise RecoverableWorkerError("runtime_unavailable")
            yield {
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": {
                        "code": error.get("code", "upstream_http_error"),
                        "message": error.get(
                            "message", f"Worker HTTP {response.status_code}"
                        ),
                    },
                },
            }
            return
        data = []
        async for line in response.aiter_lines():
            if not line:
                if data:
                    raw = "\n".join(data)
                    data = []
                    if raw == "[DONE]":
                        return
                    yield json.loads(raw)
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
        if data:
            yield json.loads("\n".join(data))


async def proxy_responses(
    client,
    url,
    payload,
    headers,
    supervisor,
    ensure_ready,
    timeout,
    heartbeat=15,
    max_attempts=3,
):
    """One public response identity/deadline across pre-output recovery attempts."""
    queue = asyncio.Queue(maxsize=1)
    response_id = "resp_gateway_" + uuid.uuid4().hex
    snapshot = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "in_progress",
        "model": payload.get("model"),
        "output": [],
        "error": None,
        "incomplete_details": None,
    }
    sequence = 0
    visible = False

    def encode(event):
        nonlocal sequence
        event = copy.deepcopy(event)
        event["sequence_number"] = sequence
        sequence += 1
        if isinstance(event.get("response"), dict):
            event["response"]["id"] = response_id
        if "response_id" in event:
            event["response_id"] = response_id
        return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()

    async def produce():
        nonlocal visible
        try:
            async with asyncio.timeout(timeout):
                for attempt in range(max_attempts):
                    generation = supervisor.generation
                    try:
                        await ensure_ready()
                        generation = supervisor.generation
                        terminal = False
                        async for event in worker_events(
                            client, url, payload, headers, timeout
                        ):
                            kind = event.get("type", "")
                            error = (
                                (event.get("response") or {}).get("error")
                                or event.get("error")
                                or {}
                            )
                            if (
                                kind in {"response.failed", "error"}
                                and error.get("code") == "runtime_unavailable"
                            ):
                                raise RecoverableWorkerError("runtime_unavailable")
                            if kind in {"response.created", "response.in_progress"}:
                                continue
                            # Conservatively commit even item-added / reasoning events.
                            visible = True
                            await queue.put(event)
                            if kind in {
                                "response.completed",
                                "response.incomplete",
                                "response.failed",
                                "error",
                            }:
                                terminal = True
                                break
                        if not terminal:
                            raise RecoverableWorkerError(
                                "Worker stream ended without a terminal event"
                            )
                        return
                    except (RecoverableWorkerError, httpx.TransportError) as exc:
                        supervisor.schedule_restart(str(exc), generation=generation)
                        if visible or attempt + 1 == max_attempts:
                            raise RecoverableWorkerError(str(exc)) from exc
                        await asyncio.sleep(0)  # allow shared restart task to start
        except (
            RecoverableWorkerError,
            httpx.HTTPError,
            RuntimeError,
            ValueError,
            TimeoutError,
        ) as exc:
            supervisor.requests_failed += 1
            await queue.put(
                {
                    "type": "response.failed",
                    "response": {
                        **snapshot,
                        "status": "failed",
                        "error": {
                            "code": (
                                "request_timeout"
                                if isinstance(exc, TimeoutError)
                                else "upstream_protocol_error"
                                if isinstance(exc, ValueError)
                                else "runtime_unavailable"
                            ),
                            "message": str(exc) or "Gateway request deadline exceeded",
                        },
                    },
                }
            )
        finally:
            # Cancellation must not block on a full queue after the consumer left.
            if not asyncio.current_task().cancelling():
                await queue.put(None)

    task = asyncio.create_task(produce())
    try:
        yield encode({"type": "response.created", "response": snapshot})
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), heartbeat)
            except asyncio.TimeoutError:
                yield encode({"type": "response.in_progress", "response": snapshot})
                continue
            if event is None:
                break
            if event["type"] in {
                "response.output_item.added",
                "response.output_item.done",
            }:
                index = event.get("output_index", 0)
                if index == len(snapshot["output"]):
                    snapshot["output"].append(event["item"])
                elif index < len(snapshot["output"]):
                    snapshot["output"][index] = event["item"]
            yield encode(event)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
