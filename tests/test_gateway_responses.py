import asyncio
import json
from types import SimpleNamespace

import httpx

from mlx_vlm_gateway.responses import proxy_responses


def event(kind, **fields):
    return "data: " + json.dumps({"type": kind, **fields}) + "\n\n"


def test_retry_before_output_uses_one_public_response_identity():
    async def run():
        calls = []
        restarts = []
        sup = SimpleNamespace(
            generation=1,
            requests_failed=0,
            schedule_restart=lambda *a, **k: restarts.append(a),
        )

        async def ready():
            pass

        async def handler(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(
                    200,
                    text=event("response.created", response={"id": "old"})
                    + event(
                        "response.failed",
                        response={"error": {"code": "runtime_unavailable"}},
                    ),
                )
            return httpx.Response(
                200,
                text=event("response.output_text.delta", delta="hello")
                + event(
                    "response.completed", response={"id": "new", "status": "completed"}
                ),
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            chunks = [
                x
                async for x in proxy_responses(
                    client, "http://worker/v1/responses", {}, {}, sup, ready, 2
                )
            ]
        records = [json.loads(x.decode().split("data: ")[1]) for x in chunks]
        assert len(calls) == 2 and len(restarts) == 1
        assert [x["type"] for x in records] == [
            "response.created",
            "response.output_text.delta",
            "response.completed",
        ]
        assert records[0]["response"]["id"] == records[-1]["response"]["id"]

    asyncio.run(run())


def test_no_retry_after_visible_output():
    async def run():
        calls = []
        sup = SimpleNamespace(
            generation=1, requests_failed=0, schedule_restart=lambda *a, **k: None
        )

        async def ready():
            pass

        async def handler(request):
            calls.append(request)
            return httpx.Response(
                200,
                text=event(
                    "response.output_item.added", item={"id": "tool"}, output_index=0
                )
                + event(
                    "response.failed",
                    response={"error": {"code": "runtime_unavailable"}},
                ),
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            chunks = [
                x
                async for x in proxy_responses(
                    client, "http://worker/v1/responses", {}, {}, sup, ready, 2
                )
            ]
        assert len(calls) == 1
        assert b"response.failed" in chunks[-1]
        assert sup.requests_failed == 1

    asyncio.run(run())


def test_heartbeat_during_readiness_and_cancel_cleanup():
    async def run():
        stopped = asyncio.Event()
        sup = SimpleNamespace(generation=1, requests_failed=0)

        async def ready():
            try:
                await asyncio.sleep(100)
            finally:
                stopped.set()

        async with httpx.AsyncClient() as client:
            stream = proxy_responses(
                client, "http://worker", {}, {}, sup, ready, 2, heartbeat=0.01
            )
            assert b"response.created" in await anext(stream)
            assert b"response.in_progress" in await anext(stream)
            await stream.aclose()
            assert stopped.is_set()

    asyncio.run(run())


def test_retry_budget_and_nonrecoverable_error():
    async def run(code, expected):
        calls = []
        sup = SimpleNamespace(
            generation=1, requests_failed=0, schedule_restart=lambda *a, **k: None
        )

        async def ready():
            pass

        async def handler(request):
            calls.append(request)
            return httpx.Response(
                503 if code == "runtime_unavailable" else 400,
                json={"error": {"code": code, "message": "test"}},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            chunks = [
                x
                async for x in proxy_responses(
                    client, "http://worker", {}, {}, sup, ready, 2
                )
            ]
        assert len(calls) == expected
        assert b"response.failed" in chunks[-1]

    asyncio.run(run("runtime_unavailable", 3))
    asyncio.run(run("invalid_request_error", 1))


def test_hard_deadline_ends_waiting_stream():
    async def run():
        sup = SimpleNamespace(generation=1, requests_failed=0)

        async def ready():
            await asyncio.sleep(100)

        async with httpx.AsyncClient() as client:
            chunks = [
                x
                async for x in proxy_responses(
                    client, "http://worker", {}, {}, sup, ready, 0.02, heartbeat=0.01
                )
            ]
        assert b"response.failed" in chunks[-1]
        assert sup.requests_failed == 1

    asyncio.run(run())


def test_cancelling_consumer_closes_worker_stream():
    async def run():
        closed = asyncio.Event()

        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield event("response.output_text.delta", delta="hello").encode()
                await asyncio.sleep(100)

            async def aclose(self):
                closed.set()

        sup = SimpleNamespace(generation=1, requests_failed=0)

        async def ready():
            pass

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=Body())
            )
        ) as client:
            stream = proxy_responses(client, "http://worker", {}, {}, sup, ready, 2)
            await anext(stream)
            assert b"hello" in await anext(stream)
            await stream.aclose()
            assert closed.is_set()

    asyncio.run(run())


def test_cancel_when_producer_is_blocked_on_full_queue_closes_upstream():
    async def run():
        closed = asyncio.Event()

        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                for _ in range(100):
                    yield event("response.output_text.delta", delta="x").encode()

            async def aclose(self):
                closed.set()

        sup = SimpleNamespace(generation=1, requests_failed=0)

        async def ready():
            pass

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=Body())
            )
        ) as client:
            stream = proxy_responses(client, "http://worker", {}, {}, sup, ready, 2)
            await anext(stream)
            await asyncio.sleep(0.02)
            await stream.aclose()
            assert closed.is_set()

    asyncio.run(run())
