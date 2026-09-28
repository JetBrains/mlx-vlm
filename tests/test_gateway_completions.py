import asyncio
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from mlx_vlm_gateway.completions import splash_completion


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("code", ["invalid_model_output", "invalid_request_error"])
async def test_output_errors_do_not_restart_or_retry(code):
    supervisor = Mock(generation=1)
    client = Mock(
        post=AsyncMock(return_value=httpx.Response(500, json={"error": {"code": code}}))
    )
    ready = AsyncMock()
    response = await splash_completion(
        client, "http://worker", {}, {}, 1, supervisor, ready
    )
    assert response.status_code == 500
    assert client.post.await_count == 1
    supervisor.schedule_restart.assert_not_called()
    ready.assert_not_awaited()


@pytest.mark.anyio
async def test_unavailable_retries_are_bounded():
    supervisor = Mock(generation=1)
    client = Mock(
        post=AsyncMock(
            return_value=httpx.Response(
                503, json={"error": {"code": "runtime_unavailable"}}
            )
        )
    )
    response = await splash_completion(
        client, "http://worker", {}, {}, 1, supervisor, AsyncMock()
    )
    assert response.status_code == 503
    assert client.post.await_count == 3
    assert supervisor.schedule_restart.call_count == 2


@pytest.mark.anyio
async def test_failed_restart_is_a_transport_error():
    supervisor = Mock(generation=1)
    client = Mock(post=AsyncMock(side_effect=httpx.ConnectError("worker lost")))
    with pytest.raises(httpx.ConnectError, match="did not recover"):
        await splash_completion(
            client,
            "http://worker",
            {},
            {},
            1,
            supervisor,
            AsyncMock(side_effect=RuntimeError("startup failed")),
        )
    assert client.post.await_count == 1


@pytest.mark.anyio
async def test_deadline_includes_recovery_and_cancellation_propagates():
    supervisor = Mock(generation=1)
    client = Mock(post=AsyncMock(side_effect=httpx.ConnectError("worker lost")))
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def ready():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    with pytest.raises(httpx.ReadTimeout, match="deadline exceeded"):
        await splash_completion(
            client, "http://worker", {}, {}, 0.01, supervisor, ready
        )
    assert cancelled.is_set()
    cancelled.clear()
    task = asyncio.create_task(
        splash_completion(client, "http://worker", {}, {}, 1, supervisor, ready)
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
