"""Recover non-streaming Splash requests before returning any output to the client."""

import asyncio

import httpx


async def splash_completion(
    client, url, payload, headers, timeout, supervisor, ensure_ready
):
    try:
        async with asyncio.timeout(timeout):
            for attempt in range(3):
                generation = supervisor.generation
                try:
                    response = await client.post(
                        url, json=payload, headers=headers, timeout=timeout
                    )
                    try:
                        error = response.json().get("error", {})
                    except (ValueError, AttributeError):
                        error = {}
                    unavailable = (
                        response.status_code == 503
                        and isinstance(error, dict)
                        and error.get("code") == "runtime_unavailable"
                    )
                    if not unavailable:
                        return response
                    if attempt == 2:
                        return response
                    reason = "Splash runtime unavailable before completion"
                except httpx.TimeoutException:
                    raise
                except httpx.TransportError:
                    if attempt == 2:
                        raise
                    reason = "Splash connection lost before completion"
                supervisor.schedule_restart(reason, generation=generation)
                await asyncio.sleep(0)
                try:
                    await ensure_ready()
                except RuntimeError as exc:
                    raise httpx.ConnectError("Splash did not recover") from exc
    except TimeoutError as exc:
        raise httpx.ReadTimeout("Splash completion deadline exceeded") from exc
