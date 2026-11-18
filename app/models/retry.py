"""Bounded retries shared by the native Bailian HTTP adapters."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
INITIAL_RETRY_DELAY_SECONDS = 0.5
MAX_RETRY_DELAY_SECONDS = 8.0

SyncSleep = Callable[[float], None]
AsyncSleep = Callable[[float], Awaitable[None]]


def retry_delay(retry_number: int) -> float:
    """Return a capped exponential delay for a zero-based retry number."""
    exponent = min(retry_number, 4)
    return min(
        INITIAL_RETRY_DELAY_SECONDS * (2**exponent),
        MAX_RETRY_DELAY_SECONDS,
    )


def is_retryable_status(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS_CODES or 500 <= status_code <= 599


def post_with_retries(
    client: httpx.Client,
    url: str,
    *,
    json: dict[str, Any],
    headers: dict[str, str],
    max_retries: int,
    sleep: SyncSleep = time.sleep,
) -> httpx.Response:
    """POST with retries limited to transient statuses and connect/read timeouts."""
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    for attempt in range(max_retries + 1):
        try:
            response = client.post(url, json=json, headers=headers)
        except (httpx.ConnectTimeout, httpx.ReadTimeout):
            if attempt == max_retries:
                raise
        else:
            if not is_retryable_status(response.status_code) or attempt == max_retries:
                response.raise_for_status()
                return response
        sleep(retry_delay(attempt))

    raise AssertionError("retry loop exhausted without returning or raising")


async def apost_with_retries(
    client: httpx.AsyncClient,
    url: str,
    *,
    json: dict[str, Any],
    headers: dict[str, str],
    max_retries: int,
    sleep: AsyncSleep = asyncio.sleep,
) -> httpx.Response:
    """Async counterpart to :func:`post_with_retries`."""
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    for attempt in range(max_retries + 1):
        try:
            response = await client.post(url, json=json, headers=headers)
        except (httpx.ConnectTimeout, httpx.ReadTimeout):
            if attempt == max_retries:
                raise
        else:
            if not is_retryable_status(response.status_code) or attempt == max_retries:
                response.raise_for_status()
                return response
        await sleep(retry_delay(attempt))

    raise AssertionError("retry loop exhausted without returning or raising")
