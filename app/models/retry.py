"""Bounded retries shared by the native Bailian HTTP adapters."""

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
INITIAL_RETRY_DELAY_SECONDS = 0.5
MAX_RETRY_DELAY_SECONDS = 8.0
DEFAULT_TOTAL_DEADLINE_SECONDS = 30.0

SyncSleep = Callable[[float], None]
AsyncSleep = Callable[[float], Awaitable[None]]
RetryJitter = Callable[[float], float]
Monotonic = Callable[[], float]


def full_jitter(delay: float) -> float:
    return random.uniform(0.0, delay)


def retry_delay(
    retry_number: int, *, jitter: RetryJitter = full_jitter
) -> float:
    """Return bounded jitter applied to a capped exponential delay."""
    exponent = min(retry_number, 4)
    capped = min(
        INITIAL_RETRY_DELAY_SECONDS * (2**exponent),
        MAX_RETRY_DELAY_SECONDS,
    )
    return max(0.0, min(float(jitter(capped)), MAX_RETRY_DELAY_SECONDS))


def is_retryable_status(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS_CODES or 500 <= status_code <= 599


def _validate_limits(max_retries: int, total_deadline_seconds: float) -> None:
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if total_deadline_seconds <= 0:
        raise ValueError("total_deadline_seconds must be positive")


def _remaining_seconds(deadline: float, monotonic: Monotonic) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise httpx.TimeoutException("Bailian retry deadline exceeded")
    return remaining


def _delay_within_deadline(
    retry_number: int,
    *,
    deadline: float,
    monotonic: Monotonic,
    jitter: RetryJitter,
) -> float:
    delay = retry_delay(retry_number, jitter=jitter)
    if delay >= _remaining_seconds(deadline, monotonic):
        raise httpx.TimeoutException("Bailian retry deadline exceeded")
    return delay


def post_with_retries(
    client: httpx.Client,
    url: str,
    *,
    json: dict[str, Any],
    headers: dict[str, str],
    max_retries: int,
    sleep: SyncSleep = time.sleep,
    jitter: RetryJitter = full_jitter,
    total_deadline_seconds: float = DEFAULT_TOTAL_DEADLINE_SECONDS,
    monotonic: Monotonic = time.monotonic,
) -> httpx.Response:
    """POST with transient retries bounded by attempts and a monotonic deadline."""
    _validate_limits(max_retries, total_deadline_seconds)
    deadline = monotonic() + total_deadline_seconds

    for attempt in range(max_retries + 1):
        _remaining_seconds(deadline, monotonic)
        try:
            response = client.post(url, json=json, headers=headers)
        except (httpx.ConnectTimeout, httpx.ReadTimeout):
            if attempt == max_retries:
                raise
        else:
            if not is_retryable_status(response.status_code) or attempt == max_retries:
                response.raise_for_status()
                return response
        delay = _delay_within_deadline(
            attempt, deadline=deadline, monotonic=monotonic, jitter=jitter
        )
        sleep(delay)

    raise AssertionError("retry loop exhausted without returning or raising")


async def apost_with_retries(
    client: httpx.AsyncClient,
    url: str,
    *,
    json: dict[str, Any],
    headers: dict[str, str],
    max_retries: int,
    sleep: AsyncSleep = asyncio.sleep,
    jitter: RetryJitter = full_jitter,
    total_deadline_seconds: float = DEFAULT_TOTAL_DEADLINE_SECONDS,
    monotonic: Monotonic = time.monotonic,
) -> httpx.Response:
    """Async counterpart to :func:`post_with_retries`."""
    _validate_limits(max_retries, total_deadline_seconds)
    deadline = monotonic() + total_deadline_seconds

    for attempt in range(max_retries + 1):
        _remaining_seconds(deadline, monotonic)
        try:
            response = await client.post(url, json=json, headers=headers)
        except (httpx.ConnectTimeout, httpx.ReadTimeout):
            if attempt == max_retries:
                raise
        else:
            if not is_retryable_status(response.status_code) or attempt == max_retries:
                response.raise_for_status()
                return response
        delay = _delay_within_deadline(
            attempt, deadline=deadline, monotonic=monotonic, jitter=jitter
        )
        await sleep(delay)

    raise AssertionError("retry loop exhausted without returning or raising")
