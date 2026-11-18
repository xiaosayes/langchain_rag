import httpx
import pytest

from app.models.retry import (
    MAX_RETRY_DELAY_SECONDS,
    apost_with_retries,
    post_with_retries,
    retry_delay,
)


def _response(status_code: int) -> httpx.Response:
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://example.invalid"),
    )


class SyncClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        response = self.responses.__next__()
        if isinstance(response, Exception):
            raise response
        return response


class AsyncClient(SyncClient):
    async def post(self, *args, **kwargs):
        return super().post(*args, **kwargs)


def test_retry_delay_is_capped_before_deterministic_jitter():
    assert retry_delay(20, jitter=lambda delay: delay) == MAX_RETRY_DELAY_SECONDS
    assert retry_delay(2, jitter=lambda delay: delay / 2) == 1.0


def test_retry_exhaustion_uses_bounded_delays_and_raises_last_status():
    client = SyncClient([_response(503), _response(503), _response(503)])
    sleeps = []

    with pytest.raises(httpx.HTTPStatusError):
        post_with_retries(
            client,
            "https://example.invalid",
            json={},
            headers={},
            max_retries=2,
            sleep=sleeps.append,
            jitter=lambda delay: delay / 2,
        )

    assert client.calls == 3
    assert sleeps == [0.25, 0.5]


def test_monotonic_total_deadline_prevents_another_retry_or_sleep():
    client = SyncClient([_response(503), _response(200)])
    times = iter([0.0, 0.0, 10.0])
    sleeps = []

    with pytest.raises(httpx.TimeoutException, match="deadline"):
        post_with_retries(
            client,
            "https://example.invalid",
            json={},
            headers={},
            max_retries=3,
            sleep=sleeps.append,
            jitter=lambda delay: delay,
            total_deadline_seconds=5.0,
            monotonic=lambda: next(times),
        )

    assert client.calls == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_async_retries_await_injected_sleep_and_jitter():
    client = AsyncClient([_response(429), _response(200)])
    sleeps = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    response = await apost_with_retries(
        client,
        "https://example.invalid",
        json={},
        headers={},
        max_retries=1,
        sleep=record_sleep,
        jitter=lambda delay: 0.125,
    )

    assert response.status_code == 200
    assert sleeps == [0.125]
