"""Tests for middleware functionality."""

import asyncio
import time

import pytest

from flashlite.middleware.base import Middleware, MiddlewareChain, PassthroughMiddleware
from flashlite.middleware.rate_limit import ConcurrencyLimiter, RateLimitMiddleware, TokenBucket
from flashlite.middleware.retry import RetryMiddleware, SimpleRetryMiddleware, _should_retry
from flashlite.types import (
    CompletionError,
    CompletionRequest,
    CompletionResponse,
    RateLimitConfig,
    RateLimitError,
    RetryConfig,
    UsageInfo,
)


# Helper to create test requests/responses
def make_request(model: str = "test-model") -> CompletionRequest:
    return CompletionRequest(
        model=model,
        messages=[{"role": "user", "content": "Hello"}],
    )


def make_response(content: str = "Hi there!") -> CompletionResponse:
    return CompletionResponse(
        content=content,
        model="test-model",
        finish_reason="stop",
        usage=UsageInfo(input_tokens=10, output_tokens=20, total_tokens=30),
    )


class TestMiddlewareChain:
    """Tests for middleware chain."""

    async def test_empty_chain_calls_handler(self) -> None:
        calls: list[str] = []

        async def handler(req: CompletionRequest) -> CompletionResponse:
            calls.append("handler")
            return make_response()

        chain = MiddlewareChain([], handler)
        result = await chain(make_request())

        assert calls == ["handler"]
        assert result.content == "Hi there!"

    async def test_middleware_order(self) -> None:
        calls: list[str] = []

        class TrackingMiddleware(Middleware):
            def __init__(self, name: str):
                self.name = name

            async def __call__(self, request, next_handler):
                calls.append(f"{self.name}_before")
                result = await next_handler(request)
                calls.append(f"{self.name}_after")
                return result

        async def handler(req: CompletionRequest) -> CompletionResponse:
            calls.append("handler")
            return make_response()

        chain = MiddlewareChain(
            [TrackingMiddleware("A"), TrackingMiddleware("B")],
            handler,
        )
        await chain(make_request())

        # A wraps B wraps handler
        assert calls == ["A_before", "B_before", "handler", "B_after", "A_after"]

    async def test_middleware_can_modify_request(self) -> None:
        class ModifyMiddleware(Middleware):
            async def __call__(self, request, next_handler):
                # Modify temperature
                request.temperature = 0.5
                return await next_handler(request)

        captured_request: CompletionRequest | None = None

        async def handler(req: CompletionRequest) -> CompletionResponse:
            nonlocal captured_request
            captured_request = req
            return make_response()

        chain = MiddlewareChain([ModifyMiddleware()], handler)
        await chain(make_request())

        assert captured_request is not None
        assert captured_request.temperature == 0.5

    async def test_passthrough_middleware(self) -> None:
        async def handler(req: CompletionRequest) -> CompletionResponse:
            return make_response("passed through")

        chain = MiddlewareChain([PassthroughMiddleware()], handler)
        result = await chain(make_request())
        assert result.content == "passed through"


class TestTokenBucket:
    """Tests for token bucket rate limiter."""

    async def test_immediate_acquire(self) -> None:
        bucket = TokenBucket(rate=10.0, capacity=10.0)

        # Should acquire immediately when bucket is full
        wait_time = await bucket.acquire(1.0)
        assert wait_time < 0.01

    async def test_acquire_depletes_tokens(self) -> None:
        bucket = TokenBucket(rate=10.0, capacity=5.0)

        # Acquire all tokens
        for _ in range(5):
            await bucket.acquire(1.0)

        # Next acquire should wait
        start = time.monotonic()
        await bucket.acquire(1.0)
        elapsed = time.monotonic() - start

        # Should have waited ~0.1s (1 token / 10 tokens per second)
        assert elapsed >= 0.05

    async def test_refill_over_time(self) -> None:
        bucket = TokenBucket(rate=100.0, capacity=10.0)

        # Deplete bucket
        await bucket.acquire(10.0)

        # Wait for some refill
        await asyncio.sleep(0.05)  # Should refill ~5 tokens at 100/s

        # Should have tokens available now
        assert bucket.available_tokens >= 4

    async def test_timeout(self) -> None:
        bucket = TokenBucket(rate=1.0, capacity=1.0)

        # Deplete bucket
        await bucket.acquire(1.0)

        # Should timeout trying to acquire more
        with pytest.raises(RateLimitError, match="timeout"):
            await bucket.acquire(1.0, timeout=0.01)


class TestRateLimitMiddleware:
    """Tests for rate limit middleware."""

    async def test_rpm_limiting(self) -> None:
        config = RateLimitConfig(requests_per_minute=6000)  # 100/s
        middleware = RateLimitMiddleware(config)

        call_times: list[float] = []

        async def handler(req: CompletionRequest) -> CompletionResponse:
            call_times.append(time.monotonic())
            return make_response()

        # Make several rapid requests
        for _ in range(5):
            await middleware(make_request(), handler)

        # All should complete (bucket has capacity for bursts)
        assert len(call_times) == 5

    async def test_no_limiting_when_disabled(self) -> None:
        config = RateLimitConfig()  # No limits set
        middleware = RateLimitMiddleware(config)

        async def handler(req: CompletionRequest) -> CompletionResponse:
            return make_response()

        # Should work without any delays
        result = await middleware(make_request(), handler)
        assert result.content == "Hi there!"


class TestConcurrencyLimiter:
    """Tests for concurrency limiter."""

    async def test_limits_concurrent_tasks(self) -> None:
        limiter = ConcurrencyLimiter(max_concurrency=2)
        active_count = 0
        max_active = 0

        async def task() -> None:
            nonlocal active_count, max_active
            async with limiter:
                active_count += 1
                max_active = max(max_active, active_count)
                await asyncio.sleep(0.05)
                active_count -= 1

        # Run 5 concurrent tasks with limit of 2
        await asyncio.gather(*[task() for _ in range(5)])

        assert max_active == 2  # Never more than 2 active

    def test_available_slots(self) -> None:
        limiter = ConcurrencyLimiter(max_concurrency=5)
        assert limiter.available_slots == 5


class TestRetryLogic:
    """Tests for retry predicate and middleware."""

    def test_should_retry_on_429(self) -> None:
        error = CompletionError("Rate limited", status_code=429)
        assert _should_retry(error) is True

    def test_should_retry_on_500(self) -> None:
        error = CompletionError("Server error", status_code=500)
        assert _should_retry(error) is True

    def test_should_not_retry_on_400(self) -> None:
        error = CompletionError("Bad request", status_code=400)
        assert _should_retry(error) is False

    def test_should_not_retry_on_401(self) -> None:
        error = CompletionError("Unauthorized", status_code=401)
        assert _should_retry(error) is False

    def test_should_retry_on_connection_error(self) -> None:
        assert _should_retry(ConnectionError("Connection failed")) is True

    def test_should_retry_on_timeout(self) -> None:
        assert _should_retry(TimeoutError("Timeout")) is True


class TestRetryMiddleware:
    """Tests for retry middleware."""

    async def test_success_on_first_try(self) -> None:
        config = RetryConfig(max_attempts=3)
        middleware = RetryMiddleware(config)
        attempts = 0

        async def handler(req: CompletionRequest) -> CompletionResponse:
            nonlocal attempts
            attempts += 1
            return make_response()

        result = await middleware(make_request(), handler)

        assert attempts == 1
        assert result.content == "Hi there!"

    async def test_retries_on_transient_error(self) -> None:
        config = RetryConfig(max_attempts=3, initial_delay=0.01, max_delay=0.1)
        middleware = RetryMiddleware(config)
        attempts = 0

        async def handler(req: CompletionRequest) -> CompletionResponse:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise CompletionError("Server error", status_code=500)
            return make_response("success on third try")

        result = await middleware(make_request(), handler)

        assert attempts == 3
        assert result.content == "success on third try"

    async def test_gives_up_after_max_attempts(self) -> None:
        config = RetryConfig(max_attempts=2, initial_delay=0.01)
        middleware = RetryMiddleware(config)
        attempts = 0

        async def handler(req: CompletionRequest) -> CompletionResponse:
            nonlocal attempts
            attempts += 1
            raise CompletionError("Always fails", status_code=500)

        with pytest.raises(CompletionError, match="Always fails"):
            await middleware(make_request(), handler)

        assert attempts == 2

    async def test_no_retry_on_client_error(self) -> None:
        config = RetryConfig(max_attempts=3)
        middleware = RetryMiddleware(config)
        attempts = 0

        async def handler(req: CompletionRequest) -> CompletionResponse:
            nonlocal attempts
            attempts += 1
            raise CompletionError("Bad request", status_code=400)

        with pytest.raises(CompletionError, match="Bad request"):
            await middleware(make_request(), handler)

        assert attempts == 1  # No retries on 400


class TestSimpleRetryMiddleware:
    """Tests for the simpler retry implementation."""

    async def test_retries_work(self) -> None:
        config = RetryConfig(max_attempts=3, initial_delay=0.01, jitter=False)
        middleware = SimpleRetryMiddleware(config)
        attempts = 0

        async def handler(req: CompletionRequest) -> CompletionResponse:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise CompletionError("Error", status_code=503)
            return make_response()

        result = await middleware(make_request(), handler)
        assert attempts == 2
        assert result.content == "Hi there!"
