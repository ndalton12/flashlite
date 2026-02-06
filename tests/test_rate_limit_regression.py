"""Regression tests for rate-limit edge cases.

These tests are marked *slow* and are **not** collected by default.
Run them explicitly with:

    pytest -m slow tests/test_rate_limit_regression.py
"""

import asyncio
import logging

import pytest

from flashlite.middleware.rate_limit import RateLimitMiddleware, TokenBucket
from flashlite.types import (
    CompletionRequest,
    CompletionResponse,
    RateLimitConfig,
    UsageInfo,
)


def _make_request() -> CompletionRequest:
    return CompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )


def _make_response(total_tokens: int = 30) -> CompletionResponse:
    return CompletionResponse(
        content="Hi there!",
        model="test-model",
        finish_reason="stop",
        usage=UsageInfo(
            input_tokens=total_tokens // 3,
            output_tokens=total_tokens - total_tokens // 3,
            total_tokens=total_tokens,
        ),
    )


@pytest.mark.slow
class TestTokenBucketCapacityClamp:
    """Regression: acquire() must not infinite-loop when tokens > capacity."""

    async def test_acquire_exceeding_capacity_does_not_hang(self) -> None:
        """If requested tokens > capacity, acquire() should still return."""
        bucket = TokenBucket(rate=100.0, capacity=50.0)

        # Request more than the bucket can ever hold.
        wait_time = await asyncio.wait_for(
            bucket.acquire(tokens=200.0),
            timeout=2.0,  # generous — would hang forever without fix
        )

        # Should return almost immediately (bucket starts full).
        assert wait_time < 0.1

    async def test_acquire_logs_warning_on_clamp(self, caplog: pytest.LogCaptureFixture) -> None:
        """A warning is emitted when tokens are clamped to capacity."""
        bucket = TokenBucket(rate=100.0, capacity=50.0)

        with caplog.at_level(logging.WARNING):
            await bucket.acquire(tokens=200.0)

        assert any("exceeds bucket capacity" in msg for msg in caplog.messages)

    async def test_acquire_at_capacity_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warning when tokens == capacity (normal operation)."""
        bucket = TokenBucket(rate=100.0, capacity=50.0)

        with caplog.at_level(logging.WARNING):
            await bucket.acquire(tokens=50.0)

        assert not any("exceeds bucket capacity" in msg for msg in caplog.messages)


@pytest.mark.slow
class TestRateLimitMiddlewareTPMClamp:
    """Regression: middleware must not hang when response tokens exceed TPM bucket capacity."""

    async def test_large_response_does_not_hang(self) -> None:
        """A response whose total_tokens > TPM bucket capacity must not deadlock."""
        config = RateLimitConfig(tokens_per_minute=20_000)
        middleware = RateLimitMiddleware(config)

        # capacity = max(1000, 20000 * 0.1) = 2000
        # A response consuming 5000 tokens would previously hang forever.
        async def handler(req: CompletionRequest) -> CompletionResponse:
            return _make_response(total_tokens=5000)

        result = await asyncio.wait_for(
            middleware(_make_request(), handler),
            timeout=5.0,
        )

        assert result.content == "Hi there!"
        assert result.usage is not None
        assert result.usage.total_tokens == 5000

    async def test_backpressure_still_applied(self) -> None:
        """Even with clamping, subsequent requests should still feel backpressure."""
        config = RateLimitConfig(tokens_per_minute=6_000)
        middleware = RateLimitMiddleware(config)

        # capacity = max(1000, 600) = 1000
        # rate = 100 tokens/s
        async def handler(req: CompletionRequest) -> CompletionResponse:
            return _make_response(total_tokens=2000)

        # First request drains the bucket (clamped to capacity=1000).
        await asyncio.wait_for(
            middleware(_make_request(), handler),
            timeout=5.0,
        )

        # Second request should need to wait for refill.
        start = asyncio.get_event_loop().time()
        await asyncio.wait_for(
            middleware(_make_request(), handler),
            timeout=30.0,
        )
        elapsed = asyncio.get_event_loop().time() - start

        # Bucket needs to refill ~1000 tokens at 100/s → ~10s.
        # Allow some tolerance; the key assertion is that we *did* wait.
        assert elapsed >= 1.0, "Expected backpressure delay but request returned instantly"
