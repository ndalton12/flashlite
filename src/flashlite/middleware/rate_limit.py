"""Rate limiting middleware using token bucket algorithm."""

import asyncio
import logging
import time
from dataclasses import dataclass, field

from ..types import CompletionRequest, CompletionResponse, RateLimitConfig, RateLimitError
from .base import CompletionHandler, Middleware

logger = logging.getLogger(__name__)


@dataclass
class TokenBucket:
    """
    Token bucket rate limiter.

    Tokens are added at a constant rate up to a maximum capacity.
    Each request consumes one or more tokens. If not enough tokens
    are available, the request waits.
    """

    rate: float  # Tokens added per second
    capacity: float  # Maximum tokens in bucket
    tokens: float = field(init=False)
    last_update: float = field(init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        self.tokens = self.capacity
        self.last_update = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

    async def acquire(self, tokens: float = 1.0, timeout: float | None = None) -> float:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait (None = wait forever)

        Returns:
            Time waited in seconds

        Raises:
            RateLimitError: If timeout exceeded
        """
        start_time = time.monotonic()
        deadline = start_time + timeout if timeout else None

        async with self._lock:
            while True:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return time.monotonic() - start_time

                # Calculate wait time for enough tokens
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.rate

                # Check timeout
                if deadline:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RateLimitError(
                            f"Rate limit timeout after {timeout}s",
                            retry_after=wait_time,
                        )
                    wait_time = min(wait_time, remaining)

                # Release lock while waiting
                self._lock.release()
                try:
                    await asyncio.sleep(wait_time)
                finally:
                    await self._lock.acquire()

    @property
    def available_tokens(self) -> float:
        """Get current available tokens (without acquiring lock)."""
        elapsed = time.monotonic() - self.last_update
        return min(self.capacity, self.tokens + elapsed * self.rate)


class RateLimitMiddleware(Middleware):
    """
    Middleware that enforces rate limits using token bucket algorithm.

    Supports both requests-per-minute (RPM) and tokens-per-minute (TPM) limits.
    """

    def __init__(self, config: RateLimitConfig | None = None):
        """
        Initialize rate limit middleware.

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        self._rpm_bucket: TokenBucket | None = None
        self._tpm_bucket: TokenBucket | None = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazily initialize buckets."""
        if self._initialized:
            return

        if self.config.requests_per_minute:
            # Convert RPM to requests per second
            rate = self.config.requests_per_minute / 60.0
            # Capacity allows small bursts (10% of per-minute rate)
            capacity = max(1.0, self.config.requests_per_minute * 0.1)
            self._rpm_bucket = TokenBucket(rate=rate, capacity=capacity)
            logger.debug(
                f"Rate limiter initialized: {self.config.requests_per_minute} RPM "
                f"(rate={rate:.2f}/s, capacity={capacity:.1f})"
            )

        if self.config.tokens_per_minute:
            rate = self.config.tokens_per_minute / 60.0
            capacity = max(1000.0, self.config.tokens_per_minute * 0.1)
            self._tpm_bucket = TokenBucket(rate=rate, capacity=capacity)
            logger.debug(
                f"Token rate limiter initialized: {self.config.tokens_per_minute} TPM"
            )

        self._initialized = True

    async def __call__(
        self,
        request: CompletionRequest,
        next_handler: CompletionHandler,
    ) -> CompletionResponse:
        """Execute with rate limiting."""
        self._ensure_initialized()

        # Acquire RPM token before making request
        if self._rpm_bucket:
            wait_time = await self._rpm_bucket.acquire()
            if wait_time > 0.1:  # Only log significant waits
                logger.debug(f"Rate limit: waited {wait_time:.2f}s for RPM token")

        # Make the request
        response = await next_handler(request)

        # For TPM limiting, consume tokens based on actual usage
        # This is post-hoc - we can't know token count before the request
        if self._tpm_bucket and response.usage:
            total_tokens = response.usage.total_tokens
            if total_tokens > 0:
                # Don't block on TPM - just record the usage
                # This creates backpressure for subsequent requests
                await self._tpm_bucket.acquire(tokens=float(total_tokens))

        return response

    @property
    def rpm_available(self) -> float | None:
        """Get available RPM tokens."""
        if self._rpm_bucket:
            return self._rpm_bucket.available_tokens
        return None

    @property
    def tpm_available(self) -> float | None:
        """Get available TPM tokens."""
        if self._tpm_bucket:
            return self._tpm_bucket.available_tokens
        return None


class ConcurrencyLimiter:
    """
    Limits concurrent requests using a semaphore.

    This is separate from rate limiting - it controls how many
    requests can be in-flight simultaneously.
    """

    def __init__(self, max_concurrency: int):
        """
        Initialize concurrency limiter.

        Args:
            max_concurrency: Maximum concurrent requests
        """
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def __aenter__(self) -> "ConcurrencyLimiter":
        await self._semaphore.acquire()
        return self

    async def __aexit__(self, *args: object) -> None:
        self._semaphore.release()

    @property
    def available_slots(self) -> int:
        """Get number of available concurrency slots."""
        # Semaphore._value is the internal counter
        return self._semaphore._value  # noqa: SLF001
