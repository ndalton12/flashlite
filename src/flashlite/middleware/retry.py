"""Retry middleware with exponential backoff."""

import asyncio
import logging
import random

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ..types import (
    CompletionError,
    CompletionRequest,
    CompletionResponse,
    RetryConfig,
)
from .base import CompletionHandler, Middleware

logger = logging.getLogger(__name__)


def _should_retry(exception: BaseException) -> bool:
    """Determine if an exception should trigger a retry."""
    if isinstance(exception, CompletionError):
        # Retry on specific status codes
        if exception.status_code in (429, 500, 502, 503, 504):
            return True
        # Don't retry on client errors (4xx except 429)
        if exception.status_code and 400 <= exception.status_code < 500:
            return False
    # Retry on connection errors, timeouts, etc.
    if isinstance(exception, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
        return True
    return False


class RetryMiddleware(Middleware):
    """
    Middleware that retries failed requests with exponential backoff.

    Uses tenacity for retry logic with:
    - Exponential backoff with jitter
    - Configurable max attempts and delays
    - Retries on transient errors (429, 5xx, connection errors)
    """

    def __init__(self, config: RetryConfig | None = None):
        """
        Initialize retry middleware.

        Args:
            config: Retry configuration. Uses defaults if not provided.
        """
        self.config = config or RetryConfig()

    async def __call__(
        self,
        request: CompletionRequest,
        next_handler: CompletionHandler,
    ) -> CompletionResponse:
        """Execute with retry logic."""
        attempt = 0

        try:
            async for attempt_state in AsyncRetrying(
                stop=stop_after_attempt(self.config.max_attempts),
                wait=wait_exponential_jitter(
                    initial=self.config.initial_delay,
                    max=self.config.max_delay,
                    exp_base=self.config.exponential_base,
                    jitter=self.config.initial_delay if self.config.jitter else 0,
                ),
                retry=retry_if_exception(_should_retry),
                reraise=True,
            ):
                with attempt_state:
                    attempt = attempt_state.retry_state.attempt_number
                    if attempt > 1:
                        logger.info(
                            f"Retry attempt {attempt}/{self.config.max_attempts} "
                            f"for model={request.model}"
                        )
                    return await next_handler(request)

        except RetryError as e:
            # Re-raise the last exception
            if e.last_attempt.failed:
                exc = e.last_attempt.exception()
                if exc is not None:
                    raise exc from e
            raise CompletionError("Retry attempts exhausted") from e

        # This should never be reached, but satisfies type checker
        raise CompletionError("Retry logic failed unexpectedly")


class SimpleRetryMiddleware(Middleware):
    """
    A simpler retry implementation without tenacity dependency.

    Useful for understanding the retry logic or when tenacity isn't available.
    """

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()

    async def __call__(
        self,
        request: CompletionRequest,
        next_handler: CompletionHandler,
    ) -> CompletionResponse:
        last_exception: Exception | None = None
        delay = self.config.initial_delay

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return await next_handler(request)
            except Exception as e:
                last_exception = e

                if not _should_retry(e):
                    raise

                if attempt == self.config.max_attempts:
                    raise

                # Calculate delay with optional jitter
                actual_delay = delay
                if self.config.jitter:
                    actual_delay = delay * (0.5 + random.random())

                logger.info(
                    f"Attempt {attempt}/{self.config.max_attempts} failed, "
                    f"retrying in {actual_delay:.2f}s: {e}"
                )

                await asyncio.sleep(actual_delay)

                # Exponential backoff for next attempt
                delay = min(delay * self.config.exponential_base, self.config.max_delay)

        # Should never reach here
        if last_exception:
            raise last_exception
        raise CompletionError("Retry logic failed unexpectedly")
