"""Logging middleware for flashlite."""

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from ..observability.callbacks import CallbackManager
from ..observability.logging import StructuredLogger
from ..observability.metrics import CostTracker
from ..types import CompletionRequest, CompletionResponse
from .base import Middleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(Middleware):
    """
    Middleware that logs requests and responses.

    Supports structured logging to files, callback-based logging,
    and cost tracking.

    Example:
        structured_logger = StructuredLogger(log_file="./logs/completions.jsonl")
        cost_tracker = CostTracker(budget_limit=10.0)

        middleware = LoggingMiddleware(
            logger=structured_logger,
            cost_tracker=cost_tracker,
            log_level="INFO",
        )
    """

    def __init__(
        self,
        structured_logger: StructuredLogger | None = None,
        cost_tracker: CostTracker | None = None,
        callbacks: CallbackManager | None = None,
        log_level: str = "INFO",
    ):
        """
        Initialize the logging middleware.

        Args:
            structured_logger: Structured logger for file logging
            cost_tracker: Cost tracker for budget monitoring
            callbacks: Callback manager for event hooks
            log_level: Minimum log level for standard logging
        """
        self._structured_logger = structured_logger
        self._cost_tracker = cost_tracker
        self._callbacks = callbacks
        self._log_level = getattr(logging, log_level.upper())

    async def __call__(
        self,
        request: CompletionRequest,
        next_handler: Callable[[CompletionRequest], Awaitable[CompletionResponse]],
    ) -> CompletionResponse:
        """Process request with logging."""
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        # Log request
        if self._structured_logger:
            self._structured_logger.log_request(request, request_id)

        if self._callbacks:
            await self._callbacks.emit_request(request, request_id)

        logger.log(
            self._log_level,
            f"[{request_id[:8]}] Starting request: model={request.model}",
        )

        try:
            # Call next handler
            response = await next_handler(request)

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Log response
            if self._structured_logger:
                self._structured_logger.log_response(
                    response, request_id, latency_ms, cached=False
                )

            if self._callbacks:
                await self._callbacks.emit_response(
                    response, request_id, latency_ms, cached=False
                )

            # Track cost
            if self._cost_tracker:
                cost = self._cost_tracker.track(response)
                logger.debug(
                    f"[{request_id[:8]}] Cost: ${cost:.6f}, "
                    f"Total: ${self._cost_tracker.total_cost:.4f}"
                )

            logger.log(
                self._log_level,
                f"[{request_id[:8]}] Completed: {latency_ms:.1f}ms, "
                f"tokens={response.usage.total_tokens if response.usage else 'N/A'}",
            )

            return response

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000

            if self._structured_logger:
                self._structured_logger.log_error(request_id, e, latency_ms)

            if self._callbacks:
                await self._callbacks.emit_error(e, request_id, latency_ms)

            logger.error(
                f"[{request_id[:8]}] Error after {latency_ms:.1f}ms: {e}",
            )
            raise


class CostTrackingMiddleware(Middleware):
    """
    Lightweight middleware that only tracks costs.

    Use this when you want cost tracking without full logging.

    Example:
        tracker = CostTracker(budget_limit=10.0)
        middleware = CostTrackingMiddleware(tracker)
    """

    def __init__(self, cost_tracker: CostTracker):
        """
        Initialize the cost tracking middleware.

        Args:
            cost_tracker: The cost tracker to use
        """
        self._cost_tracker = cost_tracker

    async def __call__(
        self,
        request: CompletionRequest,
        next_handler: Callable[[CompletionRequest], Awaitable[CompletionResponse]],
    ) -> CompletionResponse:
        """Process request with cost tracking."""
        response = await next_handler(request)
        self._cost_tracker.track(response)
        return response

    @property
    def tracker(self) -> CostTracker:
        """Get the cost tracker."""
        return self._cost_tracker
