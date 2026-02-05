"""Callback system for flashlite observability."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..types import CompletionRequest, CompletionResponse

# Callback type definitions
OnRequestCallback = Callable[[CompletionRequest, str], Awaitable[None] | None]
OnResponseCallback = Callable[
    [CompletionResponse, str, float, bool], Awaitable[None] | None
]
OnErrorCallback = Callable[[Exception, str, float], Awaitable[None] | None]


@dataclass
class CallbackManager:
    """
    Manages callbacks for request/response lifecycle events.

    Example:
        callbacks = CallbackManager()

        @callbacks.on_request
        async def log_request(request, request_id):
            print(f"Request {request_id}: {request.model}")

        @callbacks.on_response
        async def log_response(response, request_id, latency_ms, cached):
            print(f"Response {request_id}: {latency_ms}ms")

        # Or register directly
        callbacks.add_on_request(my_callback)
    """

    _on_request: list[OnRequestCallback] = field(default_factory=list)
    _on_response: list[OnResponseCallback] = field(default_factory=list)
    _on_error: list[OnErrorCallback] = field(default_factory=list)

    def add_on_request(self, callback: OnRequestCallback) -> None:
        """Add a callback to be called before each request."""
        self._on_request.append(callback)

    def add_on_response(self, callback: OnResponseCallback) -> None:
        """Add a callback to be called after each successful response."""
        self._on_response.append(callback)

    def add_on_error(self, callback: OnErrorCallback) -> None:
        """Add a callback to be called on request errors."""
        self._on_error.append(callback)

    def on_request(
        self, callback: OnRequestCallback
    ) -> OnRequestCallback:
        """Decorator to register a request callback."""
        self.add_on_request(callback)
        return callback

    def on_response(
        self, callback: OnResponseCallback
    ) -> OnResponseCallback:
        """Decorator to register a response callback."""
        self.add_on_response(callback)
        return callback

    def on_error(self, callback: OnErrorCallback) -> OnErrorCallback:
        """Decorator to register an error callback."""
        self.add_on_error(callback)
        return callback

    async def emit_request(
        self,
        request: CompletionRequest,
        request_id: str,
    ) -> None:
        """Emit a request event to all registered callbacks."""
        for callback in self._on_request:
            result = callback(request, request_id)
            if isinstance(result, Awaitable):
                await result

    async def emit_response(
        self,
        response: CompletionResponse,
        request_id: str,
        latency_ms: float,
        cached: bool = False,
    ) -> None:
        """Emit a response event to all registered callbacks."""
        for callback in self._on_response:
            result = callback(response, request_id, latency_ms, cached)
            if isinstance(result, Awaitable):
                await result

    async def emit_error(
        self,
        error: Exception,
        request_id: str,
        latency_ms: float,
    ) -> None:
        """Emit an error event to all registered callbacks."""
        for callback in self._on_error:
            result = callback(error, request_id, latency_ms)
            if isinstance(result, Awaitable):
                await result


def create_logging_callbacks(
    logger: Any,
    level: str = "INFO",
) -> CallbackManager:
    """
    Create a CallbackManager with standard logging callbacks.

    Args:
        logger: A logging.Logger instance
        level: Log level to use

    Returns:
        A configured CallbackManager
    """
    import logging as stdlib_logging

    log_level = getattr(stdlib_logging, level.upper())
    callbacks = CallbackManager()

    @callbacks.on_request
    def log_request(request: CompletionRequest, request_id: str) -> None:
        logger.log(
            log_level,
            f"[{request_id[:8]}] Request: model={request.model}",
        )

    @callbacks.on_response
    def log_response(
        response: CompletionResponse,
        request_id: str,
        latency_ms: float,
        cached: bool,
    ) -> None:
        cache_str = " (cached)" if cached else ""
        tokens = response.usage.total_tokens if response.usage else 0
        logger.log(
            log_level,
            f"[{request_id[:8]}] Response: {latency_ms:.1f}ms, {tokens} tokens{cache_str}",
        )

    @callbacks.on_error
    def log_error(error: Exception, request_id: str, latency_ms: float) -> None:
        logger.error(
            f"[{request_id[:8]}] Error after {latency_ms:.1f}ms: {error}",
        )

    return callbacks
