"""Base middleware protocol and chain implementation."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from ..types import CompletionRequest, CompletionResponse

# Type alias for the completion handler
CompletionHandler = Callable[[CompletionRequest], Awaitable[CompletionResponse]]


class Middleware(ABC):
    """
    Abstract base class for middleware.

    Middleware can intercept requests before they're sent and responses
    after they're received. They form a chain where each middleware
    calls the next one (or short-circuits).
    """

    @abstractmethod
    async def __call__(
        self,
        request: CompletionRequest,
        next_handler: CompletionHandler,
    ) -> CompletionResponse:
        """
        Process a request.

        Args:
            request: The completion request
            next_handler: The next middleware or final handler to call

        Returns:
            The completion response
        """
        pass


class MiddlewareChain:
    """
    Chains multiple middleware together.

    Middleware are executed in order, with each one wrapping the next.
    The final handler (actual API call) is at the end of the chain.
    """

    def __init__(
        self,
        middleware: list[Middleware],
        final_handler: CompletionHandler,
    ):
        """
        Initialize the middleware chain.

        Args:
            middleware: List of middleware to apply (in order)
            final_handler: The final handler (actual completion call)
        """
        self._middleware = middleware
        self._final_handler = final_handler

    async def __call__(self, request: CompletionRequest) -> CompletionResponse:
        """Execute the middleware chain."""
        return await self._execute(request, 0)

    async def _execute(self, request: CompletionRequest, index: int) -> CompletionResponse:
        """Recursively execute middleware chain."""
        if index >= len(self._middleware):
            # No more middleware, call the final handler
            return await self._final_handler(request)

        # Get current middleware and create next handler
        current = self._middleware[index]

        async def next_handler(req: CompletionRequest) -> CompletionResponse:
            return await self._execute(req, index + 1)

        return await current(request, next_handler)


class PassthroughMiddleware(Middleware):
    """A middleware that does nothing - useful for testing."""

    async def __call__(
        self,
        request: CompletionRequest,
        next_handler: CompletionHandler,
    ) -> CompletionResponse:
        return await next_handler(request)
