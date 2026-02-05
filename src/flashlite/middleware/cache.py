"""Cache middleware for flashlite."""

import logging
from collections.abc import Awaitable, Callable

from ..cache.base import CacheBackend, generate_cache_key, is_cacheable_request
from ..types import CompletionRequest, CompletionResponse
from .base import Middleware

logger = logging.getLogger(__name__)


class CacheMiddleware(Middleware):
    """
    Middleware that caches completion responses.

    Caches responses based on a hash of the request parameters.
    Emits warnings when caching is used with non-deterministic settings
    (temperature > 0 or reasoning models).

    Example:
        cache = MemoryCache(max_size=1000)
        middleware = CacheMiddleware(cache)
    """

    def __init__(
        self,
        backend: CacheBackend,
        ttl: float | None = None,
        warn_non_deterministic: bool = True,
    ):
        """
        Initialize the cache middleware.

        Args:
            backend: The cache backend to use
            ttl: Default TTL for cached entries (seconds)
            warn_non_deterministic: Whether to warn about non-deterministic caching
        """
        self._backend = backend
        self._ttl = ttl
        self._warn_non_deterministic = warn_non_deterministic
        self._warned_keys: set[str] = set()  # Track warnings to avoid spam

    async def __call__(
        self,
        request: CompletionRequest,
        next_handler: Callable[[CompletionRequest], Awaitable[CompletionResponse]],
    ) -> CompletionResponse:
        """Process request with caching."""
        # Generate cache key
        cache_key = generate_cache_key(request)

        # Check if request is suitable for caching and emit warnings
        if self._warn_non_deterministic:
            _, warning = is_cacheable_request(request)
            if warning and cache_key not in self._warned_keys:
                logger.warning(
                    f"Caching enabled but request may be non-deterministic: {warning}. "
                    "Consider disabling cache for this request with force_refresh=True, "
                    "or set temperature=0 for deterministic outputs."
                )
                self._warned_keys.add(cache_key)

        # Try to get from cache
        cached_response = await self._backend.get(cache_key)
        if cached_response is not None:
            logger.debug(f"Cache hit for key {cache_key[:16]}...")
            return cached_response

        # Cache miss - call the next handler
        logger.debug(f"Cache miss for key {cache_key[:16]}...")
        response = await next_handler(request)

        # Store in cache
        await self._backend.set(cache_key, response, self._ttl)

        return response

    @property
    def backend(self) -> CacheBackend:
        """Get the cache backend."""
        return self._backend


class CacheConfig:
    """
    Configuration for caching behavior.

    Note: Caching is disabled by default. When enabled, warnings are emitted
    for non-deterministic requests (temperature > 0 or reasoning models).
    """

    def __init__(
        self,
        enabled: bool = False,
        backend: CacheBackend | None = None,
        ttl: float | None = None,
        warn_non_deterministic: bool = True,
    ):
        """
        Initialize cache configuration.

        Args:
            enabled: Whether caching is enabled (default: False)
            backend: The cache backend to use
            ttl: Default TTL for cached entries (seconds)
            warn_non_deterministic: Whether to warn about non-deterministic caching
        """
        self.enabled = enabled
        self.backend = backend
        self.ttl = ttl
        self.warn_non_deterministic = warn_non_deterministic

        # Emit info message about caching status
        if not enabled:
            logger.info(
                "Caching is disabled by default. To enable, pass "
                "cache=CacheConfig(enabled=True, backend=...) or "
                "cache=MemoryCache(...) to the Flashlite client."
            )
