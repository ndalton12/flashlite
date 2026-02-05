"""In-memory LRU cache with TTL support."""

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass

from ..types import CompletionResponse
from .base import CacheBackend


@dataclass
class MemoryCacheEntry:
    """An entry in the memory cache."""

    response: CompletionResponse
    created_at: float
    ttl: float | None


class MemoryCache(CacheBackend):
    """
    In-memory LRU cache with optional TTL.

    This cache uses an OrderedDict to maintain LRU ordering.
    When the cache exceeds max_size, the least recently used
    entries are evicted.

    Example:
        cache = MemoryCache(max_size=1000, default_ttl=3600)

        # Store a response
        await cache.set(key, response)

        # Retrieve (returns None if expired or not found)
        cached = await cache.get(key)
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float | None = None,
    ):
        """
        Initialize the memory cache.

        Args:
            max_size: Maximum number of entries to store
            default_ttl: Default time-to-live in seconds (None = no expiration)
        """
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, MemoryCacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> CompletionResponse | None:
        """Retrieve a cached response."""
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            # Check expiration
            if entry.ttl is not None:
                if time.time() > entry.created_at + entry.ttl:
                    # Expired - remove and return None
                    del self._cache[key]
                    self._misses += 1
                    return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.response

    async def set(
        self,
        key: str,
        response: CompletionResponse,
        ttl: float | None = None,
    ) -> None:
        """Store a response in the cache."""
        async with self._lock:
            # Use provided TTL or default
            effective_ttl = ttl if ttl is not None else self._default_ttl

            entry = MemoryCacheEntry(
                response=response,
                created_at=time.time(),
                ttl=effective_ttl,
            )

            # If key exists, update and move to end
            if key in self._cache:
                self._cache[key] = entry
                self._cache.move_to_end(key)
            else:
                # Add new entry
                self._cache[key] = entry

                # Evict LRU entries if over capacity
                while len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)

    async def delete(self, key: str) -> bool:
        """Delete a cached entry."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> int:
        """Clear all cached entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            return count

    async def size(self) -> int:
        """Get the number of cached entries."""
        async with self._lock:
            return len(self._cache)

    @property
    def hits(self) -> int:
        """Number of cache hits."""
        return self._hits

    @property
    def misses(self) -> int:
        """Number of cache misses."""
        return self._misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def stats(self) -> dict[str, int | float]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }
