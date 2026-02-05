"""Tests for caching functionality."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from flashlite.cache import (
    DiskCache,
    MemoryCache,
    generate_cache_key,
    is_cacheable_request,
)
from flashlite.types import CompletionRequest, CompletionResponse, UsageInfo


# Fixtures
@pytest.fixture
def sample_request() -> CompletionRequest:
    """Create a sample completion request."""
    return CompletionRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, world!"}],
        temperature=0,
    )


@pytest.fixture
def sample_response() -> CompletionResponse:
    """Create a sample completion response."""
    return CompletionResponse(
        content="Hello! How can I help you today?",
        model="gpt-4o",
        finish_reason="stop",
        usage=UsageInfo(input_tokens=10, output_tokens=20, total_tokens=30),
    )


@pytest.fixture
def memory_cache() -> MemoryCache:
    """Create a memory cache for testing."""
    return MemoryCache(max_size=100, default_ttl=3600)


@pytest.fixture
def temp_db_path() -> Path:
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        return Path(f.name)


# Cache Key Generation Tests
class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_generate_cache_key_deterministic(self, sample_request: CompletionRequest):
        """Cache key should be deterministic."""
        key1 = generate_cache_key(sample_request)
        key2 = generate_cache_key(sample_request)
        assert key1 == key2

    def test_different_messages_different_key(self):
        """Different messages should produce different keys."""
        req1 = CompletionRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        req2 = CompletionRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "World"}],
        )
        assert generate_cache_key(req1) != generate_cache_key(req2)

    def test_different_models_different_key(self):
        """Different models should produce different keys."""
        req1 = CompletionRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        req2 = CompletionRequest(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert generate_cache_key(req1) != generate_cache_key(req2)

    def test_different_temperature_different_key(self):
        """Different temperatures should produce different keys."""
        req1 = CompletionRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0,
        )
        req2 = CompletionRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
        )
        assert generate_cache_key(req1) != generate_cache_key(req2)

    def test_key_is_hex_hash(self, sample_request: CompletionRequest):
        """Cache key should be a hex-encoded hash."""
        key = generate_cache_key(sample_request)
        # SHA-256 produces 64 hex characters
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


class TestIsCacheableRequest:
    """Tests for cache suitability checking."""

    def test_temperature_zero_is_cacheable(self):
        """Request with temperature=0 should be cacheable without warnings."""
        request = CompletionRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0,
        )
        is_cacheable, warning = is_cacheable_request(request)
        assert is_cacheable
        assert warning is None

    def test_temperature_positive_warns(self):
        """Request with temperature > 0 should warn."""
        request = CompletionRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
        )
        is_cacheable, warning = is_cacheable_request(request)
        assert is_cacheable  # Still cacheable, just warns
        assert warning is not None
        assert "temperature" in warning

    def test_reasoning_effort_warns(self):
        """Request with reasoning_effort should warn."""
        request = CompletionRequest(
            model="o1",
            messages=[{"role": "user", "content": "Hello"}],
            reasoning_effort="high",
        )
        is_cacheable, warning = is_cacheable_request(request)
        assert is_cacheable
        assert warning is not None
        assert "reasoning" in warning.lower()

    def test_thinking_enabled_warns(self):
        """Request with thinking enabled should warn."""
        request = CompletionRequest(
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Hello"}],
            thinking={"type": "enabled", "budget_tokens": 10000},
        )
        is_cacheable, warning = is_cacheable_request(request)
        assert is_cacheable
        assert warning is not None
        assert "thinking" in warning.lower()


# Memory Cache Tests
class TestMemoryCache:
    """Tests for in-memory LRU cache."""

    async def test_set_and_get(
        self, memory_cache: MemoryCache, sample_response: CompletionResponse
    ):
        """Should be able to set and get values."""
        await memory_cache.set("key1", sample_response)
        result = await memory_cache.get("key1")
        assert result is not None
        assert result.content == sample_response.content

    async def test_get_missing_key_returns_none(self, memory_cache: MemoryCache):
        """Getting a missing key should return None."""
        result = await memory_cache.get("nonexistent")
        assert result is None

    async def test_delete(
        self, memory_cache: MemoryCache, sample_response: CompletionResponse
    ):
        """Should be able to delete entries."""
        await memory_cache.set("key1", sample_response)
        assert await memory_cache.delete("key1")
        assert await memory_cache.get("key1") is None

    async def test_delete_nonexistent_returns_false(self, memory_cache: MemoryCache):
        """Deleting nonexistent key should return False."""
        result = await memory_cache.delete("nonexistent")
        assert result is False

    async def test_clear(
        self, memory_cache: MemoryCache, sample_response: CompletionResponse
    ):
        """Should be able to clear all entries."""
        await memory_cache.set("key1", sample_response)
        await memory_cache.set("key2", sample_response)
        count = await memory_cache.clear()
        assert count == 2
        assert await memory_cache.size() == 0

    async def test_size(
        self, memory_cache: MemoryCache, sample_response: CompletionResponse
    ):
        """Should track size correctly."""
        assert await memory_cache.size() == 0
        await memory_cache.set("key1", sample_response)
        assert await memory_cache.size() == 1
        await memory_cache.set("key2", sample_response)
        assert await memory_cache.size() == 2

    async def test_lru_eviction(self, sample_response: CompletionResponse):
        """Should evict least recently used entries when at capacity."""
        cache = MemoryCache(max_size=2)
        await cache.set("key1", sample_response)
        await cache.set("key2", sample_response)
        await cache.set("key3", sample_response)  # Should evict key1

        assert await cache.get("key1") is None
        assert await cache.get("key2") is not None
        assert await cache.get("key3") is not None

    async def test_ttl_expiration(self, sample_response: CompletionResponse):
        """Entries should expire after TTL."""
        cache = MemoryCache(max_size=100, default_ttl=0.01)  # 10ms TTL
        await cache.set("key1", sample_response)
        await asyncio.sleep(0.02)  # Wait for expiration
        result = await cache.get("key1")
        assert result is None

    async def test_hit_miss_tracking(
        self, memory_cache: MemoryCache, sample_response: CompletionResponse
    ):
        """Should track hits and misses."""
        await memory_cache.set("key1", sample_response)

        await memory_cache.get("key1")  # Hit
        await memory_cache.get("key2")  # Miss

        assert memory_cache.hits == 1
        assert memory_cache.misses == 1
        assert memory_cache.hit_rate == 0.5

    def test_stats(self, memory_cache: MemoryCache):
        """Should return statistics."""
        stats = memory_cache.stats()
        assert "size" in stats
        assert "max_size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats


# Disk Cache Tests
class TestDiskCache:
    """Tests for SQLite disk cache."""

    async def test_set_and_get(
        self, temp_db_path: Path, sample_response: CompletionResponse
    ):
        """Should be able to set and get values."""
        cache = DiskCache(temp_db_path)
        try:
            await cache.set("key1", sample_response)
            result = await cache.get("key1")
            assert result is not None
            assert result.content == sample_response.content
            assert result.model == sample_response.model
        finally:
            await cache.close()

    async def test_persistence(
        self, temp_db_path: Path, sample_response: CompletionResponse
    ):
        """Data should persist across cache instances."""
        # First instance - write data
        cache1 = DiskCache(temp_db_path)
        await cache1.set("key1", sample_response)
        await cache1.close()

        # Second instance - read data
        cache2 = DiskCache(temp_db_path)
        result = await cache2.get("key1")
        assert result is not None
        assert result.content == sample_response.content
        await cache2.close()

    async def test_get_missing_key_returns_none(self, temp_db_path: Path):
        """Getting a missing key should return None."""
        cache = DiskCache(temp_db_path)
        try:
            result = await cache.get("nonexistent")
            assert result is None
        finally:
            await cache.close()

    async def test_delete(
        self, temp_db_path: Path, sample_response: CompletionResponse
    ):
        """Should be able to delete entries."""
        cache = DiskCache(temp_db_path)
        try:
            await cache.set("key1", sample_response)
            assert await cache.delete("key1")
            assert await cache.get("key1") is None
        finally:
            await cache.close()

    async def test_clear(
        self, temp_db_path: Path, sample_response: CompletionResponse
    ):
        """Should be able to clear all entries."""
        cache = DiskCache(temp_db_path)
        try:
            await cache.set("key1", sample_response)
            await cache.set("key2", sample_response)
            count = await cache.clear()
            assert count == 2
            assert await cache.size() == 0
        finally:
            await cache.close()

    async def test_ttl_expiration(
        self, temp_db_path: Path, sample_response: CompletionResponse
    ):
        """Entries should expire after TTL."""
        cache = DiskCache(temp_db_path, default_ttl=0.01)  # 10ms TTL
        try:
            await cache.set("key1", sample_response)
            await asyncio.sleep(0.02)  # Wait for expiration
            result = await cache.get("key1")
            assert result is None
        finally:
            await cache.close()

    async def test_cleanup_expired(
        self, temp_db_path: Path, sample_response: CompletionResponse
    ):
        """Should be able to clean up expired entries."""
        cache = DiskCache(temp_db_path, default_ttl=0.01)
        try:
            await cache.set("key1", sample_response)
            await asyncio.sleep(0.02)
            removed = await cache.cleanup_expired()
            assert removed == 1
        finally:
            await cache.close()

    async def test_usage_info_preserved(
        self, temp_db_path: Path, sample_response: CompletionResponse
    ):
        """Usage info should be preserved through serialization."""
        cache = DiskCache(temp_db_path)
        try:
            await cache.set("key1", sample_response)
            result = await cache.get("key1")
            assert result is not None
            assert result.usage is not None
            assert result.usage.input_tokens == 10
            assert result.usage.output_tokens == 20
            assert result.usage.total_tokens == 30
        finally:
            await cache.close()
