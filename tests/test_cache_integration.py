"""Integration tests for caching with the Flashlite client."""

from pathlib import Path

from flashlite import DiskCache, Flashlite, MemoryCache


class TestClientWithMemoryCache:
    """Tests for Flashlite client with memory caching."""

    async def test_cache_hit(self):
        """Second request should be served from cache."""
        cache = MemoryCache(max_size=100)
        client = Flashlite(
            default_model="gpt-4o",
            cache=cache,
        )

        # First request (using litellm mock_response)
        response1 = await client.complete(
            messages="Hello",
            temperature=0,
            mock_response="First response",
        )
        # Second identical request - should hit cache
        response2 = await client.complete(
            messages="Hello",
            temperature=0,
            mock_response="Second response",  # This won't be used due to cache
        )

        # Both should return the cached response
        assert response1.content == "First response"
        assert response2.content == "First response"  # Cached!

        # Cache should have the entry
        assert await cache.size() == 1
        assert cache.hits == 1
        assert cache.misses == 1  # First request was a miss

    async def test_cache_miss_different_messages(self):
        """Different messages should not hit cache."""
        cache = MemoryCache(max_size=100)
        client = Flashlite(
            default_model="gpt-4o",
            cache=cache,
        )

        await client.complete(
            messages="Hello",
            temperature=0,
            mock_response="Response 1",
        )
        await client.complete(
            messages="World",
            temperature=0,
            mock_response="Response 2",
        )

        # Both should be cache misses
        assert await cache.size() == 2
        assert cache.misses == 2

    async def test_warning_on_non_zero_temperature(self, caplog):
        """Should warn when caching with temperature > 0."""
        import logging

        caplog.set_level(logging.WARNING)

        cache = MemoryCache(max_size=100)
        client = Flashlite(
            default_model="gpt-4o",
            cache=cache,
        )

        await client.complete(
            messages="Hello",
            temperature=0.7,  # Non-zero temperature
            mock_response="test",
        )

        # Check that a warning was logged
        assert any("non-deterministic" in record.message for record in caplog.records)

    async def test_warning_on_reasoning_model(self, caplog):
        """Should warn when caching with reasoning model."""
        import logging

        caplog.set_level(logging.WARNING)

        cache = MemoryCache(max_size=100)
        client = Flashlite(
            default_model="o1",
            cache=cache,
        )

        await client.complete(
            messages="Hello",
            reasoning_effort="high",
            mock_response="test",
        )

        assert any("non-deterministic" in record.message for record in caplog.records)


class TestClientWithDiskCache:
    """Tests for Flashlite client with disk caching."""

    async def test_disk_cache_persistence(self, tmp_path: Path):
        """Cache should persist across client instances."""
        cache_path = tmp_path / "test_cache.db"

        # First client - write to cache
        cache1 = DiskCache(cache_path)
        client1 = Flashlite(
            default_model="gpt-4o",
            cache=cache1,
        )

        await client1.complete(
            messages="Hello",
            temperature=0,
            mock_response="Cached response",
        )
        await cache1.close()

        # Second client - should read from cache
        cache2 = DiskCache(cache_path)
        client2 = Flashlite(
            default_model="gpt-4o",
            cache=cache2,
        )

        response = await client2.complete(
            messages="Hello",
            temperature=0,
            mock_response="New response",  # Won't be used due to cache
        )

        # Should have retrieved cached response
        assert response.content == "Cached response"
        assert cache2.hits == 1
        await cache2.close()


class TestClientWithCostTracking:
    """Tests for Flashlite client with cost tracking."""

    async def test_cost_tracking_enabled(self):
        """Should track costs when enabled."""
        client = Flashlite(
            default_model="gpt-4o",
            track_costs=True,
        )

        await client.complete(messages="Hello", mock_response="Hi")
        await client.complete(messages="World", mock_response="Hello")

        # litellm mock doesn't set token usage, so we just verify the structure works
        report = client.get_cost_report()
        assert report is not None
        assert "total_cost_usd" in report
        assert report["total_requests"] == 2


class TestClientCacheProperties:
    """Tests for client cache-related properties."""

    async def test_clear_cache(self):
        """Should be able to clear cache via client."""
        cache = MemoryCache(max_size=100)
        client = Flashlite(
            default_model="gpt-4o",
            cache=cache,
        )

        await client.complete(
            messages="Hello",
            temperature=0,
            mock_response="test",
        )

        assert await cache.size() == 1

        cleared = await client.clear_cache()
        assert cleared == 1
        assert await cache.size() == 0

    async def test_cache_stats(self):
        """Should be able to get cache stats via client."""
        cache = MemoryCache(max_size=100)
        client = Flashlite(
            default_model="gpt-4o",
            cache=cache,
        )

        await client.complete(messages="Hello", temperature=0, mock_response="test")
        await client.complete(
            messages="Hello", temperature=0, mock_response="test2"
        )  # Cache hit

        stats = await client.cache_stats()
        assert stats is not None
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_no_cache_returns_none(self):
        """Should return None when cache not configured."""
        client = Flashlite(default_model="gpt-4o")
        assert client.cache is None

    def test_no_cost_tracker_returns_zero(self):
        """Should return 0 when cost tracking not enabled."""
        client = Flashlite(default_model="gpt-4o")
        assert client.total_cost == 0.0
        assert client.total_tokens == 0
        assert client.get_cost_report() is None
