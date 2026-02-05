"""SQLite-based disk cache for persistent caching."""

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from ..types import CompletionResponse, UsageInfo
from .base import CacheBackend


class DiskCache(CacheBackend):
    """
    SQLite-based disk cache for persistent caching.

    This cache stores responses in a SQLite database, providing
    persistence across process restarts.

    Example:
        cache = DiskCache("./cache/completions.db", default_ttl=86400)

        # Store a response
        await cache.set(key, response)

        # Retrieve (returns None if expired or not found)
        cached = await cache.get(key)

        # Close when done
        await cache.close()
    """

    def __init__(
        self,
        path: str | Path,
        default_ttl: float | None = None,
        auto_vacuum: bool = True,
    ):
        """
        Initialize the disk cache.

        Args:
            path: Path to SQLite database file (will be created if doesn't exist)
            default_ttl: Default time-to-live in seconds (None = no expiration)
            auto_vacuum: Whether to run VACUUM on startup to reclaim space
        """
        self._path = Path(path)
        self._default_ttl = default_ttl
        self._auto_vacuum = auto_vacuum
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # Create tables
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                response_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL,
                model TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER
            )
        """)

        # Create index for expiration queries
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires_at ON cache(expires_at)
        """)

        self._conn.commit()

        # Optional vacuum to reclaim space
        if self._auto_vacuum:
            try:
                self._conn.execute("VACUUM")
            except sqlite3.OperationalError:
                # VACUUM can fail if database is locked
                pass

    def _serialize_response(self, response: CompletionResponse) -> str:
        """Serialize a CompletionResponse to JSON."""
        data: dict[str, Any] = {
            "content": response.content,
            "model": response.model,
            "finish_reason": response.finish_reason,
        }

        if response.usage:
            data["usage"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return json.dumps(data)

    def _deserialize_response(self, json_str: str) -> CompletionResponse:
        """Deserialize a CompletionResponse from JSON."""
        data = json.loads(json_str)

        usage = None
        if "usage" in data:
            usage = UsageInfo(
                input_tokens=data["usage"]["input_tokens"],
                output_tokens=data["usage"]["output_tokens"],
                total_tokens=data["usage"]["total_tokens"],
            )

        return CompletionResponse(
            content=data["content"],
            model=data["model"],
            finish_reason=data.get("finish_reason"),
            usage=usage,
        )

    async def get(self, key: str) -> CompletionResponse | None:
        """Retrieve a cached response."""
        async with self._lock:
            if self._conn is None:
                raise RuntimeError("Cache has been closed")

            now = time.time()

            # Query for non-expired entry
            cursor = self._conn.execute(
                """
                SELECT response_json FROM cache
                WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)
                """,
                (key, now),
            )
            row = cursor.fetchone()

            if row is None:
                self._misses += 1
                return None

            self._hits += 1
            return self._deserialize_response(row["response_json"])

    async def set(
        self,
        key: str,
        response: CompletionResponse,
        ttl: float | None = None,
    ) -> None:
        """Store a response in the cache."""
        async with self._lock:
            if self._conn is None:
                raise RuntimeError("Cache has been closed")

            effective_ttl = ttl if ttl is not None else self._default_ttl
            now = time.time()
            expires_at = now + effective_ttl if effective_ttl else None

            response_json = self._serialize_response(response)

            self._conn.execute(
                """
                INSERT OR REPLACE INTO cache
                (key, response_json, created_at, expires_at, model, input_tokens, output_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    response_json,
                    now,
                    expires_at,
                    response.model,
                    response.usage.input_tokens if response.usage else None,
                    response.usage.output_tokens if response.usage else None,
                ),
            )
            self._conn.commit()

    async def delete(self, key: str) -> bool:
        """Delete a cached entry."""
        async with self._lock:
            if self._conn is None:
                raise RuntimeError("Cache has been closed")

            cursor = self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._conn.commit()
            return cursor.rowcount > 0

    async def clear(self) -> int:
        """Clear all cached entries."""
        async with self._lock:
            if self._conn is None:
                raise RuntimeError("Cache has been closed")

            cursor = self._conn.execute("SELECT COUNT(*) as count FROM cache")
            count = cursor.fetchone()["count"]

            self._conn.execute("DELETE FROM cache")
            self._conn.commit()

            self._hits = 0
            self._misses = 0

            return count

    async def size(self) -> int:
        """Get the number of cached entries (excluding expired)."""
        async with self._lock:
            if self._conn is None:
                raise RuntimeError("Cache has been closed")

            now = time.time()
            cursor = self._conn.execute(
                """
                SELECT COUNT(*) as count FROM cache
                WHERE expires_at IS NULL OR expires_at > ?
                """,
                (now,),
            )
            return cursor.fetchone()["count"]

    async def cleanup_expired(self) -> int:
        """
        Remove expired entries from the cache.

        Returns:
            Number of entries removed
        """
        async with self._lock:
            if self._conn is None:
                raise RuntimeError("Cache has been closed")

            now = time.time()
            cursor = self._conn.execute(
                "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            self._conn.commit()
            return cursor.rowcount

    async def close(self) -> None:
        """Close the database connection."""
        async with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    @property
    def hits(self) -> int:
        """Number of cache hits (this session only)."""
        return self._hits

    @property
    def misses(self) -> int:
        """Number of cache misses (this session only)."""
        return self._misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0) for this session."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "path": str(self._path),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }
