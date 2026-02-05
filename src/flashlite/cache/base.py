"""Base cache protocol and key generation for flashlite."""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..types import CompletionRequest, CompletionResponse


@dataclass
class CacheEntry:
    """A cached completion response with metadata."""

    response: CompletionResponse
    request_hash: str
    created_at: float
    ttl: float | None = None  # None means no expiration

    def is_expired(self, current_time: float) -> bool:
        """Check if this cache entry has expired."""
        if self.ttl is None:
            return False
        return current_time > self.created_at + self.ttl


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> CompletionResponse | None:
        """
        Retrieve a cached response.

        Args:
            key: The cache key

        Returns:
            The cached CompletionResponse, or None if not found/expired
        """
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        response: CompletionResponse,
        ttl: float | None = None,
    ) -> None:
        """
        Store a response in the cache.

        Args:
            key: The cache key
            response: The response to cache
            ttl: Time-to-live in seconds (None = no expiration)
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete a cached entry.

        Args:
            key: The cache key

        Returns:
            True if the key existed and was deleted
        """
        pass

    @abstractmethod
    async def clear(self) -> int:
        """
        Clear all cached entries.

        Returns:
            Number of entries cleared
        """
        pass

    @abstractmethod
    async def size(self) -> int:
        """
        Get the number of cached entries.

        Returns:
            Number of entries in the cache
        """
        pass

    async def close(self) -> None:
        """Close any resources held by the cache backend."""
        pass


def generate_cache_key(request: CompletionRequest) -> str:
    """
    Generate a deterministic cache key for a completion request.

    The key is based on:
    - Model name
    - Messages (serialized)
    - Temperature (if set)
    - Other deterministic parameters

    Note: Requests with temperature > 0 or reasoning enabled are typically
    not good candidates for caching since responses are non-deterministic.

    Args:
        request: The completion request

    Returns:
        A hex-encoded SHA-256 hash as the cache key
    """
    # Build the key components
    key_data: dict[str, Any] = {
        "model": request.model,
        "messages": [dict(m) for m in request.messages],
    }

    # Include parameters that affect output
    if request.temperature is not None:
        key_data["temperature"] = request.temperature
    if request.max_tokens is not None:
        key_data["max_tokens"] = request.max_tokens
    if request.max_completion_tokens is not None:
        key_data["max_completion_tokens"] = request.max_completion_tokens
    if request.top_p is not None:
        key_data["top_p"] = request.top_p
    if request.stop is not None:
        key_data["stop"] = request.stop
    if request.reasoning_effort is not None:
        key_data["reasoning_effort"] = request.reasoning_effort
    if request.thinking is not None:
        key_data["thinking"] = request.thinking

    # Include extra kwargs that affect output (excluding metadata-only ones and test helpers)
    exclude_from_key = {"timeout", "metadata", "tags", "user", "mock_response"}
    for k, v in request.extra_kwargs.items():
        if k not in exclude_from_key:
            key_data[f"extra.{k}"] = v

    # Serialize to JSON with sorted keys for determinism
    serialized = json.dumps(key_data, sort_keys=True, default=str)

    # Hash to get fixed-length key
    return hashlib.sha256(serialized.encode()).hexdigest()


def is_cacheable_request(request: CompletionRequest) -> tuple[bool, str | None]:
    """
    Check if a request is suitable for caching.

    Returns:
        Tuple of (is_cacheable, warning_message).
        If is_cacheable is True, warning_message may still contain a warning.
        If is_cacheable is False, warning_message explains why.
    """
    warnings: list[str] = []

    # Check for non-deterministic temperature
    if request.temperature is not None and request.temperature > 0:
        warnings.append(
            f"temperature={request.temperature} > 0 may produce different outputs"
        )

    # Check for reasoning models (non-deterministic by nature)
    if request.reasoning_effort is not None:
        warnings.append(
            f"reasoning_effort='{request.reasoning_effort}' - "
            "reasoning models may produce varying outputs"
        )

    if request.thinking is not None:
        warnings.append(
            "thinking enabled - extended thinking models may produce varying outputs"
        )

    # Check model name for known reasoning models
    model_lower = request.model.lower()
    reasoning_model_patterns = ["o1", "o3", "claude-3-5-sonnet", "claude-sonnet-4"]
    if any(pattern in model_lower for pattern in reasoning_model_patterns):
        if not any("reasoning" in w for w in warnings):
            warnings.append(
                f"model '{request.model}' appears to be a reasoning model - outputs may vary"
            )

    if warnings:
        return True, "; ".join(warnings)

    return True, None
