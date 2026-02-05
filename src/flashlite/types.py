"""Shared types and protocols for flashlite."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict, TypeVar

from pydantic import BaseModel

# Type aliases for messages
Role = Literal["system", "user", "assistant", "tool"]


class MessageDict(TypedDict, total=False):
    """A chat message in dictionary form."""

    role: Role
    content: str
    name: str
    tool_calls: list[dict[str, Any]]
    tool_call_id: str


Message = MessageDict | dict[str, Any]
Messages = Sequence[Message]


# Convenience function for creating thinking config
def thinking_enabled(budget_tokens: int) -> "ThinkingConfig":
    """
    Create an Anthropic extended thinking configuration.

    Args:
        budget_tokens: Maximum tokens for Claude's internal reasoning.
                      Minimum is 1024. Larger budgets (16k+) recommended for complex tasks.

    Returns:
        ThinkingConfig dict to pass to complete()

    Example:
        await client.complete(
            model="claude-sonnet-4-5-20250929",
            messages="Solve this complex problem...",
            thinking=thinking_enabled(10000),
        )
    """
    return {"type": "enabled", "budget_tokens": budget_tokens}


class ThinkingConfig(TypedDict, total=False):
    """Configuration for Anthropic extended thinking."""

    type: Literal["enabled", "disabled"]
    budget_tokens: int


@dataclass
class CompletionRequest:
    """A request to complete a chat conversation."""

    model: str
    messages: Messages = {}
    template: str | None = None
    variables: dict[str, Any] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    # OpenAI reasoning model parameters (o1, o3)
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    # Anthropic extended thinking parameters (Claude)
    thinking: ThinkingConfig | None = None
    # Additional kwargs passed through to litellm
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_litellm_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs dict for litellm.completion()."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(self.messages),
        }

        # Add optional parameters if set
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_completion_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.stop is not None:
            kwargs["stop"] = self.stop
        # OpenAI reasoning effort (o1, o3 models)
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        # Anthropic extended thinking (Claude models)
        if self.thinking is not None:
            kwargs["thinking"] = self.thinking

        # Merge extra kwargs
        kwargs.update(self.extra_kwargs)

        return kwargs


@dataclass
class CompletionResponse:
    """A response from a completion request."""

    content: str
    model: str
    finish_reason: str | None = None
    usage: "UsageInfo | None" = None
    raw_response: Any = None

    @property
    def input_tokens(self) -> int:
        """Get input token count."""
        return self.usage.input_tokens if self.usage else 0

    @property
    def output_tokens(self) -> int:
        """Get output token count."""
        return self.usage.output_tokens if self.usage else 0


@dataclass
class UsageInfo:
    """Token usage information."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_litellm(cls, usage: dict[str, Any] | None) -> "UsageInfo":
        """Create from litellm usage dict."""
        if not usage:
            return cls()
        return cls(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )


# Response model type variable
ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


# Middleware protocol
class MiddlewareProtocol(Protocol):
    """Protocol for middleware that wraps completion calls."""

    async def __call__(
        self,
        request: CompletionRequest,
        next_handler: Callable[[CompletionRequest], Awaitable[CompletionResponse]],
    ) -> CompletionResponse:
        """Process a request, optionally delegating to the next handler."""
        ...


# Configuration types
@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    # HTTP status codes to retry on
    retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: float | None = None
    tokens_per_minute: float | None = None
    # If True, read limits from API response headers
    auto_detect: bool = False


# Exceptions
class FlashliteError(Exception):
    """Base exception for flashlite errors."""

    pass


class CompletionError(FlashliteError):
    """Error during completion request."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response: Any = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class RateLimitError(FlashliteError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ValidationError(FlashliteError):
    """Response validation failed."""

    def __init__(self, message: str, errors: list[Any] | None = None):
        super().__init__(message)
        self.errors = errors or []


class TemplateError(FlashliteError):
    """Template rendering error."""

    pass


class ConfigError(FlashliteError):
    """Configuration error."""

    pass
