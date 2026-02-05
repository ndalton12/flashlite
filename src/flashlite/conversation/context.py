"""Context window management for conversations."""

import logging
from dataclasses import dataclass
from typing import Any

from ..types import Messages

logger = logging.getLogger(__name__)


# Approximate token counts per character for different models
# These are rough estimates - actual tokenization varies
CHARS_PER_TOKEN_ESTIMATE = 4


@dataclass
class ContextLimits:
    """Context window limits for a model."""

    max_tokens: int
    recommended_max: int | None = None  # Leave room for response

    @classmethod
    def for_model(cls, model: str) -> "ContextLimits":
        """Get context limits for a model (approximate)."""
        model_lower = model.lower()

        # GPT-4 variants
        if "gpt-4o" in model_lower:
            return cls(max_tokens=128_000, recommended_max=120_000)
        if "gpt-4-turbo" in model_lower or "gpt-4-1106" in model_lower:
            return cls(max_tokens=128_000, recommended_max=120_000)
        if "gpt-4-32k" in model_lower:
            return cls(max_tokens=32_768, recommended_max=30_000)
        if "gpt-4" in model_lower:
            return cls(max_tokens=8_192, recommended_max=7_000)

        # GPT-3.5 variants
        if "gpt-3.5-turbo-16k" in model_lower:
            return cls(max_tokens=16_384, recommended_max=15_000)
        if "gpt-3.5" in model_lower:
            return cls(max_tokens=16_384, recommended_max=15_000)

        # Claude variants
        if "claude-3" in model_lower or "claude-sonnet-4" in model_lower:
            return cls(max_tokens=200_000, recommended_max=190_000)
        if "claude-2" in model_lower:
            return cls(max_tokens=100_000, recommended_max=95_000)
        if "claude" in model_lower:
            return cls(max_tokens=100_000, recommended_max=95_000)

        # Gemini
        if "gemini-1.5" in model_lower:
            return cls(max_tokens=1_000_000, recommended_max=900_000)
        if "gemini" in model_lower:
            return cls(max_tokens=32_768, recommended_max=30_000)

        # Mistral
        if "mistral-large" in model_lower:
            return cls(max_tokens=128_000, recommended_max=120_000)
        if "mistral" in model_lower:
            return cls(max_tokens=32_768, recommended_max=30_000)

        # Default conservative estimate
        return cls(max_tokens=8_192, recommended_max=7_000)


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in text.

    This is a rough approximation. For accurate counts, use tiktoken
    or the provider's tokenizer.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // CHARS_PER_TOKEN_ESTIMATE + 1


def estimate_messages_tokens(messages: Messages) -> int:
    """
    Estimate total tokens in a messages list.

    Args:
        messages: List of messages

    Returns:
        Estimated token count
    """
    total = 0
    for msg in messages:
        # Add overhead for message structure
        total += 4  # Approximate overhead per message

        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            # Handle multi-part content (e.g., images)
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total += estimate_tokens(part["text"])
                else:
                    total += 100  # Rough estimate for non-text content

    return total


def check_context_fit(
    messages: Messages,
    model: str,
    max_response_tokens: int = 4096,
) -> tuple[bool, dict[str, Any]]:
    """
    Check if messages fit within the model's context window.

    Args:
        messages: The messages to check
        model: The model name
        max_response_tokens: Expected max tokens in response

    Returns:
        Tuple of (fits, info_dict) where info_dict contains:
        - estimated_tokens: Estimated input tokens
        - max_tokens: Model's max context
        - remaining: Tokens remaining for response
        - warning: Optional warning message
    """
    limits = ContextLimits.for_model(model)
    estimated = estimate_messages_tokens(messages)
    remaining = limits.max_tokens - estimated - max_response_tokens

    info: dict[str, Any] = {
        "estimated_tokens": estimated,
        "max_tokens": limits.max_tokens,
        "remaining": remaining,
    }

    if remaining < 0:
        info["warning"] = (
            f"Messages ({estimated} tokens) + response ({max_response_tokens}) "
            f"exceed context limit ({limits.max_tokens})"
        )
        return False, info

    if limits.recommended_max and estimated > limits.recommended_max:
        info["warning"] = (
            f"Messages ({estimated} tokens) exceed recommended max "
            f"({limits.recommended_max}). Consider truncating."
        )

    return True, info


def truncate_messages(
    messages: Messages,
    max_tokens: int,
    strategy: str = "oldest",
    keep_system: bool = True,
) -> Messages:
    """
    Truncate messages to fit within a token budget.

    Args:
        messages: The messages to truncate
        max_tokens: Maximum tokens to keep
        strategy: Truncation strategy - "oldest" removes oldest messages first
        keep_system: Whether to always keep system messages

    Returns:
        Truncated messages list
    """
    if strategy != "oldest":
        raise ValueError(f"Unknown truncation strategy: {strategy}")

    messages_list = list(messages)
    current_tokens = estimate_messages_tokens(messages_list)

    if current_tokens <= max_tokens:
        return messages_list

    # Separate system messages if we're keeping them
    system_messages = []
    other_messages = []

    for msg in messages_list:
        if keep_system and msg.get("role") == "system":
            system_messages.append(msg)
        else:
            other_messages.append(msg)

    system_tokens = estimate_messages_tokens(system_messages)
    available_tokens = max_tokens - system_tokens

    # Remove oldest messages until we fit
    while other_messages and estimate_messages_tokens(other_messages) > available_tokens:
        other_messages.pop(0)

    result = system_messages + other_messages

    removed_count = len(messages_list) - len(result)
    if removed_count > 0:
        logger.info(
            f"Truncated {removed_count} messages to fit context window "
            f"({current_tokens} -> {estimate_messages_tokens(result)} tokens)"
        )

    return result


class ContextManager:
    """
    Manages context window for a conversation.

    Provides automatic truncation when approaching limits and
    warnings when context is getting full.

    Example:
        ctx = ContextManager(model="gpt-4o", max_response_tokens=4096)

        # Check if messages fit
        fits, info = ctx.check(messages)
        if not fits:
            messages = ctx.truncate(messages)

        # Or use auto mode
        messages = ctx.prepare(messages)  # Automatically truncates if needed
    """

    def __init__(
        self,
        model: str,
        max_response_tokens: int = 4096,
        auto_truncate: bool = True,
        truncation_strategy: str = "oldest",
        keep_system: bool = True,
        warn_threshold: float = 0.8,
    ):
        """
        Initialize context manager.

        Args:
            model: Model name to get context limits for
            max_response_tokens: Expected max tokens in response
            auto_truncate: Whether to automatically truncate when needed
            truncation_strategy: Strategy for truncation ("oldest")
            keep_system: Whether to preserve system messages during truncation
            warn_threshold: Warn when context usage exceeds this ratio (0-1)
        """
        self._model = model
        self._limits = ContextLimits.for_model(model)
        self._max_response_tokens = max_response_tokens
        self._auto_truncate = auto_truncate
        self._truncation_strategy = truncation_strategy
        self._keep_system = keep_system
        self._warn_threshold = warn_threshold

    def check(self, messages: Messages) -> tuple[bool, dict[str, Any]]:
        """Check if messages fit within context limits."""
        return check_context_fit(messages, self._model, self._max_response_tokens)

    def truncate(self, messages: Messages) -> Messages:
        """Truncate messages to fit within limits."""
        max_input = self._limits.max_tokens - self._max_response_tokens
        return truncate_messages(
            messages,
            max_input,
            strategy=self._truncation_strategy,
            keep_system=self._keep_system,
        )

    def prepare(self, messages: Messages) -> Messages:
        """
        Prepare messages for completion.

        Checks fit and truncates if needed (when auto_truncate is enabled).
        Logs warnings when approaching limits.

        Args:
            messages: The messages to prepare

        Returns:
            Messages ready for completion (possibly truncated)
        """
        fits, info = self.check(messages)

        # Check warning threshold
        usage_ratio = info["estimated_tokens"] / self._limits.max_tokens
        if usage_ratio > self._warn_threshold:
            logger.warning(
                f"Context usage at {usage_ratio:.0%} "
                f"({info['estimated_tokens']}/{self._limits.max_tokens} tokens)"
            )

        if not fits:
            if self._auto_truncate:
                logger.warning(f"Context exceeded, truncating: {info.get('warning')}")
                return self.truncate(messages)
            else:
                raise ValueError(
                    f"Messages exceed context limit: {info.get('warning')}"
                )

        return list(messages)

    @property
    def model(self) -> str:
        """The model this manager is configured for."""
        return self._model

    @property
    def max_tokens(self) -> int:
        """Maximum context tokens for the model."""
        return self._limits.max_tokens
