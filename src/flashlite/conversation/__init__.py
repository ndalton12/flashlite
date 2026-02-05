"""Conversation management module for flashlite."""

from .context import (
    ContextLimits,
    ContextManager,
    check_context_fit,
    estimate_messages_tokens,
    estimate_tokens,
    truncate_messages,
)
from .manager import Conversation, ConversationState, Turn
from .multi_agent import Agent, ChatMessage, MultiAgentChat

__all__ = [
    # Conversation management
    "Conversation",
    "ConversationState",
    "Turn",
    # Multi-agent conversations
    "MultiAgentChat",
    "Agent",
    "ChatMessage",
    # Context management
    "ContextManager",
    "ContextLimits",
    "estimate_tokens",
    "estimate_messages_tokens",
    "check_context_fit",
    "truncate_messages",
]
