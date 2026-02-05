"""
Flashlite - Batteries-included wrapper for litellm.

Features:
- Rate limiting with token bucket algorithm
- Retries with exponential backoff
- Jinja templating for prompts
- Async-first with sync wrappers
- Full passthrough of provider kwargs
- Response caching (memory and disk backends)
- Cost tracking and budget limits
- Structured logging and Inspect framework integration
"""

from .cache import CacheBackend, DiskCache, MemoryCache, generate_cache_key
from .client import Flashlite
from .config import FlashliteConfig, load_env_files, validate_api_keys
from .conversation import (
    Agent,
    ChatMessage,
    ContextLimits,
    ContextManager,
    Conversation,
    ConversationState,
    MultiAgentChat,
    Turn,
    estimate_messages_tokens,
    estimate_tokens,
    truncate_messages,
)
from .core.messages import (
    assistant_message,
    format_messages,
    system_message,
    tool_message,
    user_message,
)
from .observability import (
    BudgetExceededError,
    CallbackManager,
    CostTracker,
    FlashliteModelAPI,
    InspectLogger,
    StructuredLogger,
)
from .structured import (
    StructuredOutputError,
    generate_json_schema,
    parse_json_response,
    schema_to_prompt,
)
from .templating import TemplateEngine, TemplateRegistry
from .tools import (
    ToolCall,
    ToolDefinition,
    ToolLoopResult,
    ToolRegistry,
    ToolResult,
    format_tool_result,
    run_tool_loop,
    tool,
    tool_from_pydantic,
    tools_to_anthropic,
    tools_to_openai,
)
from .types import (
    CompletionError,
    # Request/Response types
    CompletionRequest,
    CompletionResponse,
    ConfigError,
    # Exceptions
    FlashliteError,
    # Message types
    Message,
    MessageDict,
    Messages,
    RateLimitConfig,
    RateLimitError,
    # Configuration types
    RetryConfig,
    Role,
    TemplateError,
    ThinkingConfig,
    UsageInfo,
    ValidationError,
    thinking_enabled,
)

__version__ = "0.1.0"

__all__ = [
    # Main client
    "Flashlite",
    # Configuration
    "FlashliteConfig",
    "RetryConfig",
    "RateLimitConfig",
    "ThinkingConfig",
    "thinking_enabled",
    "load_env_files",
    "validate_api_keys",
    # Request/Response
    "CompletionRequest",
    "CompletionResponse",
    "UsageInfo",
    # Messages
    "Message",
    "Messages",
    "MessageDict",
    "Role",
    "format_messages",
    "user_message",
    "system_message",
    "assistant_message",
    "tool_message",
    # Templating
    "TemplateEngine",
    "TemplateRegistry",
    # Caching
    "CacheBackend",
    "MemoryCache",
    "DiskCache",
    "generate_cache_key",
    # Conversation
    "Conversation",
    "ConversationState",
    "Turn",
    "ContextManager",
    "ContextLimits",
    "estimate_tokens",
    "estimate_messages_tokens",
    "truncate_messages",
    # Multi-agent
    "MultiAgentChat",
    "Agent",
    "ChatMessage",
    # Observability
    "StructuredLogger",
    "CostTracker",
    "BudgetExceededError",
    "CallbackManager",
    "InspectLogger",
    "FlashliteModelAPI",
    # Structured outputs
    "StructuredOutputError",
    "generate_json_schema",
    "schema_to_prompt",
    "parse_json_response",
    # Exceptions
    "FlashliteError",
    "CompletionError",
    "RateLimitError",
    "ValidationError",
    "TemplateError",
    "ConfigError",
    # Tools
    "tool",
    "tool_from_pydantic",
    "ToolDefinition",
    "ToolRegistry",
    "ToolCall",
    "ToolResult",
    "ToolLoopResult",
    "run_tool_loop",
    "tools_to_openai",
    "tools_to_anthropic",
    "format_tool_result",
]
