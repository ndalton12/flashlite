"""Tool execution loop helpers for agentic patterns.

This module provides utilities for running tool execution loops where the
model can call tools and receive results in a conversation flow.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .definitions import ToolDefinition, format_tool_result

if TYPE_CHECKING:
    from ..client import Flashlite
    from ..types import CompletionResponse, Message

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a tool call from the model."""

    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_openai(cls, tool_call: dict[str, Any]) -> "ToolCall":
        """Parse from OpenAI tool call format."""
        func = tool_call.get("function", {})
        args_str = func.get("arguments", "{}")

        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {"raw": args_str}

        return cls(
            id=tool_call.get("id", ""),
            name=func.get("name", ""),
            arguments=args,
        )


@dataclass
class ToolResult:
    """Result of executing a tool."""

    tool_call_id: str
    name: str
    result: Any
    error: str | None = None

    def to_message(self) -> dict[str, Any]:
        """Convert to tool result message."""
        return format_tool_result(
            tool_call_id=self.tool_call_id,
            result=self.error if self.error else self.result,
            is_error=self.error is not None,
        )


@dataclass
class ToolLoopResult:
    """Result of a complete tool execution loop."""

    messages: list["Message"]  # Full conversation history
    final_response: "CompletionResponse"  # Final response from model
    tool_calls_made: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    iterations: int = 0

    @property
    def content(self) -> str:
        """Get the final response content."""
        return self.final_response.content


async def execute_tool(
    tool: ToolDefinition,
    arguments: dict[str, Any],
) -> tuple[Any, str | None]:
    """
    Execute a single tool with error handling.

    Returns:
        Tuple of (result, error_message). error_message is None on success.
    """
    try:
        result = await tool.execute(**arguments)
        return result, None
    except Exception as e:
        logger.warning(f"Tool {tool.name} failed: {e}")
        return None, str(e)


async def execute_tools_parallel(
    tools: dict[str, ToolDefinition],
    tool_calls: list[ToolCall],
) -> list[ToolResult]:
    """
    Execute multiple tool calls in parallel.

    Args:
        tools: Registry of available tools
        tool_calls: Tool calls to execute

    Returns:
        List of tool results
    """
    import asyncio

    async def run_one(tc: ToolCall) -> ToolResult:
        tool = tools.get(tc.name)
        if not tool:
            return ToolResult(
                tool_call_id=tc.id,
                name=tc.name,
                result=None,
                error=f"Unknown tool: {tc.name}",
            )

        result, error = await execute_tool(tool, tc.arguments)
        return ToolResult(
            tool_call_id=tc.id,
            name=tc.name,
            result=result,
            error=error,
        )

    return await asyncio.gather(*[run_one(tc) for tc in tool_calls])


def extract_tool_calls(response: "CompletionResponse") -> list[ToolCall]:
    """
    Extract tool calls from a completion response.

    Args:
        response: The completion response

    Returns:
        List of tool calls (empty if none)
    """
    # Check raw response for tool_calls
    raw = response.raw_response
    if raw is None:
        return []

    # Handle litellm ModelResponse
    if hasattr(raw, "choices"):
        choices = raw.choices
        if choices and len(choices) > 0:
            message = choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                return [ToolCall.from_openai(tc.model_dump()) for tc in message.tool_calls]

    # Handle dict response
    if isinstance(raw, dict):
        choices = raw.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls", [])
            return [ToolCall.from_openai(tc) for tc in tool_calls]

    return []


def build_tool_registry(
    tools: list[ToolDefinition | Callable[..., Any]],
) -> dict[str, ToolDefinition]:
    """
    Build a tool registry from a list of tools.

    Accepts both ToolDefinition objects and @tool decorated functions.
    """
    registry: dict[str, ToolDefinition] = {}
    for t in tools:
        if isinstance(t, ToolDefinition):
            registry[t.name] = t
        elif hasattr(t, "_tool_definition"):
            tool_def = t._tool_definition
            registry[tool_def.name] = tool_def
        else:
            raise ValueError(f"Not a valid tool: {t}")
    return registry


async def run_tool_loop(
    client: "Flashlite",
    messages: list["Message"],
    tools: list[ToolDefinition | Callable[..., Any]],
    *,
    model: str | None = None,
    max_iterations: int = 10,
    execute_parallel: bool = True,
    on_tool_call: Callable[[ToolCall], None] | None = None,
    on_tool_result: Callable[[ToolResult], None] | None = None,
    **completion_kwargs: Any,
) -> ToolLoopResult:
    """
    Run a tool execution loop until the model stops calling tools.

    This implements the standard agentic pattern:
    1. Call the model with messages and tools
    2. If model requests tool calls, execute them
    3. Add tool results to messages and repeat
    4. Continue until model returns without tool calls or max iterations

    Args:
        client: Flashlite client for completions
        messages: Initial messages
        tools: List of tools available to the model
        model: Model to use (defaults to client's default)
        max_iterations: Maximum tool call rounds (default: 10)
        execute_parallel: Execute multiple tool calls in parallel
        on_tool_call: Callback when a tool is called
        on_tool_result: Callback when a tool returns
        **completion_kwargs: Additional args passed to complete()

    Returns:
        ToolLoopResult with final response and history

    Example:
        @tool()
        def get_weather(location: str) -> str:
            '''Get weather for a location.'''
            return f"Weather in {location}: 72°F"

        result = await run_tool_loop(
            client=client,
            messages=[{"role": "user", "content": "What's the weather in NYC?"}],
            tools=[get_weather],
        )
        print(result.content)  # Final response after tool execution
    """
    # Build tool registry and convert to litellm format
    registry = build_tool_registry(tools)

    # Determine provider format based on model
    effective_model = model or client.config.default_model or ""
    model_lower = effective_model.lower()

    if "claude" in model_lower or "anthropic" in model_lower:
        from .definitions import tools_to_anthropic

        tools_param = tools_to_anthropic(tools)
    else:
        from .definitions import tools_to_openai

        tools_param = tools_to_openai(tools)

    # Track state
    current_messages = list(messages)
    all_tool_calls: list[ToolCall] = []
    all_tool_results: list[ToolResult] = []
    iterations = 0

    while iterations < max_iterations:
        iterations += 1

        # Call model with tools
        # Pass pre-converted tools via extra kwargs to avoid double-conversion
        response = await client.complete(
            model=model,
            messages=current_messages,
            **{**completion_kwargs, "tools": tools_param},
        )

        # Check for tool calls
        tool_calls = extract_tool_calls(response)

        if not tool_calls:
            # No more tool calls - we're done
            return ToolLoopResult(
                messages=current_messages,
                final_response=response,
                tool_calls_made=all_tool_calls,
                tool_results=all_tool_results,
                iterations=iterations,
            )

        # Add assistant message with tool calls
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content or ""}

        # Add tool_calls to message (needed for conversation continuity)
        if response.raw_response and hasattr(response.raw_response, "choices"):
            choices = response.raw_response.choices
            if choices and hasattr(choices[0].message, "tool_calls"):
                assistant_msg["tool_calls"] = [
                    tc.model_dump() for tc in choices[0].message.tool_calls
                ]
        current_messages.append(assistant_msg)

        # Execute tool calls
        if on_tool_call:
            for tc in tool_calls:
                on_tool_call(tc)

        all_tool_calls.extend(tool_calls)

        if execute_parallel:
            results = await execute_tools_parallel(registry, tool_calls)
        else:
            results = []
            for tc in tool_calls:
                tool = registry.get(tc.name)
                if tool:
                    result, error = await execute_tool(tool, tc.arguments)
                    results.append(
                        ToolResult(
                            tool_call_id=tc.id,
                            name=tc.name,
                            result=result,
                            error=error,
                        )
                    )
                else:
                    results.append(
                        ToolResult(
                            tool_call_id=tc.id,
                            name=tc.name,
                            result=None,
                            error=f"Unknown tool: {tc.name}",
                        )
                    )

        # Add tool results to messages
        for tr in results:
            if on_tool_result:
                on_tool_result(tr)
            all_tool_results.append(tr)
            current_messages.append(tr.to_message())

    # Max iterations reached
    logger.warning(f"Tool loop reached max iterations ({max_iterations})")

    # Make final call without tools to get a response
    response = await client.complete(
        model=model,
        messages=current_messages,
        **completion_kwargs,
    )

    return ToolLoopResult(
        messages=current_messages,
        final_response=response,
        tool_calls_made=all_tool_calls,
        tool_results=all_tool_results,
        iterations=iterations,
    )
