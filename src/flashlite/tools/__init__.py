"""Tool/function calling helpers for Flashlite.

This module provides utilities for:
- Defining tools using decorators (@tool) or Pydantic models
- Converting tools to OpenAI/Anthropic formats
- Running tool execution loops for agentic patterns

Example:
    from flashlite import Flashlite
    from flashlite.tools import tool, run_tool_loop

    @tool()
    def get_weather(location: str, unit: str = "celsius") -> str:
        '''Get the current weather for a location.'''
        # Implementation
        return f"Weather in {location}: 72°F"

    @tool()
    def search_web(query: str, max_results: int = 5) -> list[str]:
        '''Search the web for information.'''
        return ["result1", "result2"]

    client = Flashlite(default_model="gpt-4o")

    # Run tool loop
    result = await run_tool_loop(
        client=client,
        messages=[{"role": "user", "content": "What's the weather in NYC?"}],
        tools=[get_weather, search_web],
    )
    print(result.content)
"""

from .definitions import (
    ToolDefinition,
    ToolRegistry,
    format_tool_result,
    get_tool_definition,
    tool,
    tool_from_pydantic,
    tools_to_anthropic,
    tools_to_openai,
)
from .execution import (
    ToolCall,
    ToolLoopResult,
    ToolResult,
    build_tool_registry,
    execute_tool,
    execute_tools_parallel,
    extract_tool_calls,
    run_tool_loop,
)

__all__ = [
    # Definitions
    "tool",
    "tool_from_pydantic",
    "ToolDefinition",
    "ToolRegistry",
    "get_tool_definition",
    "tools_to_openai",
    "tools_to_anthropic",
    "format_tool_result",
    # Execution
    "ToolCall",
    "ToolResult",
    "ToolLoopResult",
    "run_tool_loop",
    "execute_tool",
    "execute_tools_parallel",
    "extract_tool_calls",
    "build_tool_registry",
]
