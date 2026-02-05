"""Tool definition helpers for function calling.

This module provides utilities for defining tools that can be used with LLM
function calling capabilities. Tools can be defined using:
1. The @tool decorator on functions
2. Pydantic models for complex parameter schemas
"""

import inspect
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar, get_type_hints

from pydantic import BaseModel

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class ToolDefinition:
    """Represents a callable tool with its schema."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    func: Callable[..., Any]
    strict: bool = False  # Some providers support strict mode

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to OpenAI tool format."""
        tool_def: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
        if self.strict:
            tool_def["function"]["strict"] = True
        return tool_def

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with given arguments."""
        if inspect.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        return self.func(**kwargs)

    def execute_sync(self, **kwargs: Any) -> Any:
        """Execute the tool synchronously."""
        if inspect.iscoroutinefunction(self.func):
            import asyncio

            return asyncio.run(self.func(**kwargs))
        return self.func(**kwargs)


@dataclass
class ToolRegistry:
    """Registry of available tools."""

    tools: dict[str, ToolDefinition] = field(default_factory=dict)

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self.tools.get(name)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Convert all tools to OpenAI format."""
        return [t.to_openai_tool() for t in self.tools.values()]

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        """Convert all tools to Anthropic format."""
        return [t.to_anthropic_tool() for t in self.tools.values()]

    def __iter__(self) -> "Iterator[ToolDefinition]":
        return iter(self.tools.values())

    def __len__(self) -> int:
        return len(self.tools)


def _python_type_to_json_schema(python_type: Any) -> dict[str, Any]:
    """Convert a Python type annotation to JSON Schema."""
    # Handle None type
    if python_type is type(None):
        return {"type": "null"}

    # Handle basic types
    type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
    }

    if python_type in type_mapping:
        return type_mapping[python_type]

    # Handle Optional (Union with None)
    origin = getattr(python_type, "__origin__", None)

    if origin is list:
        args = getattr(python_type, "__args__", (Any,))
        item_type = args[0] if args else Any
        return {"type": "array", "items": _python_type_to_json_schema(item_type)}

    if origin is dict:
        return {"type": "object"}

    # Handle Union types (including Optional)
    import types

    if origin is types.UnionType or (
        hasattr(python_type, "__origin__")
        and str(getattr(python_type, "__origin__", "")) == "typing.Union"
    ):
        args = getattr(python_type, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            # Optional[X] -> X's schema
            return _python_type_to_json_schema(non_none[0])

    # Handle Pydantic models
    if isinstance(python_type, type) and issubclass(python_type, BaseModel):
        return python_type.model_json_schema()

    # Handle Literal
    if hasattr(python_type, "__origin__") and str(
        getattr(python_type, "__origin__", "")
    ).endswith("Literal"):
        args = getattr(python_type, "__args__", ())
        return {"type": "string", "enum": list(args)}

    # Fallback
    return {"type": "string"}


def _extract_function_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Extract JSON Schema from function signature."""
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        # Get type hint
        type_hint = hints.get(name, str)
        prop_schema = _python_type_to_json_schema(type_hint)

        # Add description from docstring if available
        properties[name] = prop_schema

        # Check if required (no default value)
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }

    if required:
        schema["required"] = required

    return schema


def _extract_docstring_description(func: Callable[..., Any]) -> str:
    """Extract description from function docstring."""
    doc = func.__doc__
    if not doc:
        return f"Function {func.__name__}"

    # Get first paragraph (up to first blank line)
    lines = doc.strip().split("\n")
    desc_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            break
        desc_lines.append(stripped)

    return " ".join(desc_lines) if desc_lines else f"Function {func.__name__}"


def tool(
    name: str | None = None,
    description: str | None = None,
    strict: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to define a tool from a function.

    The function's type hints are used to generate the JSON Schema for
    the tool parameters. The docstring is used for the description.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        strict: Enable strict mode for providers that support it

    Example:
        @tool()
        def get_weather(location: str, unit: str = "celsius") -> str:
            '''Get the current weather for a location.'''
            return f"Weather in {location}: 72°F"

        @tool(name="search", description="Search the web")
        async def web_search(query: str, max_results: int = 10) -> list[str]:
            '''Search the web for information.'''
            return ["result1", "result2"]
    """

    def decorator(func: F) -> F:
        tool_name = name or func.__name__
        tool_desc = description or _extract_docstring_description(func)
        parameters = _extract_function_schema(func)

        # Attach tool definition to function
        tool_def = ToolDefinition(
            name=tool_name,
            description=tool_desc,
            parameters=parameters,
            func=func,
            strict=strict,
        )
        func._tool_definition = tool_def  # type: ignore

        return func

    return decorator


def tool_from_pydantic(
    model: type[BaseModel],
    func: Callable[..., Any],
    name: str | None = None,
    description: str | None = None,
    strict: bool = False,
) -> ToolDefinition:
    """
    Create a tool definition from a Pydantic model.

    This allows more complex parameter schemas with full Pydantic features
    like Field descriptions, validators, etc.

    Args:
        model: Pydantic model defining the tool parameters
        func: Function to execute (receives model instance or dict)
        name: Tool name (defaults to model name)
        description: Tool description (defaults to model docstring)
        strict: Enable strict mode

    Example:
        class SearchParams(BaseModel):
            '''Search the web for information.'''
            query: str = Field(description="The search query")
            max_results: int = Field(default=10, ge=1, le=100)
            safe_search: bool = Field(default=True)

        async def do_search(params: SearchParams) -> list[str]:
            return search_api(params.query, params.max_results)

        search_tool = tool_from_pydantic(SearchParams, do_search)
    """
    tool_name = name or model.__name__
    tool_desc = description or model.__doc__ or f"Tool {tool_name}"

    # Get JSON schema from Pydantic model
    schema = model.model_json_schema()

    # Remove $defs if present (inline definitions)
    if "$defs" in schema:
        del schema["$defs"]

    return ToolDefinition(
        name=tool_name,
        description=tool_desc,
        parameters=schema,
        func=func,
        strict=strict,
    )


def get_tool_definition(func: Callable[..., Any]) -> ToolDefinition | None:
    """Get the tool definition attached to a decorated function."""
    return getattr(func, "_tool_definition", None)


def tools_to_openai(tools: list[ToolDefinition | Callable[..., Any]]) -> list[dict[str, Any]]:
    """
    Convert a list of tools to OpenAI format.

    Accepts both ToolDefinition objects and @tool decorated functions.

    Args:
        tools: List of tools

    Returns:
        List of tool definitions in OpenAI format
    """
    result = []
    for t in tools:
        if isinstance(t, ToolDefinition):
            result.append(t.to_openai_tool())
        elif hasattr(t, "_tool_definition"):
            result.append(t._tool_definition.to_openai_tool())
        else:
            raise ValueError(f"Not a valid tool: {t}")
    return result


def tools_to_anthropic(tools: list[ToolDefinition | Callable[..., Any]]) -> list[dict[str, Any]]:
    """
    Convert a list of tools to Anthropic format.

    Accepts both ToolDefinition objects and @tool decorated functions.

    Args:
        tools: List of tools

    Returns:
        List of tool definitions in Anthropic format
    """
    result = []
    for t in tools:
        if isinstance(t, ToolDefinition):
            result.append(t.to_anthropic_tool())
        elif hasattr(t, "_tool_definition"):
            result.append(t._tool_definition.to_anthropic_tool())
        else:
            raise ValueError(f"Not a valid tool: {t}")
    return result


def format_tool_result(tool_call_id: str, result: Any, is_error: bool = False) -> dict[str, Any]:
    """
    Format a tool result for inclusion in messages.

    Args:
        tool_call_id: The ID from the assistant's tool_call
        result: The tool execution result
        is_error: Whether this is an error result

    Returns:
        Tool message dict
    """
    if isinstance(result, BaseModel):
        content = result.model_dump_json()
    elif isinstance(result, (dict, list)):
        content = json.dumps(result)
    else:
        content = str(result)

    message: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }

    if is_error:
        message["content"] = f"Error: {content}"

    return message
