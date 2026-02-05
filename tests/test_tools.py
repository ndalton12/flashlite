"""Tests for tool/function calling helpers."""

import json

import pytest
from pydantic import BaseModel, Field

from flashlite import Flashlite
from flashlite.tools import (
    ToolCall,
    ToolLoopResult,
    ToolRegistry,
    ToolResult,
    format_tool_result,
    get_tool_definition,
    tool,
    tool_from_pydantic,
    tools_to_anthropic,
    tools_to_openai,
)

# ============================================================================
# Test Tool Decorator
# ============================================================================


class TestToolDecorator:
    """Tests for @tool decorator."""

    def test_basic_tool_definition(self):
        """Should create tool definition from decorated function."""

        @tool()
        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"Weather in {location}: 72°F"

        assert hasattr(get_weather, "_tool_definition")
        tool_def = get_tool_definition(get_weather)
        assert tool_def is not None
        assert tool_def.name == "get_weather"
        assert "Get weather for a location" in tool_def.description

    def test_custom_name_and_description(self):
        """Should use custom name and description."""

        @tool(name="weather", description="Custom weather description")
        def get_weather(location: str) -> str:
            """Original docstring."""
            return ""

        tool_def = get_tool_definition(get_weather)
        assert tool_def is not None
        assert tool_def.name == "weather"
        assert tool_def.description == "Custom weather description"

    def test_parameter_schema_generation(self):
        """Should generate JSON schema from function signature."""

        @tool()
        def search(query: str, max_results: int = 10, include_images: bool = False) -> list:
            """Search for items."""
            return []

        tool_def = get_tool_definition(search)
        assert tool_def is not None
        params = tool_def.parameters

        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert "max_results" in params["properties"]
        assert "include_images" in params["properties"]

        # query is required (no default)
        assert "query" in params.get("required", [])

        # Types should be correct
        assert params["properties"]["query"]["type"] == "string"
        assert params["properties"]["max_results"]["type"] == "integer"
        assert params["properties"]["include_images"]["type"] == "boolean"

    def test_strict_mode(self):
        """Should support strict mode."""

        @tool(strict=True)
        def strict_func(x: str) -> str:
            """Strict function."""
            return x

        tool_def = get_tool_definition(strict_func)
        assert tool_def is not None
        assert tool_def.strict is True

    def test_async_function_support(self):
        """Should support async functions."""

        @tool()
        async def async_search(query: str) -> list:
            """Async search."""
            return [query]

        tool_def = get_tool_definition(async_search)
        assert tool_def is not None
        assert tool_def.name == "async_search"


# ============================================================================
# Test Tool from Pydantic
# ============================================================================


class TestToolFromPydantic:
    """Tests for tool_from_pydantic."""

    def test_basic_pydantic_tool(self):
        """Should create tool from Pydantic model."""

        class SearchParams(BaseModel):
            """Search the web."""

            query: str = Field(description="The search query")
            max_results: int = Field(default=10, ge=1, le=100)

        def do_search(query: str, max_results: int) -> list:
            return []

        tool_def = tool_from_pydantic(SearchParams, do_search)

        assert tool_def.name == "SearchParams"
        assert "Search the web" in tool_def.description
        assert "query" in tool_def.parameters["properties"]
        assert "max_results" in tool_def.parameters["properties"]

    def test_custom_name(self):
        """Should allow custom name."""

        class Params(BaseModel):
            x: int

        tool_def = tool_from_pydantic(Params, lambda x: x, name="my_tool")
        assert tool_def.name == "my_tool"


# ============================================================================
# Test Format Conversion
# ============================================================================


class TestFormatConversion:
    """Tests for converting tools to provider formats."""

    def setup_method(self):
        """Set up test tools."""

        @tool()
        def calculator(expression: str) -> str:
            """Evaluate a math expression."""
            return str(eval(expression))

        @tool()
        def get_weather(location: str, unit: str = "celsius") -> str:
            """Get weather for a location."""
            return ""

        self.calculator = calculator
        self.get_weather = get_weather

    def test_tools_to_openai(self):
        """Should convert to OpenAI format."""
        openai_tools = tools_to_openai([self.calculator, self.get_weather])

        assert len(openai_tools) == 2
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "calculator"
        assert "parameters" in openai_tools[0]["function"]

    def test_tools_to_anthropic(self):
        """Should convert to Anthropic format."""
        anthropic_tools = tools_to_anthropic([self.calculator, self.get_weather])

        assert len(anthropic_tools) == 2
        assert anthropic_tools[0]["name"] == "calculator"
        assert "input_schema" in anthropic_tools[0]
        assert "description" in anthropic_tools[0]

    def test_tool_definition_to_openai(self):
        """Should convert ToolDefinition to OpenAI format."""
        tool_def = get_tool_definition(self.calculator)
        assert tool_def is not None

        openai_tool = tool_def.to_openai_tool()
        assert openai_tool["type"] == "function"
        assert openai_tool["function"]["name"] == "calculator"

    def test_tool_definition_to_anthropic(self):
        """Should convert ToolDefinition to Anthropic format."""
        tool_def = get_tool_definition(self.calculator)
        assert tool_def is not None

        anthropic_tool = tool_def.to_anthropic_tool()
        assert anthropic_tool["name"] == "calculator"
        assert "input_schema" in anthropic_tool


# ============================================================================
# Test Tool Execution
# ============================================================================


class TestToolExecution:
    """Tests for tool execution."""

    def test_sync_execution(self):
        """Should execute sync functions."""

        @tool()
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool_def = get_tool_definition(add)
        assert tool_def is not None

        result = tool_def.execute_sync(a=2, b=3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_async_execution(self):
        """Should execute async functions."""

        @tool()
        async def async_add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool_def = get_tool_definition(async_add)
        assert tool_def is not None

        result = await tool_def.execute(a=2, b=3)
        assert result == 5


# ============================================================================
# Test Tool Registry
# ============================================================================


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_and_get(self):
        """Should register and retrieve tools."""
        registry = ToolRegistry()

        @tool()
        def my_func(x: str) -> str:
            """Test function."""
            return x

        tool_def = get_tool_definition(my_func)
        assert tool_def is not None

        registry.register(tool_def)
        assert registry.get("my_func") is tool_def
        assert registry.get("nonexistent") is None

    def test_iteration(self):
        """Should iterate over tools."""
        registry = ToolRegistry()

        @tool()
        def func1(x: str) -> str:
            return x

        @tool()
        def func2(x: str) -> str:
            return x

        tool_def1 = get_tool_definition(func1)
        tool_def2 = get_tool_definition(func2)
        assert tool_def1 is not None
        assert tool_def2 is not None

        registry.register(tool_def1)
        registry.register(tool_def2)

        assert len(registry) == 2
        names = [t.name for t in registry]
        assert "func1" in names
        assert "func2" in names


# ============================================================================
# Test Tool Call Parsing
# ============================================================================


class TestToolCallParsing:
    """Tests for parsing tool calls from responses."""

    def test_tool_call_from_openai(self):
        """Should parse OpenAI tool call format."""
        raw_call = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "NYC", "unit": "fahrenheit"}',
            },
        }

        tool_call = ToolCall.from_openai(raw_call)
        assert tool_call.id == "call_123"
        assert tool_call.name == "get_weather"
        assert tool_call.arguments == {"location": "NYC", "unit": "fahrenheit"}

    def test_tool_call_invalid_json(self):
        """Should handle invalid JSON arguments."""
        raw_call = {
            "id": "call_123",
            "function": {
                "name": "test",
                "arguments": "invalid json",
            },
        }

        tool_call = ToolCall.from_openai(raw_call)
        assert tool_call.arguments == {"raw": "invalid json"}


# ============================================================================
# Test Tool Result Formatting
# ============================================================================


class TestToolResultFormatting:
    """Tests for formatting tool results."""

    def test_string_result(self):
        """Should format string result."""
        msg = format_tool_result("call_123", "Weather: 72°F")

        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_123"
        assert msg["content"] == "Weather: 72°F"

    def test_dict_result(self):
        """Should JSON encode dict result."""
        msg = format_tool_result("call_123", {"temp": 72, "unit": "F"})

        assert msg["content"] == json.dumps({"temp": 72, "unit": "F"})

    def test_pydantic_result(self):
        """Should JSON encode Pydantic result."""

        class Weather(BaseModel):
            temp: int
            unit: str

        msg = format_tool_result("call_123", Weather(temp=72, unit="F"))

        content = json.loads(msg["content"])
        assert content["temp"] == 72
        assert content["unit"] == "F"

    def test_error_result(self):
        """Should format error result."""
        msg = format_tool_result("call_123", "Something went wrong", is_error=True)

        assert "Error:" in msg["content"]
        assert "Something went wrong" in msg["content"]


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_to_message_success(self):
        """Should convert success to message."""
        result = ToolResult(
            tool_call_id="call_123",
            name="get_weather",
            result="72°F",
            error=None,
        )

        msg = result.to_message()
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_123"
        assert "72°F" in msg["content"]

    def test_to_message_error(self):
        """Should convert error to message."""
        result = ToolResult(
            tool_call_id="call_123",
            name="get_weather",
            result=None,
            error="API timeout",
        )

        msg = result.to_message()
        assert "Error:" in msg["content"]
        assert "API timeout" in msg["content"]


# ============================================================================
# Test Client Integration
# ============================================================================


class TestClientToolsIntegration:
    """Tests for tools integration with Flashlite client."""

    @pytest.mark.asyncio
    async def test_tools_parameter_openai(self):
        """Should pass tools to OpenAI model."""
        client = Flashlite(default_model="gpt-4o")

        @tool()
        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return ""

        # Just verify the tools are converted - use mock response
        response = await client.complete(
            messages="What's the weather?",
            tools=[get_weather],
            mock_response="I'll help you check the weather.",
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_tools_parameter_anthropic(self):
        """Should pass tools to Anthropic model."""
        # Use anthropic/ prefix for litellm compatibility
        client = Flashlite(default_model="anthropic/claude-3-sonnet-20240229")

        @tool()
        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return ""

        response = await client.complete(
            messages="What's the weather?",
            tools=[get_weather],
            mock_response="I'll help you check the weather.",
        )

        assert response is not None


# ============================================================================
# Test Tool Loop Result
# ============================================================================


class TestToolLoopResult:
    """Tests for ToolLoopResult."""

    def test_content_property(self):
        """Should access content from final response."""
        from flashlite.types import CompletionResponse

        result = ToolLoopResult(
            messages=[],
            final_response=CompletionResponse(
                content="Final answer",
                model="gpt-4o",
            ),
            iterations=1,
        )

        assert result.content == "Final answer"


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_tool_without_docstring(self):
        """Should handle function without docstring."""

        @tool()
        def no_docs(x: str) -> str:
            return x

        tool_def = get_tool_definition(no_docs)
        assert tool_def is not None
        assert tool_def.description == "Function no_docs"

    def test_tool_with_complex_types(self):
        """Should handle complex type annotations."""

        @tool()
        def complex_func(
            items: list[str],
            config: dict[str, int] | None = None,
        ) -> list[dict]:
            """Complex function."""
            return []

        tool_def = get_tool_definition(complex_func)
        assert tool_def is not None
        params = tool_def.parameters

        assert params["properties"]["items"]["type"] == "array"
        assert params["properties"]["items"]["items"]["type"] == "string"

    def test_invalid_tool_in_list(self):
        """Should raise error for invalid tool."""

        def not_a_tool(x: str) -> str:
            return x

        with pytest.raises(ValueError, match="Not a valid tool"):
            tools_to_openai([not_a_tool])
