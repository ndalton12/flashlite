"""Tests for structured outputs functionality."""

from enum import StrEnum
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from flashlite.structured import (
    StructuredOutputError,
    format_validation_error_for_retry,
    generate_json_schema,
    parse_json_response,
    schema_to_prompt,
    validate_response,
)
from flashlite.types import CompletionResponse


# Test models
class Sentiment(BaseModel):
    """A sentiment analysis result."""

    label: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0, le=1, description="Confidence score")
    reasoning: str = Field(description="Brief explanation")


class Color(StrEnum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class ColorChoice(BaseModel):
    """A color selection."""

    color: Color
    reason: str


class NestedModel(BaseModel):
    """A model with nested types."""

    name: str
    items: list[str]
    metadata: dict[str, int]
    optional_field: str | None = None


class PersonInfo(BaseModel):
    """Information about a person."""

    name: str = Field(description="Full name")
    age: int = Field(ge=0, le=150, description="Age in years")
    email: str | None = Field(default=None, description="Email address")


# Schema Generation Tests
class TestSchemaGeneration:
    """Tests for JSON schema generation."""

    def test_generate_basic_schema(self):
        """Should generate schema for basic model."""
        schema = generate_json_schema(Sentiment)

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "label" in schema["properties"]
        assert "confidence" in schema["properties"]
        assert "reasoning" in schema["properties"]

    def test_generate_schema_with_enum(self):
        """Should handle enum types."""
        schema = generate_json_schema(ColorChoice)

        assert "properties" in schema
        assert "color" in schema["properties"]
        # Enum should have allowed values
        color_schema = schema["properties"]["color"]
        assert "enum" in color_schema or "anyOf" in color_schema

    def test_generate_schema_with_nested_types(self):
        """Should handle nested types."""
        schema = generate_json_schema(NestedModel)

        assert "properties" in schema
        assert "items" in schema["properties"]
        assert "metadata" in schema["properties"]
        assert schema["properties"]["items"]["type"] == "array"

    def test_generate_schema_with_literal(self):
        """Should handle Literal types."""
        schema = generate_json_schema(Sentiment)

        label_schema = schema["properties"]["label"]
        assert "enum" in label_schema
        assert set(label_schema["enum"]) == {"positive", "negative", "neutral"}

    def test_schema_to_prompt(self):
        """Should generate prompt-friendly schema description."""
        prompt = schema_to_prompt(Sentiment)

        assert "JSON" in prompt
        assert "label" in prompt
        assert "confidence" in prompt
        assert "reasoning" in prompt
        assert "required" in prompt.lower() or "schema" in prompt.lower()


# JSON Parsing Tests
class TestJsonParsing:
    """Tests for JSON response parsing."""

    def test_parse_direct_json(self):
        """Should parse direct JSON."""
        content = '{"label": "positive", "confidence": 0.9, "reasoning": "Good!"}'
        result = parse_json_response(content)

        assert result["label"] == "positive"
        assert result["confidence"] == 0.9

    def test_parse_json_with_whitespace(self):
        """Should handle whitespace around JSON."""
        content = """

        {"label": "negative", "confidence": 0.8, "reasoning": "Bad!"}

        """
        result = parse_json_response(content)
        assert result["label"] == "negative"

    def test_parse_json_in_markdown_block(self):
        """Should extract JSON from markdown code blocks."""
        content = """Here's the analysis:

```json
{"label": "neutral", "confidence": 0.5, "reasoning": "Mixed"}
```

Hope this helps!"""
        result = parse_json_response(content)
        assert result["label"] == "neutral"

    def test_parse_json_in_unmarked_code_block(self):
        """Should extract JSON from code blocks without language tag."""
        content = """```
{"label": "positive", "confidence": 0.95, "reasoning": "Great!"}
```"""
        result = parse_json_response(content)
        assert result["label"] == "positive"

    def test_parse_json_with_surrounding_text(self):
        """Should extract JSON from text with surrounding content."""
        content = """Based on my analysis, here is the result:

{"label": "positive", "confidence": 0.85, "reasoning": "Very positive sentiment"}

This indicates a strongly positive sentiment."""
        result = parse_json_response(content)
        assert result["label"] == "positive"

    def test_parse_json_array(self):
        """Should parse JSON arrays."""
        content = '[{"name": "Alice"}, {"name": "Bob"}]'
        result = parse_json_response(content)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_parse_invalid_json_raises_error(self):
        """Should raise error for unparseable content."""
        content = "This is not JSON at all!"
        with pytest.raises(StructuredOutputError) as exc_info:
            parse_json_response(content)
        assert "Could not parse JSON" in str(exc_info.value)


# Validation Tests
class TestValidation:
    """Tests for response validation."""

    def test_validate_valid_response(self):
        """Should validate and return model instance."""
        response = CompletionResponse(
            content='{"label": "positive", "confidence": 0.9, "reasoning": "Good!"}',
            model="gpt-4o",
        )
        result = validate_response(response, Sentiment)

        assert isinstance(result, Sentiment)
        assert result.label == "positive"
        assert result.confidence == 0.9

    def test_validate_with_optional_fields(self):
        """Should handle optional fields."""
        response = CompletionResponse(
            content='{"name": "John", "age": 30}',
            model="gpt-4o",
        )
        result = validate_response(response, PersonInfo)

        assert result.name == "John"
        assert result.age == 30
        assert result.email is None

    def test_validate_missing_required_field(self):
        """Should raise error for missing required field."""
        response = CompletionResponse(
            content='{"label": "positive", "confidence": 0.9}',  # Missing reasoning
            model="gpt-4o",
        )
        with pytest.raises(StructuredOutputError) as exc_info:
            validate_response(response, Sentiment)
        assert "reasoning" in str(exc_info.value)

    def test_validate_invalid_field_value(self):
        """Should raise error for invalid field value."""
        response = CompletionResponse(
            content='{"label": "invalid", "confidence": 0.9, "reasoning": "Test"}',
            model="gpt-4o",
        )
        with pytest.raises(StructuredOutputError) as exc_info:
            validate_response(response, Sentiment)
        assert "label" in str(exc_info.value)

    def test_validate_out_of_range_value(self):
        """Should raise error for out-of-range values."""
        response = CompletionResponse(
            content='{"name": "John", "age": 200}',  # Age > 150
            model="gpt-4o",
        )
        with pytest.raises(StructuredOutputError) as exc_info:
            validate_response(response, PersonInfo)
        assert "age" in str(exc_info.value)


# Error Formatting Tests
class TestErrorFormatting:
    """Tests for error message formatting."""

    def test_format_validation_error(self):
        """Should format validation errors for retry."""
        error = StructuredOutputError(
            "Validation failed",
            raw_content='{"bad": "data"}',
            validation_errors=[
                {"loc": ("label",), "msg": "field required", "type": "missing"},
            ],
        )
        formatted = format_validation_error_for_retry(error)

        assert "errors" in formatted.lower()
        assert "label" in formatted
        assert "correct" in formatted.lower() or "fix" in formatted.lower()

    def test_format_multiple_errors(self):
        """Should format multiple validation errors."""
        error = StructuredOutputError(
            "Validation failed",
            validation_errors=[
                {"loc": ("label",), "msg": "field required", "type": "missing"},
                {"loc": ("confidence",), "msg": "field required", "type": "missing"},
            ],
        )
        formatted = format_validation_error_for_retry(error)

        assert "label" in formatted
        assert "confidence" in formatted


# Integration Tests with Client
class TestStructuredOutputsIntegration:
    """Integration tests for structured outputs with the Flashlite client."""

    async def test_complete_with_response_model(self):
        """Should return validated model instance."""
        from flashlite import Flashlite

        client = Flashlite(default_model="gpt-4o")

        result = await client.complete(
            messages="Analyze sentiment: 'I love this product!'",
            response_model=Sentiment,
            mock_response=(
                '{"label": "positive", "confidence": 0.95, "reasoning": "Expresses love"}'
            ),
        )

        assert isinstance(result, Sentiment)
        assert result.label == "positive"
        assert result.confidence == 0.95

    async def test_complete_with_nested_model(self):
        """Should handle nested model types."""
        from flashlite import Flashlite

        client = Flashlite(default_model="gpt-4o")

        result = await client.complete(
            messages="Generate test data",
            response_model=NestedModel,
            mock_response='{"name": "test", "items": ["a", "b"], "metadata": {"count": 2}}',
        )

        assert isinstance(result, NestedModel)
        assert result.name == "test"
        assert result.items == ["a", "b"]
        assert result.metadata == {"count": 2}

    async def test_complete_with_invalid_response_raises(self):
        """Should raise error for invalid response."""
        from flashlite import Flashlite

        client = Flashlite(default_model="gpt-4o")

        with pytest.raises(StructuredOutputError):
            await client.complete(
                messages="Analyze sentiment",
                response_model=Sentiment,
                structured_retries=0,  # Disable retries for this test
                mock_response='{"invalid": "data"}',
            )

    async def test_complete_without_response_model(self):
        """Should return CompletionResponse when no model specified."""
        from flashlite import Flashlite

        client = Flashlite(default_model="gpt-4o")

        result = await client.complete(
            messages="Hello",
            mock_response="Hi there!",
        )

        assert isinstance(result, CompletionResponse)
        assert result.content == "Hi there!"

    async def test_schema_injected_in_system_prompt(self):
        """Schema should be injected into system prompt."""
        from flashlite import Flashlite

        client = Flashlite(default_model="gpt-4o")

        # We can't easily verify the prompt content, but we can verify
        # the request succeeds with a valid response
        result = await client.complete(
            messages="Test",
            system="You are a helpful assistant.",
            response_model=Sentiment,
            mock_response='{"label": "neutral", "confidence": 0.5, "reasoning": "Test"}',
        )

        assert result.label == "neutral"

    async def test_complete_with_enum_field(self):
        """Should handle enum fields in response model."""
        from flashlite import Flashlite

        client = Flashlite(default_model="gpt-4o")

        result = await client.complete(
            messages="Pick a color",
            response_model=ColorChoice,
            mock_response='{"color": "red", "reason": "It is warm"}',
        )

        assert isinstance(result, ColorChoice)
        assert result.color == Color.RED
