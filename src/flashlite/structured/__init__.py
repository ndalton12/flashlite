"""Structured outputs module for Pydantic model integration."""

from .outputs import (
    StructuredOutputError,
    extract_json_from_content,
    format_validation_error_for_retry,
    parse_json_response,
    validate_response,
)
from .schema import (
    format_schema_for_openai,
    generate_json_schema,
    get_field_descriptions,
    is_supported_type,
    schema_to_prompt,
)

__all__ = [
    # Schema generation
    "generate_json_schema",
    "schema_to_prompt",
    "get_field_descriptions",
    "format_schema_for_openai",
    "is_supported_type",
    # Parsing and validation
    "parse_json_response",
    "validate_response",
    "format_validation_error_for_retry",
    "extract_json_from_content",
    "StructuredOutputError",
]
