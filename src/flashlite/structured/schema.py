"""JSON schema generation from Pydantic models."""

import json
from typing import Any

from pydantic import BaseModel


def generate_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """
    Generate a JSON schema from a Pydantic model.

    This uses Pydantic's built-in schema generation and formats it
    for use with LLM structured output features.

    Args:
        model: A Pydantic BaseModel class

    Returns:
        JSON schema dict suitable for LLM prompts or response_format
    """
    # Use Pydantic's built-in schema generation
    schema = model.model_json_schema()

    # Clean up schema for LLM consumption
    schema = _simplify_schema(schema)

    return schema


def _simplify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Simplify a JSON schema for LLM consumption.

    Removes Pydantic-specific fields and flattens $defs references
    where possible for cleaner prompts.
    """
    # Remove Pydantic-specific metadata
    keys_to_remove = ["title", "$defs", "definitions"]

    # If there are $defs, inline them
    defs = schema.get("$defs", schema.get("definitions", {}))

    def resolve_refs(obj: Any) -> Any:
        """Recursively resolve $ref references."""
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_path = obj["$ref"]
                # Extract the definition name from "#/$defs/Name"
                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path.split("/")[-1]
                    if def_name in defs:
                        # Return a copy of the resolved definition
                        resolved = resolve_refs(defs[def_name].copy())
                        # Remove title from inlined definitions
                        resolved.pop("title", None)
                        return resolved
                elif ref_path.startswith("#/definitions/"):
                    def_name = ref_path.split("/")[-1]
                    if def_name in defs:
                        resolved = resolve_refs(defs[def_name].copy())
                        resolved.pop("title", None)
                        return resolved
            return {k: resolve_refs(v) for k, v in obj.items() if k not in keys_to_remove}
        elif isinstance(obj, list):
            return [resolve_refs(item) for item in obj]
        return obj

    result = resolve_refs(schema)

    # Remove top-level title if present
    result.pop("title", None)

    return result


def schema_to_prompt(model: type[BaseModel]) -> str:
    """
    Convert a Pydantic model to a prompt-friendly schema description.

    This generates a human-readable description of the expected JSON
    structure that can be included in system prompts.

    Args:
        model: A Pydantic BaseModel class

    Returns:
        A string describing the expected JSON format
    """
    schema = generate_json_schema(model)

    lines = ["You must respond with valid JSON matching this schema:", ""]
    lines.append("```json")
    lines.append(json.dumps(schema, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("Important:")
    lines.append("- Respond ONLY with the JSON object, no other text")
    lines.append("- Ensure all required fields are present")
    lines.append("- Use the exact field names and types specified")

    return "\n".join(lines)


def get_field_descriptions(model: type[BaseModel]) -> dict[str, str]:
    """
    Extract field descriptions from a Pydantic model.

    Args:
        model: A Pydantic BaseModel class

    Returns:
        Dict mapping field names to their descriptions
    """
    descriptions = {}
    for field_name, field_info in model.model_fields.items():
        if field_info.description:
            descriptions[field_name] = field_info.description
    return descriptions


def format_schema_for_openai(model: type[BaseModel]) -> dict[str, Any]:
    """
    Format a schema for OpenAI's structured outputs feature.

    OpenAI's structured outputs (response_format with json_schema)
    requires a specific format with name and strict fields.

    Args:
        model: A Pydantic BaseModel class

    Returns:
        Dict formatted for OpenAI's response_format parameter
    """
    schema = model.model_json_schema()

    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": True,
            "schema": schema,
        },
    }


def is_supported_type(model: type[BaseModel]) -> bool:
    """
    Check if a Pydantic model uses only supported types for structured outputs.

    Most types are supported, but this can help identify potential issues.

    Args:
        model: A Pydantic BaseModel class

    Returns:
        True if all field types are supported
    """
    # For now, we support all types that Pydantic can serialize to JSON schema
    # This could be expanded to check for specific unsupported types
    try:
        model.model_json_schema()
        return True
    except Exception:
        return False
