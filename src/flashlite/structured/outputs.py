"""Structured output parsing and validation."""

import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ..types import CompletionResponse

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """Error parsing or validating structured output."""

    def __init__(
        self,
        message: str,
        raw_content: str | None = None,
        validation_errors: list[dict] | None = None,
    ):
        super().__init__(message)
        self.raw_content = raw_content
        self.validation_errors = validation_errors or []


def parse_json_response(content: str) -> dict:
    """
    Parse JSON from an LLM response.

    Handles common cases where the model wraps JSON in markdown code blocks
    or includes extra text.

    Args:
        content: The raw response content from the LLM

    Returns:
        Parsed JSON as a dict

    Raises:
        StructuredOutputError: If JSON cannot be parsed
    """
    # Strip whitespace
    content = content.strip()

    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    # Match ```json ... ``` or ``` ... ```
    json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(json_block_pattern, content)

    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try to find JSON object boundaries
    # Look for first { and last }
    first_brace = content.find("{")
    last_brace = content.rfind("}")

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(content[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    # Try to find JSON array boundaries
    first_bracket = content.find("[")
    last_bracket = content.rfind("]")

    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        try:
            return json.loads(content[first_bracket : last_bracket + 1])
        except json.JSONDecodeError:
            pass

    raise StructuredOutputError(
        f"Could not parse JSON from response: {content[:200]}...",
        raw_content=content,
    )


def validate_response(
    response: CompletionResponse,
    model: type[T],
) -> T:
    """
    Parse and validate a completion response against a Pydantic model.

    Args:
        response: The completion response to validate
        model: The Pydantic model class to validate against

    Returns:
        A validated instance of the model

    Raises:
        StructuredOutputError: If parsing or validation fails
    """
    content = response.content

    # Parse JSON
    try:
        data = parse_json_response(content)
    except StructuredOutputError:
        raise

    # Validate against model
    try:
        return model.model_validate(data)
    except ValidationError as e:
        errors = e.errors()
        error_messages = []
        for err in errors:
            loc = ".".join(str(x) for x in err["loc"])
            msg = err["msg"]
            error_messages.append(f"  - {loc}: {msg}")

        raise StructuredOutputError(
            "Validation failed:\n" + "\n".join(error_messages),
            raw_content=content,
            validation_errors=[dict(e) for e in errors],
        )


def format_validation_error_for_retry(error: StructuredOutputError) -> str:
    """
    Format a validation error as feedback for the model to retry.

    This creates a message that explains what went wrong so the model
    can correct its response.

    Args:
        error: The structured output error

    Returns:
        A formatted error message for the retry prompt
    """
    lines = ["Your previous response had the following errors:", ""]

    if error.validation_errors:
        for err in error.validation_errors:
            loc = ".".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "Unknown error")
            input_val = err.get("input")

            if loc:
                lines.append(f"- Field '{loc}': {msg}")
                if input_val is not None:
                    lines.append(f"  Got: {input_val!r}")
            else:
                lines.append(f"- {msg}")
    else:
        lines.append(f"- {str(error)}")

    lines.append("")
    lines.append("Please correct these errors and respond with valid JSON.")

    return "\n".join(lines)


def extract_json_from_content(content: str) -> str | None:
    """
    Extract just the JSON portion from content that may contain other text.

    Returns the JSON string if found, None otherwise.

    Args:
        content: The raw content

    Returns:
        The extracted JSON string or None
    """
    try:
        data = parse_json_response(content)
        return json.dumps(data)
    except StructuredOutputError:
        return None
