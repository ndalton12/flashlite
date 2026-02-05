"""Message formatting utilities."""


from ..types import Message, Messages, Role


def user_message(content: str, name: str | None = None) -> Message:
    """Create a user message."""
    msg: Message = {"role": "user", "content": content}
    if name:
        msg["name"] = name
    return msg


def system_message(content: str) -> Message:
    """Create a system message."""
    return {"role": "system", "content": content}


def assistant_message(content: str, name: str | None = None) -> Message:
    """Create an assistant message."""
    msg: Message = {"role": "assistant", "content": content}
    if name:
        msg["name"] = name
    return msg


def tool_message(content: str, tool_call_id: str) -> Message:
    """Create a tool result message."""
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": tool_call_id,
    }


def format_messages(
    messages: Messages | str | None = None,
    system: str | None = None,
    user: str | None = None,
) -> list[Message]:
    """
    Flexibly format messages into a standard list format.

    This allows multiple ways to specify messages:
    - As a list of message dicts (pass through)
    - As a single string (converted to user message)
    - Using system/user kwargs for simple single-turn

    Args:
        messages: Existing messages list or single string
        system: System prompt to prepend
        user: User message to append

    Returns:
        Normalized list of message dicts
    """
    result: list[Message] = []

    # Add system message if provided
    if system:
        result.append(system_message(system))

    # Handle messages parameter
    if messages is not None:
        if isinstance(messages, str):
            # Single string becomes user message
            result.append(user_message(messages))
        else:
            # List of messages - copy to avoid mutation
            result.extend(list(messages))

    # Add user message if provided separately
    if user:
        result.append(user_message(user))

    return result


def extract_content(message: Message) -> str:
    """Extract text content from a message."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Handle multimodal content (text parts)
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts)
    return str(content) if content else ""


def validate_messages(messages: Messages) -> list[str]:
    """
    Validate a list of messages.

    Returns list of validation errors (empty if valid).
    """
    errors: list[str] = []
    valid_roles: set[Role] = {"system", "user", "assistant", "tool"}

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"Message {i}: must be a dict, got {type(msg).__name__}")
            continue

        role = msg.get("role")
        if role not in valid_roles:
            errors.append(f"Message {i}: invalid role '{role}', must be one of {valid_roles}")

        if "content" not in msg and "tool_calls" not in msg:
            errors.append(f"Message {i}: must have 'content' or 'tool_calls'")

        if role == "tool" and "tool_call_id" not in msg:
            errors.append(f"Message {i}: tool message must have 'tool_call_id'")

    return errors


def count_messages_by_role(messages: Messages) -> dict[str, int]:
    """Count messages by role."""
    counts: dict[str, int] = {}
    for msg in messages:
        role = msg.get("role", "unknown")
        counts[role] = counts.get(role, 0) + 1
    return counts
