"""Tests for message formatting utilities."""

from flashlite.core.messages import (
    assistant_message,
    count_messages_by_role,
    extract_content,
    format_messages,
    system_message,
    tool_message,
    user_message,
    validate_messages,
)


class TestMessageCreation:
    """Tests for message creation helpers."""

    def test_user_message(self) -> None:
        msg = user_message("Hello!")
        assert msg["role"] == "user"
        assert msg["content"] == "Hello!"
        assert "name" not in msg

    def test_user_message_with_name(self) -> None:
        msg = user_message("Hello!", name="Alice")
        assert msg["role"] == "user"
        assert msg["content"] == "Hello!"
        assert msg["name"] == "Alice"

    def test_system_message(self) -> None:
        msg = system_message("You are helpful.")
        assert msg["role"] == "system"
        assert msg["content"] == "You are helpful."

    def test_assistant_message(self) -> None:
        msg = assistant_message("I can help with that.")
        assert msg["role"] == "assistant"
        assert msg["content"] == "I can help with that."

    def test_tool_message(self) -> None:
        msg = tool_message("Result: 42", tool_call_id="call_123")
        assert msg["role"] == "tool"
        assert msg["content"] == "Result: 42"
        assert msg["tool_call_id"] == "call_123"


class TestFormatMessages:
    """Tests for format_messages function."""

    def test_string_to_user_message(self) -> None:
        result = format_messages("Hello!")
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello!"

    def test_with_system_prompt(self) -> None:
        result = format_messages("Hello!", system="Be helpful.")
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be helpful."
        assert result[1]["role"] == "user"

    def test_with_separate_user_kwarg(self) -> None:
        result = format_messages(system="Be helpful.", user="Hello!")
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Hello!"

    def test_passthrough_message_list(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
        ]
        result = format_messages(messages)
        assert len(result) == 2
        assert result[0]["content"] == "System"
        assert result[1]["content"] == "User"

    def test_system_prepended_to_list(self) -> None:
        messages = [{"role": "user", "content": "User"}]
        result = format_messages(messages, system="New system")
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "New system"

    def test_empty_returns_empty(self) -> None:
        result = format_messages()
        assert result == []


class TestExtractContent:
    """Tests for content extraction."""

    def test_string_content(self) -> None:
        msg = {"role": "user", "content": "Hello"}
        assert extract_content(msg) == "Hello"

    def test_multimodal_content(self) -> None:
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe "},
                {"type": "image_url", "image_url": {"url": "..."}},
                {"type": "text", "text": "this image."},
            ],
        }
        assert extract_content(msg) == "Describe this image."

    def test_missing_content(self) -> None:
        msg = {"role": "assistant", "tool_calls": []}
        assert extract_content(msg) == ""


class TestValidation:
    """Tests for message validation."""

    def test_valid_messages(self) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        errors = validate_messages(messages)
        assert errors == []

    def test_invalid_role(self) -> None:
        messages = [{"role": "invalid", "content": "Hello"}]
        errors = validate_messages(messages)
        assert len(errors) == 1
        assert "invalid role" in errors[0]

    def test_missing_content(self) -> None:
        messages = [{"role": "user"}]
        errors = validate_messages(messages)
        assert len(errors) == 1
        assert "must have 'content'" in errors[0]

    def test_tool_without_call_id(self) -> None:
        messages = [{"role": "tool", "content": "Result"}]
        errors = validate_messages(messages)
        assert len(errors) == 1
        assert "tool_call_id" in errors[0]

    def test_count_messages_by_role(self) -> None:
        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "U2"},
            {"role": "assistant", "content": "A2"},
        ]
        counts = count_messages_by_role(messages)
        assert counts == {"system": 1, "user": 2, "assistant": 2}
