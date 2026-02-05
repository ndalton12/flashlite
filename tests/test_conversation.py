"""Tests for conversation management functionality."""

from pathlib import Path

from pydantic import BaseModel

from flashlite import Flashlite
from flashlite.conversation import (
    ContextLimits,
    ContextManager,
    Conversation,
    ConversationState,
    check_context_fit,
    estimate_messages_tokens,
    estimate_tokens,
    truncate_messages,
)


# Test Pydantic model for structured outputs
class SimpleResponse(BaseModel):
    answer: str
    confidence: float


class TestConversation:
    """Tests for Conversation class."""

    async def test_basic_conversation(self):
        """Should handle basic multi-turn conversation."""
        client = Flashlite(default_model="gpt-4o")
        conv = Conversation(client, system="You are helpful.")

        # First turn
        response1 = await conv.say("Hello", mock_response="Hi there!")
        assert response1.content == "Hi there!"
        assert conv.turn_count == 2  # user + assistant

        # Second turn
        response2 = await conv.say("How are you?", mock_response="I'm great!")
        assert response2.content == "I'm great!"
        assert conv.turn_count == 4

    async def test_conversation_with_model_override(self):
        """Should allow model override per turn."""
        client = Flashlite(default_model="gpt-4o")
        conv = Conversation(client)

        # Use different model for one turn
        response = await conv.say(
            "Hello",
            model="gpt-4o-mini",
            mock_response="Hi from mini!",
        )
        assert response.content == "Hi from mini!"

    async def test_conversation_with_structured_output(self):
        """Should support structured outputs in conversation."""
        client = Flashlite(default_model="gpt-4o")
        conv = Conversation(client)

        result = await conv.say(
            "What is 2+2?",
            response_model=SimpleResponse,
            mock_response='{"answer": "4", "confidence": 0.99}',
        )

        assert isinstance(result, SimpleResponse)
        assert result.answer == "4"
        assert result.confidence == 0.99

    async def test_conversation_fork(self):
        """Should create independent branch."""
        client = Flashlite(default_model="gpt-4o")
        conv = Conversation(client, system="You are helpful.")

        # Build some history
        await conv.say("Hello", mock_response="Hi!")

        # Fork
        branch = conv.fork()
        assert branch.turn_count == conv.turn_count

        # Diverge
        await conv.say("Option A", mock_response="A response")
        await branch.say("Option B", mock_response="B response")

        # Should have different histories now
        assert conv.turn_count == 4
        assert branch.turn_count == 4
        assert conv.turns[-1].content != branch.turns[-1].content

    async def test_conversation_clear(self):
        """Should clear history but keep system prompt."""
        client = Flashlite(default_model="gpt-4o")
        conv = Conversation(client, system="You are helpful.")

        await conv.say("Hello", mock_response="Hi!")
        assert conv.turn_count == 2

        conv.clear()
        assert conv.turn_count == 0
        assert conv.system == "You are helpful."

    async def test_conversation_rollback(self):
        """Should remove last n turns."""
        client = Flashlite(default_model="gpt-4o")
        conv = Conversation(client)

        await conv.say("Hello", mock_response="Hi!")
        await conv.say("How are you?", mock_response="Good!")
        assert conv.turn_count == 4

        removed = conv.rollback(2)
        assert len(removed) == 2
        assert conv.turn_count == 2

    async def test_conversation_max_turns(self):
        """Should limit history size."""
        client = Flashlite(default_model="gpt-4o")
        conv = Conversation(client, max_turns=2)

        await conv.say("Turn 1", mock_response="Response 1")
        await conv.say("Turn 2", mock_response="Response 2")
        await conv.say("Turn 3", mock_response="Response 3")

        # Should have kept only the most recent turns
        assert conv.turn_count <= 4  # 2 turns * 2 messages each

    def test_conversation_messages_property(self):
        """Should build messages list correctly."""
        client = Flashlite(default_model="gpt-4o")
        conv = Conversation(client, system="Be helpful.")

        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi!")

        messages = conv.messages
        assert len(messages) == 3  # system + user + assistant
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    def test_conversation_save_load(self, tmp_path: Path):
        """Should save and load conversation state."""
        client = Flashlite(default_model="gpt-4o")
        conv = Conversation(client, system="Be helpful.", model="gpt-4o")

        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi!")

        # Save
        save_path = tmp_path / "conversation.json"
        conv.save(save_path)

        # Load
        loaded = Conversation.load(client, save_path)

        assert loaded.system == conv.system
        assert loaded.model == conv.model
        assert loaded.turn_count == conv.turn_count
        assert loaded.id == conv.id

    def test_conversation_state_serialization(self):
        """Should serialize state to dict."""
        client = Flashlite(default_model="gpt-4o")
        conv = Conversation(client, system="Test", metadata={"key": "value"})

        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi!")

        state = conv.get_state()
        data = state.to_dict()

        assert "id" in data
        assert data["system_prompt"] == "Test"
        assert len(data["turns"]) == 2
        assert data["metadata"] == {"key": "value"}

        # Reconstruct
        restored = ConversationState.from_dict(data)
        assert restored.system_prompt == state.system_prompt
        assert len(restored.turns) == len(state.turns)


class TestClientConversationFactory:
    """Tests for client.conversation() factory method."""

    def test_create_conversation(self):
        """Should create conversation bound to client."""
        client = Flashlite(default_model="gpt-4o")
        conv = client.conversation(system="You are helpful.")

        assert isinstance(conv, Conversation)
        assert conv.system == "You are helpful."

    def test_create_conversation_with_model(self):
        """Should allow model override."""
        client = Flashlite(default_model="gpt-4o")
        conv = client.conversation(model="gpt-4o-mini")

        assert conv.model == "gpt-4o-mini"

    def test_create_context_manager(self):
        """Should create context manager."""
        client = Flashlite(default_model="gpt-4o")
        ctx = client.context_manager()

        assert isinstance(ctx, ContextManager)
        assert ctx.model == "gpt-4o"


class TestContextManagement:
    """Tests for context window management."""

    def test_estimate_tokens(self):
        """Should estimate token count."""
        text = "Hello, world!"
        tokens = estimate_tokens(text)
        assert tokens > 0
        assert tokens < len(text)  # Tokens should be less than chars

    def test_estimate_messages_tokens(self):
        """Should estimate tokens for message list."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        tokens = estimate_messages_tokens(messages)
        assert tokens > 0

    def test_context_limits_for_model(self):
        """Should return appropriate limits for models."""
        gpt4o = ContextLimits.for_model("gpt-4o")
        assert gpt4o.max_tokens == 128_000

        claude = ContextLimits.for_model("claude-3-5-sonnet-20241022")
        assert claude.max_tokens == 200_000

        unknown = ContextLimits.for_model("unknown-model")
        assert unknown.max_tokens > 0  # Should have a default

    def test_check_context_fit(self):
        """Should check if messages fit."""
        messages = [{"role": "user", "content": "Hello!"}]
        fits, info = check_context_fit(messages, "gpt-4o")

        assert fits is True
        assert "estimated_tokens" in info
        assert "max_tokens" in info

    def test_check_context_fit_too_large(self):
        """Should detect when messages are too large."""
        # Create a very large message
        large_content = "x" * 1_000_000
        messages = [{"role": "user", "content": large_content}]

        fits, info = check_context_fit(messages, "gpt-4", max_response_tokens=4096)
        assert fits is False
        assert "warning" in info

    def test_truncate_messages(self):
        """Should truncate to fit."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "A moderately long message that takes up some tokens"},
            {"role": "assistant", "content": "Another moderately long response from assistant"},
            {"role": "user", "content": "Yet another user message with content"},
            {"role": "assistant", "content": "And another assistant response here"},
        ]

        # Truncate to small limit - should remove some messages
        truncated = truncate_messages(messages, max_tokens=15, keep_system=True)

        # Should keep system message
        assert truncated[0]["role"] == "system"
        # Should have fewer messages
        assert len(truncated) < len(messages)

    def test_truncate_messages_keeps_system(self):
        """Should keep system message when truncating."""
        messages = [
            {"role": "system", "content": "Important system prompt"},
            {"role": "user", "content": "User message"},
        ]

        truncated = truncate_messages(messages, max_tokens=20, keep_system=True)
        assert any(m["role"] == "system" for m in truncated)

    def test_context_manager_prepare(self):
        """Should prepare messages for completion."""
        ctx = ContextManager(model="gpt-4o")
        messages = [{"role": "user", "content": "Hello!"}]

        prepared = ctx.prepare(messages)
        assert prepared == messages  # Should not modify small messages

    def test_context_manager_auto_truncate(self):
        """Should auto-truncate when needed."""
        ctx = ContextManager(
            model="gpt-4",  # Smaller context (8192)
            auto_truncate=True,
        )

        # Create multiple messages, some of which will be truncated
        messages = [
            {"role": "user", "content": "First message " * 100},
            {"role": "assistant", "content": "First response " * 100},
            {"role": "user", "content": "Second message " * 100},
            {"role": "assistant", "content": "Second response " * 100},
            {"role": "user", "content": "Third message " * 100},
            {"role": "assistant", "content": "Third response " * 100},
            {"role": "user", "content": "Fourth message that we want to keep"},
        ]

        # Should truncate without raising, keeping recent messages
        prepared = ctx.prepare(messages)
        assert len(prepared) > 0
        assert len(prepared) <= len(messages)  # Should have truncated some
        # Should keep the most recent message
        assert "Fourth message" in prepared[-1]["content"]


class TestCompleteManyMixedModels:
    """Tests for complete_many with mixed models."""

    async def test_complete_many_different_models(self):
        """Should handle different models in same batch."""
        client = Flashlite(default_model="gpt-4o")

        responses = await client.complete_many([
            {
                "model": "gpt-4o",
                "messages": "Hello from GPT",
                "mock_response": "GPT response",
            },
            {
                "model": "gpt-4o-mini",
                "messages": "Hello from mini",
                "mock_response": "Mini response",
            },
        ])

        assert len(responses) == 2
        assert responses[0].content == "GPT response"
        assert responses[1].content == "Mini response"

    async def test_complete_many_with_structured_outputs(self):
        """Should handle structured outputs in batch."""
        client = Flashlite(default_model="gpt-4o")

        responses = await client.complete_many([
            {
                "messages": "Question 1",
                "response_model": SimpleResponse,
                "mock_response": '{"answer": "A", "confidence": 0.9}',
            },
            {
                "messages": "Question 2",
                "response_model": SimpleResponse,
                "mock_response": '{"answer": "B", "confidence": 0.8}',
            },
        ])

        assert len(responses) == 2
        assert isinstance(responses[0], SimpleResponse)
        assert isinstance(responses[1], SimpleResponse)
        assert responses[0].answer == "A"
        assert responses[1].answer == "B"

    async def test_complete_many_concurrency(self):
        """Should respect max_concurrency."""
        client = Flashlite(default_model="gpt-4o")

        # Create many requests
        requests = [
            {"messages": f"Request {i}", "mock_response": f"Response {i}"}
            for i in range(10)
        ]

        responses = await client.complete_many(requests, max_concurrency=3)

        assert len(responses) == 10
        for i, response in enumerate(responses):
            assert response.content == f"Response {i}"
