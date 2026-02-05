"""Tests for multi-agent conversation support."""

import pytest

from flashlite import Agent, Flashlite, MultiAgentChat
from flashlite.conversation.multi_agent import ChatMessage


class TestAgent:
    """Tests for Agent dataclass."""

    def test_basic_agent(self):
        """Test creating a basic agent."""
        agent = Agent(name="Alice", system_prompt="You are helpful.")
        assert agent.name == "Alice"
        assert agent.system_prompt == "You are helpful."
        assert agent.model is None

    def test_agent_with_model(self):
        """Test creating agent with specific model."""
        agent = Agent(
            name="Bob",
            system_prompt="You are curious.",
            model="gpt-4o",
        )
        assert agent.name == "Bob"
        assert agent.model == "gpt-4o"


class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_basic_message(self):
        """Test creating a basic message."""
        msg = ChatMessage(agent_name="Alice", content="Hello!")
        assert msg.agent_name == "Alice"
        assert msg.content == "Hello!"
        assert msg.metadata == {}

    def test_message_with_metadata(self):
        """Test message with metadata."""
        msg = ChatMessage(
            agent_name="Bob",
            content="Hi there!",
            metadata={"tokens": 10, "model": "gpt-4o"},
        )
        assert msg.metadata["tokens"] == 10
        assert msg.metadata["model"] == "gpt-4o"


class TestMultiAgentChat:
    """Tests for MultiAgentChat class."""

    def test_init(self):
        """Test initialization."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        assert len(chat.agents) == 0
        assert len(chat.transcript) == 0
        assert chat.agent_names == []

    def test_add_agent(self):
        """Test adding agents."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        agent = Agent(name="Alice", system_prompt="You are helpful.")
        result = chat.add_agent(agent)

        # Test method chaining
        assert result is chat
        assert "Alice" in chat.agents
        assert chat.agent_names == ["Alice"]

    def test_add_multiple_agents(self):
        """Test adding multiple agents with chaining."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        chat.add_agent(Agent("Alice", "You are helpful.")).add_agent(
            Agent("Bob", "You are curious.")
        )

        assert len(chat.agents) == 2
        assert "Alice" in chat.agents
        assert "Bob" in chat.agents

    def test_remove_agent(self):
        """Test removing an agent."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)
        chat.add_agent(Agent("Alice", "You are helpful."))

        removed = chat.remove_agent("Alice")
        assert removed is True
        assert "Alice" not in chat.agents

        # Removing non-existent agent returns False
        removed = chat.remove_agent("Alice")
        assert removed is False

    def test_add_message(self):
        """Test manually adding messages."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        result = chat.add_message("Moderator", "Welcome to the discussion!")

        # Test method chaining
        assert result is chat
        assert len(chat.transcript) == 1
        assert chat.transcript[0].agent_name == "Moderator"
        assert chat.transcript[0].content == "Welcome to the discussion!"

    def test_add_message_with_metadata(self):
        """Test adding message with metadata."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        chat.add_message("User", "Hello!", metadata={"source": "test"})

        assert chat.transcript[0].metadata["source"] == "test"

    @pytest.mark.asyncio
    async def test_speak(self):
        """Test having an agent speak."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        chat.add_agent(Agent("Alice", "You are helpful. Be brief."))
        chat.add_message("User", "Hello!")

        response = await chat.speak("Alice", mock_response="Hello! How can I help?")

        assert response == "Hello! How can I help?"
        assert len(chat.transcript) == 2
        assert chat.transcript[1].agent_name == "Alice"
        assert chat.transcript[1].content == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_speak_unknown_agent_raises(self):
        """Test that speaking with unknown agent raises error."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        with pytest.raises(ValueError, match="Unknown agent"):
            await chat.speak("NonExistent")

    @pytest.mark.asyncio
    async def test_speak_with_additional_context(self):
        """Test speaking with additional context."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        chat.add_agent(Agent("Alice", "You are helpful."))

        response = await chat.speak(
            "Alice",
            additional_context="Please be very brief.",
            mock_response="OK!",
        )

        assert response == "OK!"

    @pytest.mark.asyncio
    async def test_speak_with_model_override(self):
        """Test agent with model override."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        # Agent with specific model
        chat.add_agent(Agent("Alice", "You are helpful.", model="gpt-4o"))

        response = await chat.speak("Alice", mock_response="Hello!")
        assert response == "Hello!"

    @pytest.mark.asyncio
    async def test_round_robin(self):
        """Test round-robin speaking."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        chat.add_agent(Agent("Alice", "Say 'Alice here.'"))
        chat.add_agent(Agent("Bob", "Say 'Bob here.'"))
        chat.add_message("Moderator", "Start!")

        responses = await chat.round_robin(
            rounds=2,
            mock_response="Response",
        )

        # 2 agents * 2 rounds = 4 responses
        assert len(responses) == 4
        # Transcript should have 1 moderator + 4 agent messages
        assert len(chat.transcript) == 5

    @pytest.mark.asyncio
    async def test_speak_sequence(self):
        """Test speaking in a specific sequence."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        chat.add_agent(Agent("Alice", "You are Alice."))
        chat.add_agent(Agent("Bob", "You are Bob."))
        chat.add_agent(Agent("Charlie", "You are Charlie."))

        responses = await chat.speak_sequence(
            ["Bob", "Alice", "Bob"],
            mock_response="Speaking",
        )

        assert len(responses) == 3
        assert chat.transcript[0].agent_name == "Bob"
        assert chat.transcript[1].agent_name == "Alice"
        assert chat.transcript[2].agent_name == "Bob"

    def test_format_transcript(self):
        """Test formatting transcript as string."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        chat.add_message("Alice", "Hello!")
        chat.add_message("Bob", "Hi there!")

        formatted = chat.format_transcript()

        assert "[Alice]:" in formatted
        assert "Hello!" in formatted
        assert "[Bob]:" in formatted
        assert "Hi there!" in formatted

    def test_format_transcript_with_metadata(self):
        """Test formatting transcript with metadata."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        chat.add_message("Alice", "Hello!", metadata={"tokens": 5})

        formatted = chat.format_transcript(include_metadata=True)

        assert "tokens=5" in formatted

    def test_get_messages_for(self):
        """Test getting messages from an agent's perspective."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        chat.add_agent(Agent("Alice", "You are Alice."))
        chat.add_agent(Agent("Bob", "You are Bob."))

        chat.add_message("Alice", "Hello Bob!")
        chat.add_message("Bob", "Hi Alice!")

        alice_messages = chat.get_messages_for("Alice")

        # System + Alice's message (assistant) + Bob's message (user)
        assert len(alice_messages) == 3
        assert alice_messages[0]["role"] == "system"
        assert alice_messages[0]["content"] == "You are Alice."
        assert alice_messages[1]["role"] == "assistant"
        assert alice_messages[1]["content"] == "Hello Bob!"
        assert alice_messages[2]["role"] == "user"
        assert "[Bob]:" in alice_messages[2]["content"]

    def test_get_messages_for_unknown_agent_raises(self):
        """Test that getting messages for unknown agent raises error."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        with pytest.raises(ValueError, match="Unknown agent"):
            chat.get_messages_for("NonExistent")

    def test_clear(self):
        """Test clearing transcript."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        chat.add_agent(Agent("Alice", "You are helpful."))
        chat.add_message("Alice", "Hello!")
        chat.add_message("Bob", "Hi!")

        result = chat.clear()

        # Test method chaining
        assert result is chat
        assert len(chat.transcript) == 0
        # Agents should still be there
        assert "Alice" in chat.agents

    def test_len(self):
        """Test __len__ returns transcript length."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        assert len(chat) == 0

        chat.add_message("Alice", "Hello!")
        assert len(chat) == 1

        chat.add_message("Bob", "Hi!")
        assert len(chat) == 2

    def test_repr(self):
        """Test string representation."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        chat.add_agent(Agent("Alice", "You are helpful."))
        chat.add_agent(Agent("Bob", "You are curious."))
        chat.add_message("Test", "Hello")

        repr_str = repr(chat)

        assert "MultiAgentChat" in repr_str
        assert "Alice" in repr_str
        assert "Bob" in repr_str
        assert "messages=1" in repr_str


class TestMultiAgentChatIntegration:
    """Integration tests for multi-agent conversations."""

    @pytest.mark.asyncio
    async def test_two_agent_conversation(self):
        """Test a simple two-agent conversation."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        chat.add_agent(Agent("Optimist", "You see the bright side. Be brief."))
        chat.add_agent(Agent("Pessimist", "You see problems. Be brief."))

        chat.add_message("Moderator", "Discuss: Is the glass half full?")

        # Optimist speaks first
        response1 = await chat.speak(
            "Optimist",
            mock_response="The glass is definitely half full!",
        )
        assert "half full" in response1

        # Pessimist responds
        response2 = await chat.speak(
            "Pessimist",
            mock_response="Actually, the glass is half empty and leaking.",
        )
        assert "half empty" in response2

        # Verify transcript
        assert len(chat.transcript) == 3
        assert chat.transcript[0].agent_name == "Moderator"
        assert chat.transcript[1].agent_name == "Optimist"
        assert chat.transcript[2].agent_name == "Pessimist"

    @pytest.mark.asyncio
    async def test_message_perspective(self):
        """Test that agents see messages from correct perspective."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        chat.add_agent(Agent("Alice", "You are Alice."))
        chat.add_agent(Agent("Bob", "You are Bob."))

        # Add some messages
        chat.add_message("Alice", "Hello Bob!")
        chat.add_message("Bob", "Hi Alice!")
        chat.add_message("Alice", "How are you?")

        # Get Alice's perspective
        alice_msgs = chat.get_messages_for("Alice")
        # Alice should see her messages as assistant, Bob's as user
        assert alice_msgs[1]["role"] == "assistant"  # Alice's first message
        assert alice_msgs[2]["role"] == "user"  # Bob's message
        assert alice_msgs[3]["role"] == "assistant"  # Alice's second message

        # Get Bob's perspective
        bob_msgs = chat.get_messages_for("Bob")
        # Bob should see Alice's messages as user, his as assistant
        assert bob_msgs[1]["role"] == "user"  # Alice's first message
        assert bob_msgs[2]["role"] == "assistant"  # Bob's message
        assert bob_msgs[3]["role"] == "user"  # Alice's second message
