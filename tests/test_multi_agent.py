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
        assert alice_messages[1].get("name") == "Alice"
        assert alice_messages[2]["role"] == "user"
        assert alice_messages[2]["content"] == "Hi Alice!"
        assert alice_messages[2].get("name") == "Bob"

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


class TestAgentValidation:
    """Tests for Agent validation."""

    def test_agent_requires_prompt_or_template(self):
        """Test that agent must have either system_prompt or system_template."""
        with pytest.raises(ValueError, match="must have either"):
            Agent(name="Empty")

    def test_agent_cannot_have_both_prompt_and_template(self):
        """Test that agent cannot have both system_prompt and system_template."""
        with pytest.raises(ValueError, match="cannot have both"):
            Agent(
                name="Both",
                system_prompt="raw",
                system_template="template_name",
            )

    def test_agent_with_private_context(self):
        """Test agent with private context."""
        agent = Agent(
            name="Judge",
            system_prompt="You are a judge.",
            private_context="Score on clarity and evidence.",
        )
        assert agent.private_context == "Score on clarity and evidence."

    def test_agent_with_template(self):
        """Test agent with template."""
        agent = Agent(
            name="Analyst",
            system_template="analyst_persona",
            system_variables={"domain": "finance"},
        )
        assert agent.system_template == "analyst_persona"
        assert agent.system_variables == {"domain": "finance"}
        assert agent.system_prompt is None


class TestVisibleTo:
    """Tests for per-message visibility control."""

    def test_add_message_with_visible_to(self):
        """Test adding a private message."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        chat.add_message(
            "GameMaster", "Secret info", visible_to=["Adversary"]
        )

        assert chat.transcript[0].visible_to == ["Adversary"]

    def test_visible_to_filters_messages(self):
        """Test that visible_to filters messages in agent perspective."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        chat.add_agent(Agent("Alice", system_prompt="You are Alice."))
        chat.add_agent(Agent("Bob", system_prompt="You are Bob."))

        # Public message - both see it
        chat.add_message("Moderator", "Hello everyone!")
        # Private to Alice only
        chat.add_message(
            "Moderator", "Secret for Alice", visible_to=["Alice"]
        )
        # Private to Bob only
        chat.add_message(
            "Moderator", "Secret for Bob", visible_to=["Bob"]
        )

        alice_msgs = chat.get_messages_for("Alice")
        bob_msgs = chat.get_messages_for("Bob")

        # Alice: system + public + her secret = 3
        assert len(alice_msgs) == 3
        assert alice_msgs[1]["content"] == "Hello everyone!"
        assert alice_msgs[2]["content"] == "Secret for Alice"

        # Bob: system + public + his secret = 3
        assert len(bob_msgs) == 3
        assert bob_msgs[1]["content"] == "Hello everyone!"
        assert bob_msgs[2]["content"] == "Secret for Bob"

    def test_visible_to_multiple_agents(self):
        """Test visible_to with multiple agents in the list."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        chat.add_agent(Agent("A", system_prompt="Agent A"))
        chat.add_agent(Agent("B", system_prompt="Agent B"))
        chat.add_agent(Agent("C", system_prompt="Agent C"))

        # Visible to A and B, not C
        chat.add_message("GM", "Shared secret", visible_to=["A", "B"])

        a_msgs = chat.get_messages_for("A")
        b_msgs = chat.get_messages_for("B")
        c_msgs = chat.get_messages_for("C")

        assert len(a_msgs) == 2  # system + message
        assert len(b_msgs) == 2  # system + message
        assert len(c_msgs) == 1  # system only


class TestPrivateContext:
    """Tests for static private_context on agents."""

    def test_private_context_in_messages(self):
        """Test that private_context is injected as system message."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        chat.add_agent(Agent(
            "Judge",
            system_prompt="You are a judge.",
            private_context="Score on: clarity (1-5), evidence (1-5).",
        ))
        chat.add_agent(Agent("Debater", system_prompt="You are a debater."))

        judge_msgs = chat.get_messages_for("Judge")
        debater_msgs = chat.get_messages_for("Debater")

        # Judge sees: system prompt + private context = 2 system messages
        assert judge_msgs[0]["role"] == "system"
        assert judge_msgs[0]["content"] == "You are a judge."
        assert judge_msgs[1]["role"] == "system"
        assert judge_msgs[1]["content"] == "Score on: clarity (1-5), evidence (1-5)."

        # Debater sees only their system prompt
        assert len(debater_msgs) == 1
        assert debater_msgs[0]["content"] == "You are a debater."


class TestStats:
    """Tests for conversation stats."""

    def test_stats_empty(self):
        """Test stats on empty conversation."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        stats = chat.stats
        assert stats["total_messages"] == 0
        assert stats["total_tokens"] == 0
        assert stats["by_agent"] == {}

    def test_stats_with_messages(self):
        """Test stats computation from metadata."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        chat.add_message("Alice", "Hello!", metadata={"tokens": 10, "latency_ms": 100.0})
        chat.add_message("Bob", "Hi!", metadata={"tokens": 8, "latency_ms": 80.0})
        chat.add_message("Alice", "Bye!", metadata={"tokens": 5, "latency_ms": 50.0})

        stats = chat.stats
        assert stats["total_messages"] == 3
        assert stats["total_tokens"] == 23
        assert stats["by_agent"]["Alice"]["messages"] == 2
        assert stats["by_agent"]["Alice"]["total_tokens"] == 15
        assert stats["by_agent"]["Bob"]["messages"] == 1
        assert stats["by_agent"]["Bob"]["total_tokens"] == 8


class TestFormatTranscript:
    """Tests for transcript formatting with new features."""

    def test_format_with_private_annotations(self):
        """Test formatting with include_private flag."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        chat.add_message("GM", "Public info")
        chat.add_message("GM", "Secret", visible_to=["Agent1"])

        formatted = chat.format_transcript(include_private=True)
        assert "(visible_to: Agent1)" in formatted

        # Without include_private, no annotations
        formatted_plain = chat.format_transcript(include_private=False)
        assert "visible_to" not in formatted_plain


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

    @pytest.mark.asyncio
    async def test_name_field_attribution(self):
        """Test that messages use the name field for attribution."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        chat.add_agent(Agent("Alice", "You are Alice."))
        chat.add_agent(Agent("Bob", "You are Bob."))

        chat.add_message("Alice", "Hello Bob!")
        chat.add_message("Bob", "Hi Alice!")

        # From Alice's perspective, Bob's message has name field
        alice_msgs = chat.get_messages_for("Alice")
        bob_msg = alice_msgs[2]  # Bob's message from Alice's POV
        assert bob_msg["role"] == "user"
        assert bob_msg["content"] == "Hi Alice!"  # No [Bob]: prefix
        assert bob_msg.get("name") == "Bob"

        # Alice's own message has her name too
        alice_msg = alice_msgs[1]
        assert alice_msg["role"] == "assistant"
        assert alice_msg.get("name") == "Alice"

    @pytest.mark.asyncio
    async def test_social_simulation_scenario(self):
        """Test a social simulation with private whispers."""
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        chat.add_agent(Agent(
            "Detective",
            system_prompt="You are a detective investigating a mystery.",
        ))
        chat.add_agent(Agent(
            "Adversary",
            system_prompt="You are a player in a mystery game.",
            private_context="You are secretly the culprit.",
        ))

        # Public message
        chat.add_message("GameMaster", "Welcome to the mystery!")

        # Private intel to adversary
        chat.add_message(
            "GameMaster",
            "Secret: the evidence is hidden in the garden.",
            visible_to=["Adversary"],
        )

        # Detective should not see the secret
        det_msgs = chat.get_messages_for("Detective")
        det_contents = [m["content"] for m in det_msgs]
        assert "Secret: the evidence is hidden in the garden." not in det_contents

        # Adversary sees everything including private context and secret
        adv_msgs = chat.get_messages_for("Adversary")
        adv_contents = [m["content"] for m in adv_msgs]
        assert "You are secretly the culprit." in adv_contents
        assert "Secret: the evidence is hidden in the garden." in adv_contents
        assert "Welcome to the mystery!" in adv_contents
