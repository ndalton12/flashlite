"""
Multi-Agent Chat Example
========================

This example demonstrates how to create multiple AI agents that can
converse with each other using Flashlite's MultiAgentChat.

Features demonstrated:
- Creating agents with different personas
- Agents taking turns in conversation
- Round-robin and directed conversations
- Conversation history and transcripts
- Using different models for different agents

To run this example:
    uv run python examples/multi_agent_chat.py

Note: Requires OPENAI_API_KEY environment variable or .env file.
"""

import asyncio

from flashlite import Agent, Flashlite, MultiAgentChat

# ============================================================================
# Example Conversations
# ============================================================================


async def simple_debate_example():
    """Two agents debate a topic."""
    print("\n" + "=" * 60)
    print("Example 1: Simple Debate")
    print("=" * 60)

    client = Flashlite(default_model="gpt-4o-mini")
    chat = MultiAgentChat(client)

    # Create two agents with different perspectives
    chat.add_agent(
        Agent(
            name="Optimist",
            system_prompt=(
                "You are an optimistic futurist who sees the positive potential "
                "in technology. Keep responses concise (2-3 sentences). "
                "Engage with the other speaker's points."
            ),
        )
    )

    chat.add_agent(
        Agent(
            name="Skeptic",
            system_prompt=(
                "You are a thoughtful skeptic who raises important concerns about "
                "technology. Keep responses concise (2-3 sentences). "
                "Engage with the other speaker's points."
            ),
        )
    )

    # Start the conversation with a topic
    chat.add_message("Moderator", "Discuss: Will AI make human programmers obsolete?")

    # Have them debate for 2 rounds
    print("\nDebate topic: Will AI make human programmers obsolete?\n")

    for round_num in range(2):
        print(f"--- Round {round_num + 1} ---\n")

        response1 = await chat.speak("Optimist")
        print(f"Optimist: {response1}\n")

        response2 = await chat.speak("Skeptic")
        print(f"Skeptic: {response2}\n")


async def collaborative_story_example():
    """Multiple agents collaborate on a story."""
    print("\n" + "=" * 60)
    print("Example 2: Collaborative Storytelling")
    print("=" * 60)

    client = Flashlite(default_model="gpt-4o-mini")
    chat = MultiAgentChat(client)

    # Create storytelling agents
    chat.add_agent(
        Agent(
            name="Narrator",
            system_prompt="""You are a storyteller who sets scenes and describes action.
Write 2-3 sentences continuing the story. Focus on setting and atmosphere.""",
        )
    )

    chat.add_agent(
        Agent(
            name="Character Voice",
            system_prompt="""You write dialogue and character thoughts.
Write 2-3 sentences of dialogue or inner monologue. Make characters come alive.""",
        )
    )

    chat.add_agent(
        Agent(
            name="Plot Twister",
            system_prompt="""You introduce unexpected elements and complications.
Add one surprising element or complication in 2-3 sentences.""",
        )
    )

    # Start the story
    chat.add_message(
        "Moderator",
        "Begin a mystery story: A detective receives an anonymous letter...",
    )

    print("\nCollaborative Story:\n")

    # One round of storytelling
    for agent_name in ["Narrator", "Character Voice", "Plot Twister"]:
        response = await chat.speak(agent_name)
        print(f"[{agent_name}]: {response}\n")


async def multi_model_agents_example():
    """Agents using different models."""
    print("\n" + "=" * 60)
    print("Example 3: Multi-Model Agents")
    print("=" * 60)

    client = Flashlite(default_model="gpt-4o-mini")
    chat = MultiAgentChat(client)

    # Create agents with different models
    # Note: In practice you'd use models you have access to
    chat.add_agent(
        Agent(
            name="GPT-4o Mini Agent",
            system_prompt="You are a helpful assistant. Keep responses brief (1-2 sentences).",
            model="gpt-4o-mini",
        )
    )

    chat.add_agent(
        Agent(
            name="GPT-4o Agent",
            system_prompt="You are a thoughtful assistant. Keep responses brief (1-2 sentences).",
            model="gpt-4o-mini",  # Using same model for demo; change to "gpt-4o" if available
        )
    )

    chat.add_message("User", "What makes a good programming language?")

    print("\nMulti-model discussion:\n")

    response1 = await chat.speak("GPT-4o Mini Agent")
    print(f"GPT-4o Mini Agent: {response1}\n")

    response2 = await chat.speak("GPT-4o Agent")
    print(f"GPT-4o Agent: {response2}\n")


async def brainstorming_session_example():
    """Multiple agents brainstorm ideas."""
    print("\n" + "=" * 60)
    print("Example 4: Brainstorming Session")
    print("=" * 60)

    client = Flashlite(default_model="gpt-4o-mini")
    chat = MultiAgentChat(client)

    # Create diverse brainstorming agents
    personas = [
        (
            "Creative",
            "You generate wild, creative ideas. Think outside the box. "
            "One idea per response.",
        ),
        (
            "Practical",
            "You focus on feasible, practical ideas. Consider resources "
            "and constraints. One idea per response.",
        ),
        (
            "Critic",
            "You identify potential issues and improvements. Be constructive. "
            "One critique per response.",
        ),
    ]

    for name, prompt in personas:
        chat.add_agent(Agent(name=name, system_prompt=prompt))

    # Start brainstorming
    chat.add_message(
        "Facilitator",
        "Brainstorm: How can we make code review more engaging for developers?",
    )

    print("\nBrainstorming: Making code review more engaging\n")

    # Each agent contributes
    for name, _ in personas:
        response = await chat.speak(name)
        print(f"{name}: {response}\n")


async def main():
    """Run all multi-agent examples."""
    print("=" * 60)
    print("Flashlite Multi-Agent Chat Examples")
    print("=" * 60)

    await simple_debate_example()
    await collaborative_story_example()
    await multi_model_agents_example()
    await brainstorming_session_example()

    print("\n" + "=" * 60)
    print("All multi-agent examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
