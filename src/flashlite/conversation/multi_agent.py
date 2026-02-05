"""Multi-agent conversation support for agent-to-agent interactions."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..types import CompletionResponse

if TYPE_CHECKING:
    from ..client import Flashlite


@dataclass
class Agent:
    """
    An agent with a name, persona, and optional model override.

    Attributes:
        name: Display name for the agent (used in transcript and message attribution)
        system_prompt: The agent's personality, instructions, and behavior guidelines
        model: Optional model override (uses MultiAgentChat default if None)

    Example:
        agent = Agent(
            name="Scientist",
            system_prompt="You are a curious scientist who loves experiments.",
            model="gpt-4o",  # Optional: use specific model for this agent
        )
    """

    name: str
    system_prompt: str
    model: str | None = None


@dataclass
class ChatMessage:
    """A message in the multi-agent conversation."""

    agent_name: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MultiAgentChat:
    """
    Manages conversations between multiple AI agents.

    This class enables agent-to-agent conversations where multiple AI agents
    can discuss, debate, or collaborate. Each agent maintains its own persona
    and sees the conversation from its perspective.

    Key features:
    - Multiple agents with different personas and optionally different models
    - Automatic context building from each agent's perspective
    - Round-robin or directed turn-taking
    - Full conversation transcript with metadata
    - Support for injecting external messages (moderator, user input)

    How it works:
    - Each agent has a system prompt defining their persona
    - When an agent speaks, they see:
      - Their own previous messages as "assistant" role
      - Other agents' messages as "user" role with name attribution
    - This creates natural back-and-forth conversation

    Example:
        client = Flashlite(default_model="gpt-4o-mini")
        chat = MultiAgentChat(client)

        # Add agents with different personas
        chat.add_agent(Agent(
            name="Optimist",
            system_prompt="You see the positive side of everything. Be concise."
        ))
        chat.add_agent(Agent(
            name="Skeptic",
            system_prompt="You question assumptions. Be concise."
        ))

        # Start with a topic
        chat.add_message("Moderator", "Discuss: Will AI help or hurt jobs?")

        # Have agents take turns
        await chat.speak("Optimist")  # Optimist responds
        await chat.speak("Skeptic")   # Skeptic responds to Optimist
        await chat.speak("Optimist")  # Continue the debate

        # Or use round-robin for structured turns
        await chat.round_robin(rounds=2)

        # Get formatted transcript
        print(chat.format_transcript())
    """

    def __init__(
        self,
        client: "Flashlite",
        default_model: str | None = None,
    ):
        """
        Initialize a multi-agent chat.

        Args:
            client: Flashlite client for making completions
            default_model: Default model for agents (uses client default if None)
        """
        self._client = client
        self._default_model = default_model
        self._agents: dict[str, Agent] = {}
        self._transcript: list[ChatMessage] = []

    def add_agent(self, agent: Agent) -> "MultiAgentChat":
        """
        Add an agent to the chat.

        Args:
            agent: Agent to add

        Returns:
            Self for method chaining

        Example:
            chat.add_agent(Agent("Alice", "You are helpful."))
                .add_agent(Agent("Bob", "You are curious."))
        """
        self._agents[agent.name] = agent
        return self

    def remove_agent(self, agent_name: str) -> bool:
        """
        Remove an agent from the chat.

        Args:
            agent_name: Name of agent to remove

        Returns:
            True if agent was removed, False if not found
        """
        if agent_name in self._agents:
            del self._agents[agent_name]
            return True
        return False

    def add_message(
        self,
        agent_name: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> "MultiAgentChat":
        """
        Manually add a message to the transcript.

        Useful for:
        - Injecting moderator or facilitator prompts
        - Adding user input to the conversation
        - Simulating agent messages for testing

        Args:
            agent_name: Name to attribute the message to
            content: Message content
            metadata: Optional metadata to attach

        Returns:
            Self for method chaining
        """
        self._transcript.append(
            ChatMessage(
                agent_name=agent_name,
                content=content,
                metadata=metadata or {},
            )
        )
        return self

    async def speak(
        self,
        agent_name: str,
        additional_context: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Have an agent respond to the conversation.

        The agent sees the full conversation history from their perspective:
        - Their own previous messages appear as "assistant" messages
        - Other agents' messages appear as "user" messages with name attribution

        Args:
            agent_name: Name of the agent to speak
            additional_context: Optional extra context/instruction for this turn
            **kwargs: Additional kwargs passed to client.complete()

        Returns:
            The agent's response content

        Raises:
            ValueError: If agent_name is not found
        """
        if agent_name not in self._agents:
            raise ValueError(
                f"Unknown agent: {agent_name}. Available agents: {list(self._agents.keys())}"
            )

        agent = self._agents[agent_name]

        # Build messages from this agent's perspective
        messages = self._build_messages_for(agent)

        # Add any additional context as a user message
        if additional_context:
            messages.append({"role": "user", "content": additional_context})

        # Make completion
        response: CompletionResponse = await self._client.complete(
            model=agent.model or self._default_model,
            messages=messages,
            **kwargs,
        )

        # Record in transcript with metadata
        self._transcript.append(
            ChatMessage(
                agent_name=agent_name,
                content=response.content,
                metadata={
                    "model": response.model,
                    "tokens": response.usage.total_tokens if response.usage else None,
                },
            )
        )

        return response.content

    def _build_messages_for(self, agent: Agent) -> list[dict[str, str]]:
        """
        Build the message history from a specific agent's perspective.

        The agent's own messages become "assistant" role (what they said).
        Other agents' messages become "user" role with speaker attribution.
        """
        messages: list[dict[str, str]] = []

        # System prompt for this agent
        messages.append({"role": "system", "content": agent.system_prompt})

        # Add conversation history
        for msg in self._transcript:
            if msg.agent_name == agent.name:
                # Agent's own previous messages
                messages.append({"role": "assistant", "content": msg.content})
            else:
                # Other agents'/sources' messages - prefix with speaker name
                messages.append({"role": "user", "content": f"[{msg.agent_name}]: {msg.content}"})

        return messages

    async def round_robin(
        self,
        rounds: int = 1,
        **kwargs: Any,
    ) -> list[str]:
        """
        Have all agents speak in turn for the specified number of rounds.

        Each agent speaks once per round, in the order they were added.

        Args:
            rounds: Number of complete rounds (each agent speaks once per round)
            **kwargs: Additional kwargs passed to speak()

        Returns:
            List of all responses in order
        """
        responses = []
        agent_names = list(self._agents.keys())

        for _ in range(rounds):
            for name in agent_names:
                response = await self.speak(name, **kwargs)
                responses.append(response)

        return responses

    async def speak_sequence(
        self,
        agent_sequence: list[str],
        **kwargs: Any,
    ) -> list[str]:
        """
        Have agents speak in a specific sequence.

        Args:
            agent_sequence: List of agent names in desired speaking order
            **kwargs: Additional kwargs passed to speak()

        Returns:
            List of responses in order
        """
        responses = []
        for name in agent_sequence:
            response = await self.speak(name, **kwargs)
            responses.append(response)
        return responses

    @property
    def transcript(self) -> list[ChatMessage]:
        """Get a copy of the conversation transcript."""
        return list(self._transcript)

    @property
    def agents(self) -> dict[str, Agent]:
        """Get the registered agents."""
        return dict(self._agents)

    @property
    def agent_names(self) -> list[str]:
        """Get list of agent names."""
        return list(self._agents.keys())

    def format_transcript(self, include_metadata: bool = False) -> str:
        """
        Format the transcript as a readable string.

        Args:
            include_metadata: Whether to include metadata like tokens used

        Returns:
            Formatted transcript string
        """
        lines = []
        for msg in self._transcript:
            lines.append(f"[{msg.agent_name}]:")
            # Indent content for readability
            for line in msg.content.split("\n"):
                lines.append(f"  {line}")
            if include_metadata and msg.metadata:
                meta_str = ", ".join(f"{k}={v}" for k, v in msg.metadata.items() if v)
                if meta_str:
                    lines.append(f"  ({meta_str})")
            lines.append("")
        return "\n".join(lines)

    def get_messages_for(self, agent_name: str) -> list[dict[str, str]]:
        """
        Get the messages list as a specific agent would see it.

        Useful for debugging or custom processing.

        Args:
            agent_name: Name of agent to get perspective for

        Returns:
            Messages list from that agent's perspective
        """
        if agent_name not in self._agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        return self._build_messages_for(self._agents[agent_name])

    def clear(self) -> "MultiAgentChat":
        """
        Clear the conversation transcript.

        Does not remove agents.

        Returns:
            Self for method chaining
        """
        self._transcript = []
        return self

    def __len__(self) -> int:
        """Number of messages in transcript."""
        return len(self._transcript)

    def __repr__(self) -> str:
        return (
            f"MultiAgentChat(agents={list(self._agents.keys())}, messages={len(self._transcript)})"
        )
