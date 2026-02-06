"""Multi-agent conversation support for agent-to-agent interactions."""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, overload

from pydantic import BaseModel

from ..core.messages import assistant_message, system_message, user_message
from ..structured import (
    StructuredOutputError,
    format_validation_error_for_retry,
    schema_to_prompt,
    validate_response,
)
from ..types import CompletionResponse

if TYPE_CHECKING:
    from ..client import Flashlite

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

_INVALID_NAME_CHARS = re.compile(r"[\s<|\\/>]+")


def _sanitize_name(name: str) -> str:
    """Sanitize a display name for use in the OpenAI message ``name`` field.

    The API requires names to match ``^[^\\s<|\\\\/>]+$``.  This helper
    replaces any run of invalid characters with ``_`` and strips leading/
    trailing underscores so that human-friendly display names like
    ``"Character Voice"`` become ``"Character_Voice"``.
    """
    return _INVALID_NAME_CHARS.sub("_", name).strip("_")


@dataclass
class Agent:
    """
    An agent with a name, persona, and optional model override.

    Agents can define their system prompt either as a raw string or as a
    Jinja template (rendered at speak-time via the client's TemplateEngine).
    Agents can also have private context that only they see.

    Attributes:
        name: Display name (used in transcript and message attribution)
        system_prompt: The agent's personality/instructions (raw string)
        model: Optional model override (uses MultiAgentChat default if None)
        system_template: Jinja template name (alternative to system_prompt)
        system_variables: Variables for template rendering
        private_context: Static context only this agent sees (injected as system message)

    Examples:
        # Raw system prompt
        Agent(name="Scientist", system_prompt="You are a curious scientist.")

        # Jinja template
        Agent(name="Analyst", system_template="analyst_persona",
              system_variables={"domain": "finance"})

        # With private context
        Agent(name="Judge", system_prompt="You are a debate judge.",
              private_context="Score on: clarity (1-5), evidence (1-5).")
    """

    name: str
    system_prompt: str | None = None
    model: str | None = None
    # Jinja template support (alternative to system_prompt)
    system_template: str | None = None
    system_variables: dict[str, Any] | None = None
    # Private context only this agent sees
    private_context: str | None = None

    def __post_init__(self) -> None:
        if not self.system_prompt and not self.system_template:
            raise ValueError(
                f"Agent '{self.name}' must have either system_prompt or system_template"
            )
        if self.system_prompt and self.system_template:
            raise ValueError(
                f"Agent '{self.name}' cannot have both system_prompt and system_template"
            )


@dataclass
class ChatMessage:
    """A message in the multi-agent conversation.

    Attributes:
        agent_name: Who sent this message
        content: The message content
        metadata: Additional metadata (tokens, latency, model, etc.)
        visible_to: If set, only these agents can see this message.
                   None means all agents can see it.
    """

    agent_name: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    visible_to: list[str] | None = None


class MultiAgentChat:
    """
    Manages conversations between multiple AI agents.

    Integrates with flashlite's templating, logging, structured outputs,
    and observability features.

    Key features:
    - Multiple agents with different personas and optionally different models
    - Jinja template support for agent system prompts
    - Per-message visibility control (private whispers to specific agents)
    - Structured output support via Pydantic models (per-turn, flexible)
    - Automatic context building from each agent's perspective
    - Conversation-level logging and per-agent stats
    - Round-robin or directed turn-taking

    How it works:
    - Each agent has a system prompt (raw or Jinja template) defining their persona
    - When an agent speaks, they see:
      - Their own previous messages as "assistant" role
      - Other agents' messages as "user" role with the ``name`` field for attribution
      - Only messages they are allowed to see (filtered by ``visible_to``)
    - Private context on an agent is injected as a system message only they see

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

        # Whisper private info to one agent
        chat.add_message("Moderator", "Secret: focus on healthcare jobs.",
                        visible_to=["Optimist"])

        # Have agents take turns
        await chat.speak("Optimist")
        await chat.speak("Skeptic")

        # Structured output from a judge
        class Score(BaseModel):
            winner: str
            reasoning: str

        result = await chat.speak("Judge", response_model=Score)

        # Round-robin for structured turns
        await chat.round_robin(rounds=2)

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

    # -- Agent management ------------------------------------------------

    def add_agent(self, agent: Agent) -> "MultiAgentChat":
        """
        Add an agent to the chat.

        Args:
            agent: Agent to add

        Returns:
            Self for method chaining

        Example:
            chat.add_agent(Agent("Alice", system_prompt="You are helpful."))
                .add_agent(Agent("Bob", system_prompt="You are curious."))
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

    # -- Message injection -----------------------------------------------

    def add_message(
        self,
        agent_name: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        visible_to: list[str] | None = None,
    ) -> "MultiAgentChat":
        """
        Manually add a message to the transcript.

        Useful for injecting moderator prompts, user input, or private
        whispers to specific agents.

        Args:
            agent_name: Name to attribute the message to
            content: Message content
            metadata: Optional metadata to attach
            visible_to: If set, only these agents can see this message.
                       None means all agents see it.

        Returns:
            Self for method chaining

        Examples:
            # Public message everyone sees
            chat.add_message("Moderator", "New topic: climate change.")

            # Private whisper only the Adversary sees
            chat.add_message("GameMaster", "Secret: the key is in the library.",
                            visible_to=["Adversary"])
        """
        self._transcript.append(
            ChatMessage(
                agent_name=agent_name,
                content=content,
                metadata=metadata or {},
                visible_to=visible_to,
            )
        )
        logger.debug(
            "Message injected from '%s'%s",
            agent_name,
            f" (visible_to={visible_to})" if visible_to else "",
        )
        return self

    # -- Speaking --------------------------------------------------------

    @overload
    async def speak(
        self,
        agent_name: str,
        *,
        additional_context: str | None = ...,
        response_model: None = ...,
        structured_retries: int = ...,
        visible_to: list[str] | None = ...,
        **kwargs: Any,
    ) -> str: ...

    @overload
    async def speak(
        self,
        agent_name: str,
        *,
        additional_context: str | None = ...,
        response_model: type[T] = ...,
        structured_retries: int = ...,
        visible_to: list[str] | None = ...,
        **kwargs: Any,
    ) -> T: ...

    async def speak(
        self,
        agent_name: str,
        *,
        additional_context: str | None = None,
        response_model: type[T] | None = None,
        structured_retries: int = 1,
        visible_to: list[str] | None = None,
        **kwargs: Any,
    ) -> str | T:
        """
        Have an agent respond to the conversation.

        The agent sees the full conversation history from their perspective:
        - Their own previous messages appear as "assistant" messages
        - Other agents' messages appear as "user" messages with name attribution
        - Messages with ``visible_to`` set are filtered by visibility

        Args:
            agent_name: Name of the agent to speak
            additional_context: Optional extra context/instruction for this turn
            response_model: Pydantic model class for structured output parsing.
                          When provided, returns a validated model instance.
                          Can change per call for flexible per-turn schemas.
            structured_retries: Number of retries for structured output validation
            visible_to: If set, only these agents see this agent's response.
                       None means all agents see it.
            **kwargs: Additional kwargs passed to client.complete()

        Returns:
            The agent's response content (str), or a validated Pydantic model
            instance if response_model is provided.

        Raises:
            ValueError: If agent_name is not found
            StructuredOutputError: If structured output validation fails
                after all retries are exhausted
        """
        if agent_name not in self._agents:
            raise ValueError(
                f"Unknown agent: {agent_name}. "
                f"Available agents: {list(self._agents.keys())}"
            )

        agent = self._agents[agent_name]
        start_time = time.perf_counter()

        # Build messages from this agent's perspective
        messages = self._build_messages_for(agent)

        # Add any additional context as a user message
        if additional_context:
            messages.append(user_message(additional_context))

        # Handle structured output: inject schema into system prompt
        extra_kwargs = dict(kwargs)
        if response_model is not None:
            messages, extra_kwargs = self._inject_schema(
                messages, extra_kwargs, response_model, agent
            )

        # Make completion (without response_model so we get CompletionResponse
        # and can store raw content in the transcript)
        response: CompletionResponse = await self._client.complete(
            model=agent.model or self._default_model,
            messages=messages,
            **extra_kwargs,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Record in transcript with metadata
        self._transcript.append(
            ChatMessage(
                agent_name=agent_name,
                content=response.content,
                metadata={
                    "model": response.model,
                    "tokens": response.usage.total_tokens if response.usage else None,
                    "input_tokens": (
                        response.usage.input_tokens if response.usage else None
                    ),
                    "output_tokens": (
                        response.usage.output_tokens if response.usage else None
                    ),
                    "latency_ms": round(latency_ms, 1),
                },
                visible_to=visible_to,
            )
        )

        logger.info(
            "%s spoke (model=%s, tokens=%s, %.1fms)%s",
            agent_name,
            response.model,
            response.usage.total_tokens if response.usage else "N/A",
            latency_ms,
            f" [visible_to={visible_to}]" if visible_to else "",
        )

        # Validate structured output if requested
        if response_model is not None:
            return self._validate_structured(
                response=response,
                response_model=response_model,
                messages=messages,
                extra_kwargs=extra_kwargs,
                agent=agent,
                structured_retries=structured_retries,
                visible_to=visible_to,
            )

        return response.content

    # -- Internal helpers ------------------------------------------------

    def _inject_schema(
        self,
        messages: list[dict[str, Any]],
        extra_kwargs: dict[str, Any],
        response_model: type[BaseModel],
        agent: Agent,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Inject structured output schema into the system prompt and kwargs."""
        schema_prompt = schema_to_prompt(response_model)

        # Append schema to the system message
        if messages and messages[0].get("role") == "system":
            messages[0] = {
                **messages[0],
                "content": messages[0]["content"] + "\n\n" + schema_prompt,
            }
        else:
            messages.insert(0, system_message(schema_prompt))

        # Enable JSON mode for supported providers
        if "response_format" not in extra_kwargs:
            resolved_model = (agent.model or self._default_model or "").lower()
            if any(
                p in resolved_model
                for p in ["gpt-4", "gpt-3.5", "claude", "gemini", "mistral"]
            ):
                extra_kwargs["response_format"] = {"type": "json_object"}

        return messages, extra_kwargs

    async def _validate_structured(
        self,
        response: CompletionResponse,
        response_model: type[T],
        messages: list[dict[str, Any]],
        extra_kwargs: dict[str, Any],
        agent: Agent,
        structured_retries: int,
        visible_to: list[str] | None,
    ) -> T:
        """Validate structured output with retry support."""
        last_error: StructuredOutputError | None = None
        current_messages = list(messages)

        for attempt in range(structured_retries + 1):
            try:
                return validate_response(response, response_model)
            except StructuredOutputError as e:
                last_error = e
                logger.warning(
                    "%s structured output validation failed (attempt %d): %s",
                    agent.name,
                    attempt + 1,
                    e,
                )
                if attempt < structured_retries:
                    # Ask the model to fix its response
                    error_feedback = format_validation_error_for_retry(e)
                    current_messages.append(assistant_message(response.content))
                    current_messages.append(user_message(error_feedback))

                    response = await self._client.complete(
                        model=agent.model or self._default_model,
                        messages=current_messages,
                        **extra_kwargs,
                    )
                    # Update transcript with corrected response
                    self._transcript[-1] = ChatMessage(
                        agent_name=agent.name,
                        content=response.content,
                        metadata=self._transcript[-1].metadata,
                        visible_to=visible_to,
                    )

        raise last_error  # type: ignore[misc]

    def _resolve_system_prompt(self, agent: Agent) -> str:
        """
        Resolve an agent's system prompt from raw string or Jinja template.

        Args:
            agent: The agent to resolve the prompt for

        Returns:
            The rendered system prompt string

        Raises:
            ValueError: If template engine is not configured
        """
        if agent.system_template:
            engine = self._client.template_engine
            if engine is None:
                raise ValueError(
                    f"Agent '{agent.name}' uses system_template but no template "
                    "engine is configured. Pass template_dir to the Flashlite client "
                    "or call client.register_template()."
                )
            return engine.render(agent.system_template, agent.system_variables)
        return agent.system_prompt or ""

    def _build_messages_for(self, agent: Agent) -> list[dict[str, Any]]:
        """
        Build the message history from a specific agent's perspective.

        - System prompt (from raw string or Jinja template)
        - Private context (if any, as an additional system message)
        - Transcript messages filtered by visibility:
          - Agent's own messages become "assistant" role with ``name`` field
          - Other agents' messages become "user" role with ``name`` field
        """
        messages: list[dict[str, Any]] = []

        # System prompt for this agent
        prompt = self._resolve_system_prompt(agent)
        messages.append(system_message(prompt))

        # Private context (static, only this agent sees)
        if agent.private_context:
            messages.append(system_message(agent.private_context))

        # Conversation history, filtered by visibility
        for msg in self._transcript:
            # Check visibility
            if msg.visible_to is not None and agent.name not in msg.visible_to:
                continue

            if msg.agent_name == agent.name:
                # Agent's own previous messages
                messages.append(
                    assistant_message(msg.content, name=_sanitize_name(agent.name))
                )
            else:
                # Other agents'/sources' messages with name attribution
                messages.append(
                    user_message(msg.content, name=_sanitize_name(msg.agent_name))
                )

        return messages

    # -- Batch speaking --------------------------------------------------

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
        responses: list[str] = []
        agent_names = list(self._agents.keys())

        for round_num in range(1, rounds + 1):
            logger.info(
                "Round %d/%d started (agents: %s)",
                round_num,
                rounds,
                ", ".join(agent_names),
            )
            for name in agent_names:
                response = await self.speak(name, **kwargs)
                responses.append(response)
            logger.info("Round %d/%d complete", round_num, rounds)

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
        responses: list[str] = []
        for name in agent_sequence:
            response = await self.speak(name, **kwargs)
            responses.append(response)
        return responses

    # -- Transcript access -----------------------------------------------

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

    @property
    def stats(self) -> dict[str, Any]:
        """
        Get per-agent statistics from the conversation.

        Returns a dict with total and per-agent breakdowns of tokens,
        latency, and message counts.
        """
        agent_stats: dict[str, dict[str, Any]] = {}
        total_tokens = 0
        total_messages = 0

        for msg in self._transcript:
            name = msg.agent_name
            if name not in agent_stats:
                agent_stats[name] = {
                    "messages": 0,
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_latency_ms": 0.0,
                }
            stats = agent_stats[name]
            stats["messages"] += 1
            total_messages += 1

            tokens = msg.metadata.get("tokens")
            if tokens is not None:
                stats["total_tokens"] += tokens
                total_tokens += tokens

            input_t = msg.metadata.get("input_tokens")
            if input_t is not None:
                stats["input_tokens"] += input_t

            output_t = msg.metadata.get("output_tokens")
            if output_t is not None:
                stats["output_tokens"] += output_t

            latency = msg.metadata.get("latency_ms")
            if latency is not None:
                stats["total_latency_ms"] += latency

        return {
            "total_messages": total_messages,
            "total_tokens": total_tokens,
            "by_agent": agent_stats,
        }

    def format_transcript(
        self,
        include_metadata: bool = False,
        include_private: bool = False,
    ) -> str:
        """
        Format the transcript as a readable string.

        Args:
            include_metadata: Whether to include metadata like tokens used
            include_private: Whether to show visibility annotations

        Returns:
            Formatted transcript string
        """
        lines: list[str] = []
        for msg in self._transcript:
            header = f"[{msg.agent_name}]"
            if include_private and msg.visible_to is not None:
                header += f" (visible_to: {', '.join(msg.visible_to)})"
            header += ":"
            lines.append(header)
            # Indent content for readability
            for line in msg.content.split("\n"):
                lines.append(f"  {line}")
            if include_metadata and msg.metadata:
                meta_str = ", ".join(
                    f"{k}={v}" for k, v in msg.metadata.items() if v is not None
                )
                if meta_str:
                    lines.append(f"  ({meta_str})")
            lines.append("")
        return "\n".join(lines)

    def get_messages_for(self, agent_name: str) -> list[dict[str, Any]]:
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
            f"MultiAgentChat(agents={list(self._agents.keys())}, "
            f"messages={len(self._transcript)})"
        )
