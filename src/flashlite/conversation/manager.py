"""Conversation management for multi-turn interactions."""

import copy
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from ..types import CompletionResponse, Message, Messages

if TYPE_CHECKING:
    from ..client import Flashlite

T = TypeVar("T", bound=BaseModel)


@dataclass
class Turn:
    """A single turn in a conversation."""

    role: str
    content: str
    model: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> Message:
        """Convert to a message dict."""
        return {"role": self.role, "content": self.content}


@dataclass
class ConversationState:
    """Serializable state of a conversation."""

    id: str
    system_prompt: str | None
    turns: list[Turn]
    default_model: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "system_prompt": self.system_prompt,
            "turns": [
                {
                    "role": t.role,
                    "content": t.content,
                    "model": t.model,
                    "timestamp": t.timestamp,
                    "metadata": t.metadata,
                }
                for t in self.turns
            ],
            "default_model": self.default_model,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationState":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            system_prompt=data.get("system_prompt"),
            turns=[
                Turn(
                    role=t["role"],
                    content=t["content"],
                    model=t.get("model"),
                    timestamp=t.get("timestamp", ""),
                    metadata=t.get("metadata", {}),
                )
                for t in data.get("turns", [])
            ],
            default_model=data.get("default_model"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


class Conversation:
    """
    Manages multi-turn conversations with LLMs.

    Features:
    - Automatic message history management
    - System prompt support
    - Model switching mid-conversation
    - Branching/forking for tree-of-thought patterns
    - Serialization for persistence

    Example:
        client = Flashlite(default_model="gpt-4o")
        conv = Conversation(client, system="You are a helpful assistant.")

        # Multi-turn conversation
        response1 = await conv.say("What is Python?")
        response2 = await conv.say("How do I install it?")

        # Fork for exploration
        branch = conv.fork()
        alt_response = await branch.say("What about JavaScript instead?")

        # Save and restore
        conv.save("conversation.json")
        restored = Conversation.load(client, "conversation.json")
    """

    def __init__(
        self,
        client: "Flashlite",
        system: str | None = None,
        model: str | None = None,
        max_turns: int | None = None,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize a new conversation.

        Args:
            client: Flashlite client to use for completions
            system: System prompt for the conversation
            model: Default model to use (overrides client default)
            max_turns: Maximum number of turns to keep (None = unlimited)
            conversation_id: Custom conversation ID (auto-generated if not provided)
            metadata: Custom metadata to attach to the conversation
        """
        self._client = client
        self._system = system
        self._model = model
        self._max_turns = max_turns
        self._turns: list[Turn] = []
        self._id = conversation_id or str(uuid.uuid4())
        self._metadata = metadata or {}
        self._created_at = datetime.now(UTC).isoformat()
        self._updated_at = self._created_at

    async def say(
        self,
        message: str,
        *,
        model: str | None = None,
        response_model: type[T] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse | T:
        """
        Send a message and get a response.

        The message is added to history, and the assistant's response
        is also added automatically.

        Args:
            message: The user message to send
            model: Model to use for this turn (overrides conversation default)
            response_model: Pydantic model for structured output
            **kwargs: Additional arguments for complete()

        Returns:
            CompletionResponse or validated model instance if response_model provided
        """
        # Add user message to history
        self._add_turn("user", message, model=model)

        # Build messages
        messages = self._build_messages()

        # Determine model
        effective_model = model or self._model

        # Make completion
        response = await self._client.complete(
            model=effective_model,
            messages=messages,
            response_model=response_model,
            **kwargs,
        )

        # Extract content and add to history
        if isinstance(response, BaseModel):
            # For structured outputs, store the JSON representation
            content = response.model_dump_json()
            self._add_turn("assistant", content, model=effective_model)
            return response
        else:
            self._add_turn("assistant", response.content, model=response.model)
            return response

    def add_user_message(self, content: str) -> None:
        """Add a user message to history without making a completion."""
        self._add_turn("user", content)

    def add_assistant_message(self, content: str, model: str | None = None) -> None:
        """Add an assistant message to history."""
        self._add_turn("assistant", content, model=model)

    def _add_turn(
        self,
        role: str,
        content: str,
        model: str | None = None,
    ) -> None:
        """Add a turn to the conversation history."""
        turn = Turn(role=role, content=content, model=model)
        self._turns.append(turn)
        self._updated_at = datetime.now(UTC).isoformat()

        # Enforce max_turns limit
        if self._max_turns is not None:
            # Keep the most recent turns, but always keep assistant responses
            # paired with their user messages
            while len(self._turns) > self._max_turns * 2:
                self._turns.pop(0)

    def _build_messages(self) -> Messages:
        """Build the messages list for completion."""
        messages: list[Message] = []

        # Add system prompt if present
        if self._system:
            messages.append({"role": "system", "content": self._system})

        # Add conversation history
        for turn in self._turns:
            messages.append(turn.to_message())

        return messages

    def fork(self) -> "Conversation":
        """
        Create a branch of this conversation.

        The branch shares history up to this point but can diverge.
        Useful for exploring alternative conversation paths.

        Returns:
            A new Conversation with copied history
        """
        branch = Conversation(
            client=self._client,
            system=self._system,
            model=self._model,
            max_turns=self._max_turns,
            metadata=copy.deepcopy(self._metadata),
        )
        branch._turns = copy.deepcopy(self._turns)
        branch._created_at = self._created_at
        return branch

    def clear(self) -> None:
        """Clear conversation history (keeps system prompt)."""
        self._turns = []
        self._updated_at = datetime.now(UTC).isoformat()

    def rollback(self, n: int = 1) -> list[Turn]:
        """
        Remove the last n turns from history.

        Args:
            n: Number of turns to remove

        Returns:
            The removed turns
        """
        removed = []
        for _ in range(min(n, len(self._turns))):
            removed.append(self._turns.pop())
        self._updated_at = datetime.now(UTC).isoformat()
        return list(reversed(removed))

    def get_state(self) -> ConversationState:
        """Get the current conversation state."""
        return ConversationState(
            id=self._id,
            system_prompt=self._system,
            turns=copy.deepcopy(self._turns),
            default_model=self._model,
            metadata=copy.deepcopy(self._metadata),
            created_at=self._created_at,
            updated_at=self._updated_at,
        )

    def save(self, path: str | Path) -> None:
        """
        Save conversation to a JSON file.

        Args:
            path: Path to save to
        """
        state = self.get_state()
        with open(path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

    @classmethod
    def load(
        cls,
        client: "Flashlite",
        path: str | Path,
    ) -> "Conversation":
        """
        Load a conversation from a JSON file.

        Args:
            client: Flashlite client to use
            path: Path to load from

        Returns:
            Restored Conversation instance
        """
        with open(path) as f:
            data = json.load(f)

        state = ConversationState.from_dict(data)

        conv = cls(
            client=client,
            system=state.system_prompt,
            model=state.default_model,
            conversation_id=state.id,
            metadata=state.metadata,
        )
        conv._turns = state.turns
        conv._created_at = state.created_at
        conv._updated_at = state.updated_at

        return conv

    @property
    def id(self) -> str:
        """Conversation ID."""
        return self._id

    @property
    def system(self) -> str | None:
        """System prompt."""
        return self._system

    @system.setter
    def system(self, value: str | None) -> None:
        """Set system prompt."""
        self._system = value
        self._updated_at = datetime.now(UTC).isoformat()

    @property
    def model(self) -> str | None:
        """Default model for this conversation."""
        return self._model

    @model.setter
    def model(self, value: str | None) -> None:
        """Set default model."""
        self._model = value

    @property
    def turns(self) -> list[Turn]:
        """List of conversation turns (read-only copy)."""
        return copy.deepcopy(self._turns)

    @property
    def messages(self) -> Messages:
        """Current messages list for the conversation."""
        return self._build_messages()

    @property
    def turn_count(self) -> int:
        """Number of turns in the conversation."""
        return len(self._turns)

    def __len__(self) -> int:
        """Number of turns."""
        return len(self._turns)

    def __repr__(self) -> str:
        return f"Conversation(id={self._id!r}, turns={len(self._turns)}, model={self._model!r})"
