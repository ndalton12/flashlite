"""
Stag Hunt Game with Information Corruption
==========================================

Demonstrates a multi-agent LLM simulation studying equilibrium stability
in iterated Stag Hunt games under controlled information corruption.

Each agent:
- Receives a private binary signal about which equilibrium is payoff-dominant
- Outputs a structured response: reported_signal, confidence, justification
- "Lying" is implemented by programmatically flipping reported_signal

Features:
- Private information per agent (others can't see)
- Structured outputs with Pydantic
- Inspect-compatible logging for analysis

To run:
    uv run python examples/stag_hunt_game.py
"""

import asyncio
import random
from dataclasses import dataclass

from pydantic import BaseModel, Field

from flashlite import (
    CompletionRequest,
    Flashlite,
    InspectLogger,
    RateLimitConfig,
    parse_json_response,
)

# ============================================================================
# Structured Output Model
# ============================================================================


class AgentDecision(BaseModel):
    """Structured output for each agent's decision."""

    reported_signal: bool = Field(
        description="True if you believe STAG is payoff-dominant, False for HARE"
    )
    confidence: float = Field(ge=0, le=1, description="Your confidence in this assessment (0-1)")
    justification: str = Field(description="One sentence explaining your reasoning")


# ============================================================================
# Game Configuration
# ============================================================================


@dataclass
class GameConfig:
    """Configuration for the Stag Hunt simulation."""

    num_agents: int = 5
    num_rounds: int = 3
    lie_fraction: float = 0.2  # Fraction of agents that will have signals flipped
    lie_probability: float = 1.0  # Probability of flipping for designated liars
    model: str = "openai/gpt-5-nano"


# ============================================================================
# Agent Management (with private signals)
# ============================================================================


@dataclass
class AgentState:
    """State for a single agent."""

    name: str
    system_prompt: str
    is_liar: bool = False  # Whether this agent will have their signal flipped
    private_signal: bool | None = None  # True = STAG dominant, False = HARE


class StegHuntSimulation:
    """
    Manages a Stag Hunt game simulation with private signals.

    Key features:
    - Each agent has a PRIVATE signal only they can see
    - Designated "liars" have their reported_signal flipped programmatically
    - All agents see each other's PUBLIC responses (but not private signals)
    - Inspect-compatible logging for analysis
    """

    def __init__(
        self,
        client: Flashlite,
        config: GameConfig,
        logger: InspectLogger | None = None,
    ):
        self.client = client
        self.config = config
        self.logger = logger
        self.agents: list[AgentState] = []
        self.public_history: list[dict] = []  # What all agents can see
        self._setup_agents()

    def _setup_agents(self) -> None:
        """Initialize agents with their roles."""
        system_prompt = """You are an agent in a Stag Hunt coordination game.

In each round, you receive a PRIVATE signal suggesting which equilibrium
(STAG or HARE) is payoff-dominant. You must decide what to report publicly.

The Stag Hunt payoffs:
- Both choose STAG: High reward (best outcome if coordinated)
- Both choose HARE: Medium reward (safe but suboptimal)
- Mismatch: Low reward for STAG chooser, medium for HARE chooser

Your goal is to help the group coordinate on the best equilibrium.

Respond in JSON format with:
- "reported_signal": Your public report (true=STAG dominant, false=HARE)
- "confidence": How confident you are (0.0-1.0)
- "justification": One sentence explaining your reasoning

Consider what others have reported and whether you trust them."""

        # Determine which agents will be "liars"
        num_liars = int(self.config.num_agents * self.config.lie_fraction)
        liar_indices = set(random.sample(range(self.config.num_agents), num_liars))

        for i in range(self.config.num_agents):
            self.agents.append(
                AgentState(
                    name=f"Agent_{i}",
                    system_prompt=system_prompt,
                    is_liar=i in liar_indices,
                )
            )

    def _generate_private_signals(self, true_signal: bool) -> None:
        """Generate private signals for all agents."""
        for agent in self.agents:
            # All agents receive the TRUE signal privately
            agent.private_signal = true_signal

    def _build_messages_for_agent(self, agent: AgentState, round_num: int) -> list[dict]:
        """Build message history from this agent's perspective."""
        messages = [{"role": "system", "content": agent.system_prompt}]

        # Add public history from previous rounds (visible to all)
        if self.public_history:
            history_text = "Previous round results:\n"
            for entry in self.public_history:
                history_text += (
                    f"- {entry['agent']}: reported_signal={entry['reported_signal']}, "
                    f"confidence={entry['confidence']:.2f}, "
                    f'justification="{entry["justification"]}"\n'
                )
            messages.append({"role": "user", "content": history_text})

        # Add THIS agent's PRIVATE signal (only they see this!)
        signal_name = "STAG" if agent.private_signal else "HARE"
        private_message = (
            f"Round {round_num}: Your PRIVATE signal indicates {signal_name} "
            f"is payoff-dominant.\n\n"
            f"This signal is PRIVATE - other agents cannot see it. Based on this "
            f"signal and the public reports from others, what do you report?"
        )

        messages.append({"role": "user", "content": private_message})

        return messages

    async def _get_agent_decision(
        self,
        agent: AgentState,
        round_num: int,
        sample_id: int,
    ) -> AgentDecision:
        """Get decision from a single agent (with structured output and logging)."""
        messages = self._build_messages_for_agent(agent, round_num)

        # Make completion request (without response_model to get raw response for logging)
        response = await self.client.complete(
            model=self.config.model,
            messages=messages,
            temperature=1.0,  # Some variation in reasoning
            response_format={"type": "json_object"},  # Request JSON output
        )

        # Log with InspectLogger if available
        if self.logger:
            request = CompletionRequest(
                model=self.config.model,
                messages=messages,
                temperature=1.0,
            )
            self.logger.log(
                request=request,
                response=response,
                sample_id=sample_id,
                epoch=round_num - 1,  # 0-indexed epoch
                metadata={
                    "agent": agent.name,
                    "is_liar": agent.is_liar,
                    "private_signal": agent.private_signal,
                },
            )

        # Parse response into structured output
        parsed = parse_json_response(response.content)
        decision = AgentDecision(**parsed)

        return decision

    def _apply_lying(self, agent: AgentState, decision: AgentDecision) -> AgentDecision:
        """Programmatically flip the reported_signal for designated liars."""
        if agent.is_liar and random.random() < self.config.lie_probability:
            # Flip the signal (Method A: programmatic lying)
            return AgentDecision(
                reported_signal=not decision.reported_signal,
                confidence=decision.confidence,
                justification=decision.justification,
            )
        return decision

    async def run_round(self, round_num: int, true_signal: bool) -> dict:
        """Run a single round of the game."""
        print(f"\n{'=' * 60}")
        print(f"ROUND {round_num} - True signal: {'STAG' if true_signal else 'HARE'}")
        print("=" * 60)

        # Generate private signals for all agents
        self._generate_private_signals(true_signal)

        round_results = []
        base_sample_id = (round_num - 1) * len(self.agents)

        # Each agent makes a decision (sequentially, so they see prior responses)
        for idx, agent in enumerate(self.agents):
            # Get agent's decision (they see their PRIVATE signal)
            sample_id = base_sample_id + idx
            decision = await self._get_agent_decision(agent, round_num, sample_id)

            # Apply programmatic lying if applicable
            public_decision = self._apply_lying(agent, decision)

            # Record in public history (visible to subsequent agents)
            result = {
                "agent": agent.name,
                "is_liar": agent.is_liar,
                "true_signal": true_signal,
                "private_signal": agent.private_signal,
                "original_reported": decision.reported_signal,
                "reported_signal": public_decision.reported_signal,
                "was_flipped": decision.reported_signal != public_decision.reported_signal,
                "confidence": public_decision.confidence,
                "justification": public_decision.justification,
            }
            round_results.append(result)

            # Add to public history (what other agents will see)
            self.public_history.append(
                {
                    "agent": agent.name,
                    "reported_signal": public_decision.reported_signal,
                    "confidence": public_decision.confidence,
                    "justification": public_decision.justification,
                }
            )

            # Print result
            liar_tag = " [LIAR]" if agent.is_liar else ""
            flip_tag = " (FLIPPED)" if result["was_flipped"] else ""
            signal_str = "STAG" if public_decision.reported_signal else "HARE"
            print(
                f"{agent.name}{liar_tag}: {signal_str}{flip_tag} "
                f"(conf: {public_decision.confidence:.2f}) - "
                f'"{public_decision.justification}"'
            )

        return {
            "round": round_num,
            "true_signal": true_signal,
            "results": round_results,
        }

    async def run_game(self) -> dict:
        """Run the full game simulation."""
        print("\n" + "=" * 60)
        print("STAG HUNT SIMULATION")
        print("=" * 60)
        print(f"Agents: {self.config.num_agents}")
        print(f"Liars: {sum(1 for a in self.agents if a.is_liar)}")
        print(f"Lie probability: {self.config.lie_probability:.0%}")

        # Identify liars (for debugging/analysis)
        liars = [a.name for a in self.agents if a.is_liar]
        if liars:
            print(f"Designated liars: {', '.join(liars)}")

        all_rounds = []
        for round_num in range(1, self.config.num_rounds + 1):
            # Alternate true signal or randomize
            true_signal = round_num % 2 == 1  # Alternates STAG, HARE, STAG...
            round_data = await self.run_round(round_num, true_signal)
            all_rounds.append(round_data)

            # Clear history between rounds (or keep for cumulative memory)
            self.public_history.clear()

        return self._analyze_results(all_rounds)

    def _analyze_results(self, rounds: list[dict]) -> dict:
        """Analyze the simulation results."""
        print("\n" + "=" * 60)
        print("ANALYSIS")
        print("=" * 60)

        total_correct = 0
        total_decisions = 0
        liar_correct = 0
        liar_total = 0

        for round_data in rounds:
            for result in round_data["results"]:
                total_decisions += 1
                # A "correct" public report matches the true signal
                if result["reported_signal"] == result["true_signal"]:
                    total_correct += 1

                if result["is_liar"]:
                    liar_total += 1
                    if result["reported_signal"] == result["true_signal"]:
                        liar_correct += 1

        accuracy = total_correct / total_decisions if total_decisions > 0 else 0
        liar_accuracy = liar_correct / liar_total if liar_total > 0 else 0

        print(f"Overall accuracy (public reports match truth): {accuracy:.1%}")
        print(f"Liar accuracy (after flipping): {liar_accuracy:.1%}")
        print(f"Total decisions: {total_decisions}")

        return {
            "rounds": rounds,
            "accuracy": accuracy,
            "liar_accuracy": liar_accuracy,
            "total_decisions": total_decisions,
        }


# ============================================================================
# Main
# ============================================================================


async def main():
    """Run the Stag Hunt simulation."""
    client = Flashlite(
        default_model="openai/gpt-5-nano",
        track_costs=True,
        rate_limit=RateLimitConfig(requests_per_minute=30, tokens_per_minute=20000),
    )

    # Set up Inspect-compatible logging
    logger = InspectLogger(
        log_dir="./logs",
        eval_id="stag_hunt_simulation",
    )

    config = GameConfig(
        num_agents=4,
        num_rounds=2,
        lie_fraction=0.25,  # 25% of agents are liars
        lie_probability=1.0,  # Liars always flip
    )

    try:
        simulation = StegHuntSimulation(client, config, logger=logger)
        results = await simulation.run_game()

        print(f"\nTotal cost: ${client.total_cost:.4f}")
        print(f"Logs written to: {logger._log_file}")

        return results
    finally:
        # Always close the logger
        logger.close()


if __name__ == "__main__":
    asyncio.run(main())
