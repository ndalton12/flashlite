"""Metrics and cost tracking for flashlite."""

import logging
from dataclasses import dataclass, field
from typing import Any

from ..types import CompletionResponse

logger = logging.getLogger(__name__)


# Approximate costs per 1K tokens (USD) as of early 2025
# These are estimates and may be outdated - use litellm's cost tracking for accuracy
DEFAULT_COSTS: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "o1": {"input": 0.015, "output": 0.06},
    "o1-mini": {"input": 0.003, "output": 0.012},
    "o3-mini": {"input": 0.003, "output": 0.012},
    # Anthropic
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku-20241022": {"input": 0.001, "output": 0.005},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    # Add more as needed
}


@dataclass
class CostMetrics:
    """Accumulated cost metrics."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    total_cost_usd: float = 0.0
    cost_by_model: dict[str, float] = field(default_factory=dict)
    tokens_by_model: dict[str, dict[str, int]] = field(default_factory=dict)


class CostTracker:
    """
    Tracks token usage and estimated costs across requests.

    Example:
        tracker = CostTracker(budget_limit=10.0)

        # Track a response
        tracker.track(response)

        # Check totals
        print(f"Total cost: ${tracker.total_cost:.2f}")
        print(f"Budget remaining: ${tracker.budget_remaining:.2f}")

        # Export report
        report = tracker.get_report()
    """

    def __init__(
        self,
        budget_limit: float | None = None,
        warn_at_percent: float = 80.0,
        custom_costs: dict[str, dict[str, float]] | None = None,
    ):
        """
        Initialize the cost tracker.

        Args:
            budget_limit: Maximum budget in USD (None = no limit)
            warn_at_percent: Warn when this percentage of budget is used
            custom_costs: Custom cost overrides per model
        """
        self._budget_limit = budget_limit
        self._warn_at_percent = warn_at_percent
        self._costs = {**DEFAULT_COSTS, **(custom_costs or {})}
        self._metrics = CostMetrics()
        self._budget_warning_issued = False

    def _get_model_cost(self, model: str) -> dict[str, float]:
        """Get cost per 1K tokens for a model."""
        # Exact match
        if model in self._costs:
            return self._costs[model]

        # Try to match by prefix (e.g., "gpt-4o" matches "gpt-4o-2024-...")
        model_lower = model.lower()
        for known_model, cost in self._costs.items():
            if model_lower.startswith(known_model.lower()):
                return cost

        # Default to GPT-4o pricing as a reasonable estimate
        logger.debug(f"Unknown model cost for '{model}', using gpt-4o pricing estimate")
        return self._costs.get("gpt-4o", {"input": 0.0025, "output": 0.01})

    def track(self, response: CompletionResponse) -> float:
        """
        Track a completion response and return its cost.

        Args:
            response: The completion response to track

        Returns:
            The estimated cost in USD for this response
        """
        if not response.usage:
            return 0.0

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        model = response.model

        # Calculate cost
        model_costs = self._get_model_cost(model)
        input_cost = (input_tokens / 1000) * model_costs["input"]
        output_cost = (output_tokens / 1000) * model_costs["output"]
        total_cost = input_cost + output_cost

        # Update metrics
        self._metrics.total_input_tokens += input_tokens
        self._metrics.total_output_tokens += output_tokens
        self._metrics.total_requests += 1
        self._metrics.total_cost_usd += total_cost

        # Per-model tracking
        if model not in self._metrics.cost_by_model:
            self._metrics.cost_by_model[model] = 0.0
            self._metrics.tokens_by_model[model] = {"input": 0, "output": 0}

        self._metrics.cost_by_model[model] += total_cost
        self._metrics.tokens_by_model[model]["input"] += input_tokens
        self._metrics.tokens_by_model[model]["output"] += output_tokens

        # Check budget
        self._check_budget()

        return total_cost

    def _check_budget(self) -> None:
        """Check if budget thresholds are exceeded."""
        if self._budget_limit is None:
            return

        percent_used = (self._metrics.total_cost_usd / self._budget_limit) * 100

        # Warning threshold
        if percent_used >= self._warn_at_percent and not self._budget_warning_issued:
            logger.warning(
                f"Budget warning: {percent_used:.1f}% of ${self._budget_limit:.2f} budget used "
                f"(${self._metrics.total_cost_usd:.4f} spent)"
            )
            self._budget_warning_issued = True

        # Hard limit
        if self._metrics.total_cost_usd >= self._budget_limit:
            raise BudgetExceededError(
                f"Budget limit of ${self._budget_limit:.2f} exceeded "
                f"(${self._metrics.total_cost_usd:.4f} spent)"
            )

    @property
    def total_cost(self) -> float:
        """Total cost in USD."""
        return self._metrics.total_cost_usd

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self._metrics.total_input_tokens + self._metrics.total_output_tokens

    @property
    def total_requests(self) -> int:
        """Total number of requests tracked."""
        return self._metrics.total_requests

    @property
    def budget_remaining(self) -> float | None:
        """Remaining budget in USD, or None if no limit."""
        if self._budget_limit is None:
            return None
        return max(0.0, self._budget_limit - self._metrics.total_cost_usd)

    def get_report(self) -> dict[str, Any]:
        """
        Get a detailed cost report.

        Returns:
            Dictionary with cost breakdown
        """
        return {
            "total_cost_usd": self._metrics.total_cost_usd,
            "total_requests": self._metrics.total_requests,
            "total_tokens": {
                "input": self._metrics.total_input_tokens,
                "output": self._metrics.total_output_tokens,
                "total": self.total_tokens,
            },
            "budget_limit_usd": self._budget_limit,
            "budget_remaining_usd": self.budget_remaining,
            "by_model": {
                model: {
                    "cost_usd": self._metrics.cost_by_model[model],
                    "tokens": self._metrics.tokens_by_model[model],
                }
                for model in self._metrics.cost_by_model
            },
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics = CostMetrics()
        self._budget_warning_issued = False


class BudgetExceededError(Exception):
    """Raised when the budget limit is exceeded."""

    pass
