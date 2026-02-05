"""Tests for types and data structures."""


from flashlite.types import (
    CompletionRequest,
    CompletionResponse,
    RateLimitConfig,
    RetryConfig,
    UsageInfo,
    thinking_enabled,
)


class TestCompletionRequest:
    """Tests for CompletionRequest."""

    def test_basic_request(self) -> None:
        request = CompletionRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )

        kwargs = request.to_litellm_kwargs()

        assert kwargs["model"] == "gpt-4o"
        assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]

    def test_with_all_params(self) -> None:
        request = CompletionRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.7,
            max_tokens=100,
            max_completion_tokens=200,
            top_p=0.9,
            stop=["END"],
        )

        kwargs = request.to_litellm_kwargs()

        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 100
        assert kwargs["max_completion_tokens"] == 200
        assert kwargs["top_p"] == 0.9
        assert kwargs["stop"] == ["END"]

    def test_with_openai_reasoning_effort(self) -> None:
        """Test OpenAI o1/o3 reasoning_effort parameter."""
        request = CompletionRequest(
            model="o3",
            messages=[{"role": "user", "content": "Solve this"}],
            reasoning_effort="high",
            max_completion_tokens=16000,
        )

        kwargs = request.to_litellm_kwargs()

        assert kwargs["reasoning_effort"] == "high"
        assert kwargs["max_completion_tokens"] == 16000
        # thinking should not be in kwargs when not set
        assert "thinking" not in kwargs

    def test_with_anthropic_thinking(self) -> None:
        """Test Anthropic Claude extended thinking parameter."""
        request = CompletionRequest(
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Complex task"}],
            thinking={"type": "enabled", "budget_tokens": 10000},
            max_tokens=16000,
        )

        kwargs = request.to_litellm_kwargs()

        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 10000}
        assert kwargs["max_tokens"] == 16000
        # reasoning_effort should not be in kwargs when not set
        assert "reasoning_effort" not in kwargs

    def test_with_thinking_enabled_helper(self) -> None:
        """Test using the thinking_enabled helper function."""
        request = CompletionRequest(
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Task"}],
            thinking=thinking_enabled(5000),
        )

        kwargs = request.to_litellm_kwargs()

        assert kwargs["thinking"]["type"] == "enabled"
        assert kwargs["thinking"]["budget_tokens"] == 5000

    def test_extra_kwargs_passthrough(self) -> None:
        request = CompletionRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test"}],
            extra_kwargs={
                "presence_penalty": 0.5,
                "frequency_penalty": 0.3,
                "custom_param": "value",
            },
        )

        kwargs = request.to_litellm_kwargs()

        assert kwargs["presence_penalty"] == 0.5
        assert kwargs["frequency_penalty"] == 0.3
        assert kwargs["custom_param"] == "value"

    def test_none_params_not_included(self) -> None:
        """Parameters set to None should not appear in kwargs."""
        request = CompletionRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test"}],
            temperature=None,
            max_tokens=None,
        )

        kwargs = request.to_litellm_kwargs()

        assert "temperature" not in kwargs
        assert "max_tokens" not in kwargs


class TestThinkingEnabled:
    """Tests for thinking_enabled helper."""

    def test_basic_usage(self) -> None:
        config = thinking_enabled(10000)

        assert config["type"] == "enabled"
        assert config["budget_tokens"] == 10000

    def test_minimum_budget(self) -> None:
        config = thinking_enabled(1024)
        assert config["budget_tokens"] == 1024

    def test_large_budget(self) -> None:
        config = thinking_enabled(32000)
        assert config["budget_tokens"] == 32000


class TestCompletionResponse:
    """Tests for CompletionResponse."""

    def test_basic_response(self) -> None:
        response = CompletionResponse(
            content="Hello!",
            model="gpt-4o",
            finish_reason="stop",
        )

        assert response.content == "Hello!"
        assert response.model == "gpt-4o"
        assert response.finish_reason == "stop"

    def test_with_usage(self) -> None:
        usage = UsageInfo(input_tokens=10, output_tokens=20, total_tokens=30)
        response = CompletionResponse(
            content="Response",
            model="gpt-4o",
            usage=usage,
        )

        assert response.input_tokens == 10
        assert response.output_tokens == 20

    def test_tokens_without_usage(self) -> None:
        response = CompletionResponse(content="Test", model="gpt-4o")

        assert response.input_tokens == 0
        assert response.output_tokens == 0


class TestUsageInfo:
    """Tests for UsageInfo."""

    def test_from_litellm(self) -> None:
        litellm_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

        usage = UsageInfo.from_litellm(litellm_usage)

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_from_litellm_none(self) -> None:
        usage = UsageInfo.from_litellm(None)

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_from_litellm_empty(self) -> None:
        usage = UsageInfo.from_litellm({})

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0


class TestConfigTypes:
    """Tests for configuration types."""

    def test_retry_config_defaults(self) -> None:
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert 429 in config.retry_on_status

    def test_retry_config_custom(self) -> None:
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.5,
            jitter=False,
        )

        assert config.max_attempts == 5
        assert config.initial_delay == 0.5
        assert config.jitter is False

    def test_rate_limit_config_defaults(self) -> None:
        config = RateLimitConfig()

        assert config.requests_per_minute is None
        assert config.tokens_per_minute is None
        assert config.auto_detect is False

    def test_rate_limit_config_custom(self) -> None:
        config = RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=100000,
        )

        assert config.requests_per_minute == 60
        assert config.tokens_per_minute == 100000
