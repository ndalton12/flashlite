"""Tests for the main Flashlite client using litellm mock responses."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from flashlite import (
    CompletionResponse,
    Flashlite,
    FlashliteConfig,
    RateLimitConfig,
    RetryConfig,
    thinking_enabled,
)
from flashlite.types import UsageInfo


class TestFlashliteBasic:
    """Basic client tests."""

    def test_init_default(self) -> None:
        client = Flashlite()
        assert client.config is not None
        assert client.config.retry.max_attempts == 3

    def test_init_with_config(self) -> None:
        config = FlashliteConfig(
            default_model="gpt-4o",
            log_requests=True,
        )
        client = Flashlite(config=config)

        assert client.config.default_model == "gpt-4o"
        assert client.config.log_requests is True

    def test_init_with_params(self) -> None:
        client = Flashlite(
            default_model="claude-3-opus",
            retry=RetryConfig(max_attempts=5),
            rate_limit=RateLimitConfig(requests_per_minute=30),
            timeout=120.0,
        )

        assert client.config.default_model == "claude-3-opus"
        assert client.config.retry.max_attempts == 5
        assert client.config.rate_limit.requests_per_minute == 30
        assert client.config.timeout == 120.0

    def test_init_with_template_dir(self, temp_template_dir: Path) -> None:
        client = Flashlite(template_dir=temp_template_dir)

        assert client.template_engine is not None
        # Should have loaded templates from directory
        assert client.template_engine.registry.has("greeting")


class TestFlashliteCompletion:
    """Tests for completion functionality using mocked litellm."""

    @pytest.fixture
    def mock_litellm_response(self) -> CompletionResponse:
        """Create a mock completion response."""
        return CompletionResponse(
            content="Hello! I'm doing well, thanks for asking.",
            model="gpt-4o",
            finish_reason="stop",
            usage=UsageInfo(input_tokens=10, output_tokens=15, total_tokens=25),
        )

    async def test_complete_basic(self, mock_litellm_response: CompletionResponse) -> None:
        client = Flashlite(default_model="gpt-4o")

        with patch.object(client, "_core_complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_litellm_response

            response = await client.complete(messages=[{"role": "user", "content": "Hello!"}])

            assert response.content == "Hello! I'm doing well, thanks for asking."
            assert response.model == "gpt-4o"
            mock_complete.assert_called_once()

    async def test_complete_with_string_message(
        self, mock_litellm_response: CompletionResponse
    ) -> None:
        client = Flashlite(default_model="gpt-4o")

        with patch.object(client, "_core_complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_litellm_response

            response = await client.complete(messages="Hello!")

            assert response.content is not None
            # Check that the request was properly formatted
            call_args = mock_complete.call_args[0][0]
            assert call_args.messages[0]["role"] == "user"
            assert call_args.messages[0]["content"] == "Hello!"

    async def test_complete_with_system_prompt(
        self, mock_litellm_response: CompletionResponse
    ) -> None:
        client = Flashlite(default_model="gpt-4o")

        with patch.object(client, "_core_complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_litellm_response

            await client.complete(
                messages="What's Python?",
                system="You are a programming tutor.",
            )

            call_args = mock_complete.call_args[0][0]
            assert len(call_args.messages) == 2
            assert call_args.messages[0]["role"] == "system"
            assert call_args.messages[1]["role"] == "user"

    async def test_complete_with_template(
        self, temp_template_dir: Path, mock_litellm_response: CompletionResponse
    ) -> None:
        client = Flashlite(default_model="gpt-4o", template_dir=temp_template_dir)

        with patch.object(client, "_core_complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_litellm_response

            await client.complete(
                template="greeting",
                variables={"name": "Alice", "place": "Wonderland"},
            )

            call_args = mock_complete.call_args[0][0]
            # Template should be rendered into message content
            assert "Alice" in call_args.messages[0]["content"]
            assert "Wonderland" in call_args.messages[0]["content"]

    async def test_complete_passes_kwargs(self, mock_litellm_response: CompletionResponse) -> None:
        client = Flashlite(default_model="gpt-4o")

        with patch.object(client, "_core_complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_litellm_response

            await client.complete(
                messages="Hello",
                temperature=0.7,
                max_tokens=100,
                top_p=0.9,
            )

            call_args = mock_complete.call_args[0][0]
            assert call_args.temperature == 0.7
            assert call_args.max_tokens == 100
            assert call_args.top_p == 0.9

    async def test_complete_with_openai_reasoning_params(
        self, mock_litellm_response: CompletionResponse
    ) -> None:
        """Test OpenAI o1/o3 reasoning_effort parameter."""
        client = Flashlite(default_model="o3")

        with patch.object(client, "_core_complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_litellm_response

            await client.complete(
                messages="Solve this problem",
                reasoning_effort="high",
                max_completion_tokens=16000,
            )

            call_args = mock_complete.call_args[0][0]
            assert call_args.reasoning_effort == "high"
            assert call_args.max_completion_tokens == 16000

    async def test_complete_with_anthropic_thinking_params(
        self, mock_litellm_response: CompletionResponse
    ) -> None:
        """Test Anthropic Claude extended thinking parameter."""
        client = Flashlite(default_model="claude-sonnet-4-5-20250929")

        with patch.object(client, "_core_complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_litellm_response

            await client.complete(
                messages="Solve this complex problem",
                thinking=thinking_enabled(10000),
                max_tokens=16000,
            )

            call_args = mock_complete.call_args[0][0]
            assert call_args.thinking == {"type": "enabled", "budget_tokens": 10000}
            assert call_args.max_tokens == 16000

    async def test_complete_with_thinking_dict_directly(
        self, mock_litellm_response: CompletionResponse
    ) -> None:
        """Test passing thinking config as dict directly."""
        client = Flashlite(default_model="claude-sonnet-4-5-20250929")

        with patch.object(client, "_core_complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_litellm_response

            await client.complete(
                messages="Complex task",
                thinking={"type": "enabled", "budget_tokens": 5000},
            )

            call_args = mock_complete.call_args[0][0]
            assert call_args.thinking["type"] == "enabled"
            assert call_args.thinking["budget_tokens"] == 5000

    async def test_complete_extra_kwargs_passthrough(
        self, mock_litellm_response: CompletionResponse
    ) -> None:
        client = Flashlite(default_model="gpt-4o")

        with patch.object(client, "_core_complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_litellm_response

            await client.complete(
                messages="Hello",
                presence_penalty=0.5,
                frequency_penalty=0.3,
                custom_param="custom_value",
            )

            call_args = mock_complete.call_args[0][0]
            assert call_args.extra_kwargs["presence_penalty"] == 0.5
            assert call_args.extra_kwargs["frequency_penalty"] == 0.3
            assert call_args.extra_kwargs["custom_param"] == "custom_value"

    async def test_complete_no_model_raises(self) -> None:
        client = Flashlite()  # No default model

        with pytest.raises(ValueError, match="No model specified"):
            await client.complete(messages="Hello")

    async def test_complete_no_messages_raises(self) -> None:
        client = Flashlite(default_model="gpt-4o")

        with pytest.raises(ValueError, match="No messages"):
            await client.complete()


class TestFlashliteBatch:
    """Tests for batch completion."""

    async def test_complete_many(self) -> None:
        client = Flashlite(default_model="gpt-4o")

        responses = [
            CompletionResponse(
                content=f"Response {i}",
                model="gpt-4o",
                usage=UsageInfo(10, 10, 20),
            )
            for i in range(3)
        ]

        call_count = 0

        async def mock_complete(*args, **kwargs):
            nonlocal call_count
            result = responses[call_count]
            call_count += 1
            return result

        with patch.object(client, "complete", side_effect=mock_complete):
            results = await client.complete_many(
                [
                    {"messages": "Request 1"},
                    {"messages": "Request 2"},
                    {"messages": "Request 3"},
                ],
                max_concurrency=2,
            )

        assert len(results) == 3
        assert results[0].content == "Response 0"
        assert results[1].content == "Response 1"
        assert results[2].content == "Response 2"


class TestFlashliteTemplates:
    """Tests for template management."""

    def test_register_template(self) -> None:
        client = Flashlite()
        client.register_template("custom", "Hello {{ name }}!")

        result = client.render_template("custom", {"name": "World"})
        assert result == "Hello World!"

    def test_render_inline_template(self) -> None:
        client = Flashlite()

        result = client.render_template("The answer is {{ answer }}.", {"answer": 42})
        assert result == "The answer is 42."


class TestFlashliteSync:
    """Tests for sync API."""

    def test_complete_sync(self) -> None:
        client = Flashlite(default_model="gpt-4o")

        mock_response = CompletionResponse(
            content="Sync response",
            model="gpt-4o",
            usage=UsageInfo(5, 10, 15),
        )

        with patch.object(client, "_core_complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_response

            response = client.complete_sync(messages="Hello sync!")

            assert response.content == "Sync response"
