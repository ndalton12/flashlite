"""Integration tests using litellm's mock_response feature."""

from flashlite import Flashlite, RateLimitConfig, thinking_enabled


class TestLitellmMock:
    """
    Tests using litellm's built-in mock_response feature.

    When you pass mock_response="..." to litellm, it returns that string
    as the completion without hitting any API.
    """

    async def test_mock_response_basic(self) -> None:
        """Test basic completion with mock response."""
        client = Flashlite()

        response = await client.complete(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Say hello"}],
            mock_response="Hello from mock!",
        )

        assert response.content == "Hello from mock!"
        assert response.model is not None

    async def test_mock_response_with_system(self) -> None:
        """Test completion with system prompt using mock."""
        client = Flashlite()

        response = await client.complete(
            model="gpt-4o",
            messages="What is 2+2?",
            system="You are a math tutor.",
            mock_response="The answer is 4.",
        )

        assert response.content == "The answer is 4."

    async def test_mock_response_with_template(self, temp_template_dir) -> None:
        """Test template rendering with mock response."""
        client = Flashlite(template_dir=temp_template_dir)

        response = await client.complete(
            model="gpt-4o",
            template="greeting",
            variables={"name": "Test", "place": "Mockland"},
            mock_response="Greetings acknowledged!",
        )

        assert response.content == "Greetings acknowledged!"

    async def test_mock_response_batch(self) -> None:
        """Test batch completion with mock responses."""
        client = Flashlite()

        requests = [
            {
                "model": "gpt-4o",
                "messages": f"Message {i}",
                "mock_response": f"Response {i}",
            }
            for i in range(3)
        ]

        responses = await client.complete_many(requests, max_concurrency=2)

        assert len(responses) == 3
        assert responses[0].content == "Response 0"
        assert responses[1].content == "Response 1"
        assert responses[2].content == "Response 2"

    async def test_mock_with_rate_limiting(self) -> None:
        """Test that rate limiting works with mock responses."""
        client = Flashlite(
            rate_limit=RateLimitConfig(requests_per_minute=6000),  # High limit
        )

        # Make several requests - should all succeed
        for i in range(5):
            response = await client.complete(
                model="gpt-4o",
                messages=f"Request {i}",
                mock_response=f"Response {i}",
            )
            assert response.content == f"Response {i}"

    def test_mock_sync(self) -> None:
        """Test sync completion with mock response."""
        client = Flashlite()

        response = client.complete_sync(
            model="gpt-4o",
            messages="Hello sync",
            mock_response="Hello from sync mock!",
        )

        assert response.content == "Hello from sync mock!"

    async def test_mock_with_openai_reasoning(self) -> None:
        """Test OpenAI reasoning model parameters with mock."""
        client = Flashlite()

        response = await client.complete(
            model="o3",
            messages="Complex reasoning task",
            reasoning_effort="high",
            max_completion_tokens=16000,
            mock_response="Solved with deep reasoning!",
        )

        assert response.content == "Solved with deep reasoning!"

    async def test_mock_with_anthropic_thinking(self) -> None:
        """Test Anthropic extended thinking with mock."""
        client = Flashlite()

        response = await client.complete(
            model="claude-sonnet-4-5-20250929",
            messages="Complex problem requiring deep thought",
            thinking=thinking_enabled(10000),
            max_tokens=16000,
            mock_response="Thought deeply and found the answer!",
        )

        assert response.content == "Thought deeply and found the answer!"

    async def test_mock_with_thinking_dict(self) -> None:
        """Test passing thinking as raw dict."""
        client = Flashlite()

        response = await client.complete(
            model="claude-sonnet-4-5-20250929",
            messages="Another complex problem",
            thinking={"type": "enabled", "budget_tokens": 5000},
            mock_response="Solution found!",
        )

        assert response.content == "Solution found!"


class TestEndToEnd:
    """End-to-end tests simulating real usage patterns."""

    async def test_multi_turn_conversation(self) -> None:
        """Test a multi-turn conversation pattern."""
        client = Flashlite()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"},
        ]

        # First turn
        response1 = await client.complete(
            model="gpt-4o",
            messages=messages,
            mock_response="Python is a programming language.",
        )

        # Add response to history
        messages.append({"role": "assistant", "content": response1.content})
        messages.append({"role": "user", "content": "What are its main uses?"})

        # Second turn
        response2 = await client.complete(
            model="gpt-4o",
            messages=messages,
            mock_response="Python is used for web development, data science, and more.",
        )

        assert "programming language" in response1.content
        assert "web development" in response2.content

    async def test_structured_output_simulation(self) -> None:
        """Test pattern for structured outputs (JSON response)."""
        client = Flashlite()

        # Simulate requesting JSON output
        response = await client.complete(
            model="gpt-4o",
            messages="Extract the sentiment from: 'I love this product!'",
            system=(
                'Respond with JSON: {"sentiment": "positive/negative/neutral", '
                '"confidence": 0.0-1.0}'
            ),
            mock_response='{"sentiment": "positive", "confidence": 0.95}',
        )

        # Parse the response
        import json

        data = json.loads(response.content)

        assert data["sentiment"] == "positive"
        assert data["confidence"] == 0.95

    async def test_agent_batch_pattern(self) -> None:
        """Test pattern for running multiple agents in parallel."""
        client = Flashlite()

        # Simulate 5 agents each making a decision
        agent_requests = [
            {
                "model": "gpt-4o",
                "messages": f"Agent {i}: Decide whether to cooperate or defect.",
                "system": "You are agent {i}. Respond with COOPERATE or DEFECT.",
                "mock_response": "COOPERATE" if i % 2 == 0 else "DEFECT",
            }
            for i in range(5)
        ]

        responses = await client.complete_many(agent_requests, max_concurrency=3)

        decisions = [r.content for r in responses]
        assert decisions == ["COOPERATE", "DEFECT", "COOPERATE", "DEFECT", "COOPERATE"]

    async def test_eval_pattern(self) -> None:
        """Test pattern for running evaluations."""
        client = Flashlite()

        test_cases = [
            {"input": "2+2", "expected": "4"},
            {"input": "3*3", "expected": "9"},
            {"input": "10/2", "expected": "5"},
        ]

        requests = [
            {
                "model": "gpt-4o",
                "messages": f"Calculate: {tc['input']}",
                "system": "You are a calculator. Respond with only the numeric answer.",
                "mock_response": tc["expected"],
            }
            for tc in test_cases
        ]

        responses = await client.complete_many(requests)

        # Check all answers match expected
        for response, tc in zip(responses, test_cases):
            assert response.content == tc["expected"]
