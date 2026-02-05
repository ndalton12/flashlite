"""
Basic Flashlite Example
========================

This example demonstrates the core features of Flashlite:
- Basic completions
- Structured outputs with Pydantic
- Templating with Jinja2
- Caching for repeated requests
- Tool/function calling
- Cost tracking

To run this example:
    uv run python examples/basic_example.py

Note: Requires OPENAI_API_KEY environment variable or .env file.
"""

import asyncio

from pydantic import BaseModel, Field

from flashlite import (
    Flashlite,
    MemoryCache,
    run_tool_loop,
    tool,
)

# ============================================================================
# Structured Output Models
# ============================================================================


class Sentiment(BaseModel):
    """Sentiment analysis result."""

    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
    keywords: list[str] = Field(description="Key words that indicate sentiment")


class Recipe(BaseModel):
    """A simple recipe."""

    name: str
    ingredients: list[str]
    steps: list[str]
    prep_time_minutes: int


# ============================================================================
# Tool Definitions
# ============================================================================


@tool()
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get current weather for a location.

    Args:
        location: City name or location
        unit: Temperature unit (celsius or fahrenheit)
    """
    # In a real app, this would call a weather API
    return f"Weather in {location}: 22°{'C' if unit == 'celsius' else 'F'}, partly cloudy"


@tool()
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.

    Args:
        expression: Math expression to evaluate (e.g., "2 + 2 * 3")
    """
    try:
        # Note: In production, use a safer eval method
        result = eval(expression)  # noqa: S307
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


# ============================================================================
# Main Examples
# ============================================================================


async def basic_completion_example():
    """Simple completion without any frills."""
    print("\n" + "=" * 60)
    print("1. Basic Completion")
    print("=" * 60)

    client = Flashlite(default_model="gpt-5-nano")

    # Simple string message
    response = await client.complete(messages="What is the capital of France?")
    print("Q: What is the capital of France?")
    print(f"A: {response.content}")
    print(f"Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")


async def structured_output_example():
    """Using Pydantic models for structured outputs."""
    print("\n" + "=" * 60)
    print("2. Structured Outputs")
    print("=" * 60)

    client = Flashlite(default_model="gpt-5-nano")

    # Analyze sentiment with structured output
    result: Sentiment = await client.complete(
        messages="Analyze the sentiment: 'I absolutely love this product! Best purchase ever!'",
        response_model=Sentiment,
    )

    print(f"Sentiment: {result.sentiment}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Keywords: {', '.join(result.keywords)}")


async def templating_example():
    """Using Jinja2 templates for prompts."""
    print("\n" + "=" * 60)
    print("3. Templating")
    print("=" * 60)

    client = Flashlite(default_model="gpt-5-nano")

    # Register a template
    client.register_template(
        "analyze",
        """Analyze the following {{ content_type }} and provide insights:

{{ content }}

Focus on: {{ focus_areas | join(', ') }}""",
    )

    # Use the template
    response = await client.complete(
        template="analyze",
        variables={
            "content_type": "code snippet",
            "content": "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
            "focus_areas": ["efficiency", "readability", "improvements"],
        },
    )

    print("Template-based analysis:")
    print(response.content[:500] + "..." if len(response.content) > 500 else response.content)


async def caching_example():
    """Demonstrating caching for repeated requests."""
    print("\n" + "=" * 60)
    print("4. Caching")
    print("=" * 60)

    # Enable caching with temperature=0 for deterministic responses
    cache = MemoryCache(max_size=100)
    client = Flashlite(default_model="gpt-5-nano", cache=cache)

    question = "What is 2 + 2?"

    # First request (cache miss)
    response1 = await client.complete(messages=question, temperature=0)
    stats1 = cache.stats()
    print(f"First request - Cache misses: {stats1['misses']}, hits: {stats1['hits']}")

    # Second identical request (cache hit)
    response2 = await client.complete(messages=question, temperature=0)
    stats2 = cache.stats()
    print(f"Second request - Cache misses: {stats2['misses']}, hits: {stats2['hits']}")

    print(f"Same response: {response1.content == response2.content}")


async def tool_calling_example():
    """Using tools/function calling."""
    print("\n" + "=" * 60)
    print("5. Tool/Function Calling")
    print("=" * 60)

    client = Flashlite(default_model="gpt-5-nano")

    # Run a tool loop - model can call tools as needed
    result = await run_tool_loop(
        client=client,
        messages=[
            {
                "role": "user",
                "content": "What's the weather in Paris? Also, what's 15 * 7?",
            }
        ],
        tools=[get_weather, calculate],
        on_tool_call=lambda tc: print(f"  Tool called: {tc.name}({tc.arguments})"),
        on_tool_result=lambda tr: print(f"  Tool result: {tr.output[:50]}..."),
    )

    print(f"\nFinal response: {result.final_response.content}")
    print(f"Tool calls made: {len(result.tool_calls)}")


async def cost_tracking_example():
    """Tracking costs across requests."""
    print("\n" + "=" * 60)
    print("6. Cost Tracking")
    print("=" * 60)

    client = Flashlite(default_model="gpt-5-nano", track_costs=True)

    # Make a few requests
    await client.complete(messages="Hello!")
    await client.complete(messages="How are you?")
    await client.complete(messages="What's 1+1?")

    # Get cost report
    report = client.get_cost_report()
    if report:
        print(f"Total requests: {report['total_requests']}")
        print(f"Total tokens: {report['total_tokens']}")
        print(f"Estimated cost: ${report['total_cost']:.6f}")


async def parallel_requests_example():
    """Running multiple requests in parallel."""
    print("\n" + "=" * 60)
    print("7. Parallel Requests")
    print("=" * 60)

    client = Flashlite(default_model="gpt-5-nano")

    questions = [
        {"messages": "What is Python?"},
        {"messages": "What is JavaScript?"},
        {"messages": "What is Rust?"},
    ]

    # Process all in parallel with concurrency limit
    responses = await client.complete_many(questions, max_concurrency=3)

    for q, r in zip(questions, responses):
        answer = r.content[:100] + "..." if len(r.content) > 100 else r.content
        print(f"Q: {q['messages']}")
        print(f"A: {answer}\n")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Flashlite Basic Examples")
    print("=" * 60)

    await basic_completion_example()
    await structured_output_example()
    await templating_example()
    await caching_example()
    await tool_calling_example()
    await cost_tracking_example()
    await parallel_requests_example()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
