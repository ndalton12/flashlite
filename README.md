# Flashlite

A batteries-included wrapper for [litellm](https://github.com/BerriAI/litellm) designed for high-volume prompting workloads like evals, agentic loops, and social simulations.

## Features

- **Rate Limiting** - Token bucket algorithm for RPM and TPM limits
- **Retries** - Exponential backoff with jitter via tenacity
- **Jinja Templating** - Prompt templates with custom filters
- **Caching** - In-memory LRU and SQLite disk caching
- **Structured Outputs** - Native Pydantic model parsing without instructor
- **Tool/Function Calling** - `@tool` decorator and execution loops
- **Multi-Turn Conversations** - Conversation management with branching
- **Cost Tracking** - Token counting and budget limits
- **Observability** - Structured logging and Inspect framework integration
- **Async-First** - Native async with sync wrappers

## Installation

```bash
pip install flashlite
# or with uv
uv add flashlite
```

## Quick Start

```python
from flashlite import Flashlite

# Create client (loads .env automatically)
client = Flashlite(default_model="gpt-4o")

# Simple completion
response = await client.complete(
    messages="What is the capital of France?"
)
print(response.content)

# Sync version
response = client.complete_sync(messages="Hello!")
```

### Structured Outputs

```python
from pydantic import BaseModel, Field
from flashlite import Flashlite

class Sentiment(BaseModel):
    label: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(ge=0, le=1)

client = Flashlite(default_model="gpt-4o")

result: Sentiment = await client.complete(
    messages="Analyze: 'I love this product!'",
    response_model=Sentiment,
)
print(f"{result.label} ({result.confidence:.0%})")
```

### Tool/Function Calling

```python
from flashlite import Flashlite, tool, run_tool_loop

@tool()
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: 72°F, sunny"

client = Flashlite(default_model="gpt-4o")

result = await run_tool_loop(
    client=client,
    messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    tools=[get_weather],
)
print(result.content)
```

### Parallel Processing

```python
# Process many requests with concurrency control
responses = await client.complete_many(
    requests=[
        {"messages": f"Summarize: {doc}"}
        for doc in documents
    ],
    max_concurrency=10,
)
```

### Caching

```python
from flashlite import Flashlite, MemoryCache, DiskCache

# In-memory caching
client = Flashlite(
    cache=MemoryCache(max_size=1000),
    default_model="gpt-4o",
)

# Or persistent disk cache
client = Flashlite(
    cache=DiskCache("./cache.db"),
    default_model="gpt-4o",
)
```

### Reasoning Models

```python
from flashlite import Flashlite, thinking_enabled

client = Flashlite()

# OpenAI o1/o3
response = await client.complete(
    model="o3",
    messages="Solve this complex problem...",
    reasoning_effort="high",
)

# Anthropic Claude extended thinking
response = await client.complete(
    model="anthropic/claude-sonnet-4-5-20250929",
    messages="Complex reasoning task...",
    thinking=thinking_enabled(10000),
)
```

## Documentation

- [Quick Start Guide](QUICK_START.md) - Detailed examples for all features
- [Development Guide](DEV.md) - Setup and contributing
- [Implementation Plan](plan.md) - Architecture and roadmap

## Requirements

- Python 3.13+
- litellm
- pydantic>=2.0
- jinja2
- tenacity
- python-dotenv

## License

MIT
