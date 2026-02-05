# Flashlite Quick Start

Flashlite is a batteries-included wrapper for [litellm](https://github.com/BerriAI/litellm) that adds rate limiting, retries, caching, templating, and observability.

## Installation

```bash
pip install flashlite
# or with uv
uv add flashlite
```

## Basic Usage

```python
from flashlite import Flashlite

# Create client (loads .env automatically)
client = Flashlite()

# Simple completion
response = await client.complete(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content)

# Even simpler - just pass a string
response = await client.complete(
    model="gpt-4o",
    messages="What is the capital of France?"
)

# Sync version (for scripts without async)
response = client.complete_sync(
    model="gpt-4o",
    messages="Hello!"
)
```

## Configuration

### Environment Variables

Create a `.env` file:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
FLASHLITE_DEFAULT_MODEL=gpt-4o
```

### Client Options

```python
from flashlite import Flashlite, RetryConfig, RateLimitConfig

client = Flashlite(
    default_model="gpt-4o",
    retry=RetryConfig(max_attempts=3),
    rate_limit=RateLimitConfig(requests_per_minute=60),
    log_requests=True,
)
```

## Rate Limiting

Automatically respects API rate limits:

```python
from flashlite import Flashlite, RateLimitConfig

client = Flashlite(
    rate_limit=RateLimitConfig(
        requests_per_minute=60,      # 1 request per second
        tokens_per_minute=100_000,   # Token budget
    )
)

# Requests are automatically throttled
for prompt in prompts:
    response = await client.complete(model="gpt-4o", messages=prompt)
```

## Retries with Backoff

Automatic retries on transient failures:

```python
from flashlite import Flashlite, RetryConfig

client = Flashlite(
    retry=RetryConfig(
        max_attempts=3,
        initial_delay=1.0,
        exponential_base=2.0,
        jitter=True,
    )
)
```

## Parallel Processing

Process multiple requests in parallel with concurrency control:

```python
from flashlite import Flashlite

client = Flashlite(default_model="gpt-4o")

# Prepare requests
requests = [
    {"messages": "Summarize: " + doc}
    for doc in documents
]

# Run in parallel (max 10 concurrent)
responses = await client.complete_many(requests, max_concurrency=10)

for doc, response in zip(documents, responses):
    print(f"Summary: {response.content}")
```

## Jinja Templating

### Inline Templates

```python
from flashlite import Flashlite

client = Flashlite()

response = await client.complete(
    model="gpt-4o",
    template="Translate '{{ text }}' to {{ language }}",
    variables={"text": "Hello", "language": "French"}
)
```

### Template Files

Create `prompts/summarize.jinja`:

```jinja
You are a helpful assistant that summarizes text.

Summarize the following in {{ num_sentences }} sentences:

{{ content }}
```

Use it:

```python
client = Flashlite(template_dir="./prompts")

response = await client.complete(
    model="gpt-4o",
    template="summarize",
    variables={"content": long_text, "num_sentences": 3}
)
```

### Built-in Filters

```jinja
{# JSON encoding #}
{{ data | json }}

{# Truncation #}
{{ long_text | truncate_words(100) }}
{{ long_text | truncate_chars(500) }}

{# Lists #}
{{ items | bullet_list }}
{{ items | numbered_list }}

{# XML wrapping #}
{{ content | wrap_xml("document") }}
```

## Structured Outputs

Get validated Pydantic models directly from LLM responses:

```python
from pydantic import BaseModel, Field
from typing import Literal
from flashlite import Flashlite

class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0, le=1, description="Confidence score")
    reasoning: str = Field(description="Brief explanation")

client = Flashlite(default_model="gpt-4o")

# Returns a validated Sentiment instance, not a string!
result: Sentiment = await client.complete(
    messages="Analyze the sentiment: 'I absolutely love this product!'",
    response_model=Sentiment,
)

print(f"Sentiment: {result.label} ({result.confidence:.0%})")
print(f"Reasoning: {result.reasoning}")
```

### Nested and Complex Types

```python
from pydantic import BaseModel
from typing import Literal
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Task(BaseModel):
    title: str
    description: str
    priority: Priority
    tags: list[str]

class TaskList(BaseModel):
    tasks: list[Task]
    summary: str

result: TaskList = await client.complete(
    messages="Create a task list for launching a new product",
    response_model=TaskList,
)

for task in result.tasks:
    print(f"[{task.priority.value}] {task.title}")
```

### Validation Retries

If the model returns invalid JSON, Flashlite automatically retries with error feedback:

```python
result = await client.complete(
    messages="...",
    response_model=MyModel,
    structured_retries=2,  # Retry up to 2 times on validation failure
)
```

The retry sends the validation error back to the model so it can correct its response.

## Caching

Cache responses to avoid redundant API calls:

```python
from flashlite import Flashlite, MemoryCache, DiskCache

# In-memory cache (for single process)
client = Flashlite(
    cache=MemoryCache(max_size=1000, default_ttl=3600),
)

# Disk cache (persists across restarts)
client = Flashlite(
    cache=DiskCache("./cache/completions.db", default_ttl=86400),
)

# Same request returns cached response
response1 = await client.complete(model="gpt-4o", messages="Hello", temperature=0)
response2 = await client.complete(model="gpt-4o", messages="Hello", temperature=0)
# response2 is served from cache!

# Check cache stats
stats = await client.cache_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")

# Clear cache
await client.clear_cache()
```

> **Note:** Caching is disabled by default. A warning is emitted when caching with `temperature > 0` or reasoning models since responses may vary.

## Cost Tracking

Monitor token usage and costs:

```python
from flashlite import Flashlite

client = Flashlite(
    track_costs=True,
    budget_limit=10.00,  # USD - raises error if exceeded
)

# Make requests...
await client.complete(model="gpt-4o", messages="Hello")
await client.complete(model="gpt-4o", messages="World")

# Check costs
print(f"Total cost: ${client.total_cost:.4f}")
print(f"Total tokens: {client.total_tokens}")

# Detailed report
report = client.get_cost_report()
print(report)
# {
#   "total_cost_usd": 0.0015,
#   "total_requests": 2,
#   "total_tokens": {"input": 10, "output": 20, "total": 30},
#   "by_model": {"gpt-4o": {"cost_usd": 0.0015, "tokens": {...}}}
# }
```

## Structured Logging

Log all requests/responses to files:

```python
from flashlite import Flashlite, StructuredLogger

logger = StructuredLogger(
    log_file="./logs/completions.jsonl",
    include_messages=True,
    include_content=True,
)

client = Flashlite(structured_logger=logger)

# All requests are logged as JSON lines
await client.complete(model="gpt-4o", messages="Hello")

# Log file contains:
# {"type": "request", "request_id": "...", "model": "gpt-4o", ...}
# {"type": "response", "request_id": "...", "latency_ms": 234, ...}
```

## Reasoning Models

### OpenAI o1/o3

```python
from flashlite import Flashlite

client = Flashlite()

response = await client.complete(
    model="o3",
    messages="Solve this step by step: What is 123 * 456?",
    reasoning_effort="high",  # low, medium, or high
    max_completion_tokens=16000,
)
```

### Anthropic Claude (Extended Thinking)

```python
from flashlite import Flashlite, thinking_enabled

client = Flashlite()

response = await client.complete(
    model="claude-sonnet-4-5-20250929",
    messages="Solve this complex problem...",
    thinking=thinking_enabled(10000),  # budget_tokens
    max_tokens=16000,
)

# Or pass as dict
response = await client.complete(
    model="claude-sonnet-4-5-20250929",
    messages="Another problem...",
    thinking={"type": "enabled", "budget_tokens": 10000},
)
```

## Inspect Framework Integration

Use Flashlite with UK AISI's [Inspect](https://inspect.ai-safety-institute.org.uk/) framework:

```python
from flashlite import Flashlite, InspectLogger, FlashliteModelAPI

# Log in Inspect-compatible format
inspect_logger = InspectLogger(log_dir="./logs", eval_id="my-eval")

client = Flashlite(default_model="gpt-4o")

# Log completions
response = await client.complete(messages="Hello")
inspect_logger.log(request, response, sample_id=0)

# Or use as Inspect ModelAPI adapter
model_api = FlashliteModelAPI(client, model="gpt-4o")
result = await model_api.generate(messages=[...])
```

## Event Callbacks

Hook into request/response lifecycle:

```python
from flashlite import Flashlite, CallbackManager

callbacks = CallbackManager()

@callbacks.on_request
async def log_request(request, request_id):
    print(f"[{request_id}] Sending request to {request.model}")

@callbacks.on_response
async def log_response(response, request_id, latency_ms, cached):
    cache_str = " (cached)" if cached else ""
    print(f"[{request_id}] Got response in {latency_ms:.0f}ms{cache_str}")

@callbacks.on_error
async def log_error(error, request_id, latency_ms):
    print(f"[{request_id}] Error: {error}")

client = Flashlite(callbacks=callbacks)
```

## Full Example: Multi-Agent Simulation

```python
import asyncio
from pydantic import BaseModel, Field
from flashlite import Flashlite, MemoryCache, RateLimitConfig

# Define structured output
class AgentDecision(BaseModel):
    action: str = Field(description="The action to take")
    reasoning: str = Field(description="Why this action")
    confidence: float = Field(ge=0, le=1)

async def run_simulation():
    client = Flashlite(
        default_model="gpt-4o",
        rate_limit=RateLimitConfig(requests_per_minute=60),
        cache=MemoryCache(max_size=1000),
        track_costs=True,
        template_dir="./prompts",
    )
    
    # Run agents in parallel
    agents = ["Alice", "Bob", "Charlie"]
    requests = [
        {
            "template": "agent_decision",
            "variables": {"agent_name": name, "scenario": "..."},
            "temperature": 0.7,
        }
        for name in agents
    ]
    
    responses = await client.complete_many(requests, max_concurrency=10)
    
    for agent, response in zip(agents, responses):
        print(f"{agent}: {response.content}")
    
    print(f"\nTotal cost: ${client.total_cost:.4f}")

asyncio.run(run_simulation())
```

## Message Helpers

```python
from flashlite import user_message, system_message, assistant_message

# Build messages programmatically
messages = [
    system_message("You are a helpful assistant."),
    user_message("Hello!"),
    assistant_message("Hi! How can I help?"),
    user_message("What's the weather?"),
]

response = await client.complete(model="gpt-4o", messages=messages)
```

## Conversation Management

Multi-turn conversations with automatic history tracking:

```python
from flashlite import Flashlite

client = Flashlite(default_model="gpt-4o")

# Create a conversation
conv = client.conversation(system="You are a helpful math tutor.")

# Multi-turn interaction
response1 = await conv.say("What is 2 + 2?")
print(response1.content)  # "4"

response2 = await conv.say("And if I multiply that by 3?")
print(response2.content)  # "12" (has context from previous turn)

# Get full history
for msg in conv.messages:
    print(f"{msg['role']}: {msg['content']}")
```

### Branching Conversations

```python
# Fork for exploration
fork = conv.fork()
response_fork = await fork.say("What about 5 + 5?")

# Original conversation unchanged
response_orig = await conv.say("Divide by 2 instead")
```

### Save and Load State

```python
import json

# Save conversation
state = conv.save()
with open("conversation.json", "w") as f:
    json.dump(state.model_dump(), f)

# Load conversation
with open("conversation.json") as f:
    data = json.load(f)
    
loaded_conv = client.conversation(system=data["system_prompt"])
loaded_conv._messages = data["messages"]  # Restore history
```

### Context Window Management

```python
from flashlite import ContextManager, estimate_tokens, truncate_messages

# Estimate tokens in messages
tokens = estimate_tokens(messages)

# Get model limits
ctx = client.context_manager(model="gpt-4")

# Prepare messages - auto-truncates if needed
prepared = ctx.prepare(very_long_conversation)

# Or manually truncate
truncated = truncate_messages(
    messages, 
    max_tokens=4000, 
    keep_system=True
)
```

## Mixed Model Parallel Processing

Use different models in the same parallel execution:

```python
# Different models for different tasks
requests = [
    {"model": "gpt-4o", "messages": "Complex reasoning task"},
    {"model": "gpt-4o-mini", "messages": "Simple classification"},
    {"model": "claude-3-sonnet-20240229", "messages": "Creative writing"},
]

responses = await client.complete_many(requests, max_concurrency=3)
```

## Multi-Agent Conversations

Create conversations between multiple AI agents:

```python
from flashlite import Flashlite, MultiAgentChat, Agent

client = Flashlite(default_model="gpt-4o-mini")
chat = MultiAgentChat(client)

# Add agents with different personas
chat.add_agent(Agent(
    name="Optimist",
    system_prompt="You see the positive side. Be concise (1-2 sentences)."
))
chat.add_agent(Agent(
    name="Skeptic",
    system_prompt="You question assumptions. Be concise (1-2 sentences)."
))

# Start with a topic
chat.add_message("Moderator", "Discuss: Will AI help or hurt jobs?")

# Have agents take turns
response1 = await chat.speak("Optimist")
print(f"Optimist: {response1}")

response2 = await chat.speak("Skeptic")
print(f"Skeptic: {response2}")
```

### Round-Robin Turns

```python
# Have all agents speak in order for multiple rounds
responses = await chat.round_robin(rounds=3)

# Or specify exact sequence
responses = await chat.speak_sequence(["Optimist", "Skeptic", "Optimist"])
```

### Agent Model Overrides

```python
# Different agents can use different models
chat.add_agent(Agent(
    name="GPT-4 Expert",
    system_prompt="You provide expert analysis.",
    model="gpt-4o",  # Uses GPT-4 for this agent
))
chat.add_agent(Agent(
    name="Quick Responder",
    system_prompt="You give fast, brief responses.",
    model="gpt-4o-mini",  # Uses cheaper model
))
```

### Transcript and History

```python
# Get formatted transcript
print(chat.format_transcript())

# Get messages from an agent's perspective
alice_view = chat.get_messages_for("Alice")

# Clear and start over
chat.clear()
```

## Tool/Function Calling

Define tools using the `@tool` decorator and let the model call them:

```python
from flashlite import Flashlite, tool, run_tool_loop

# Define tools with the @tool decorator
@tool()
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location."""
    # Real implementation would call a weather API
    return f"Weather in {location}: 22°{unit[0].upper()}, partly cloudy"

@tool()
def search_web(query: str, max_results: int = 5) -> list[str]:
    """Search the web for information."""
    return [f"Result {i}: Information about {query}" for i in range(max_results)]

client = Flashlite(default_model="gpt-4o")

# Simple tool usage - pass tools and let the model decide
response = await client.complete(
    messages="What's the weather like in Paris?",
    tools=[get_weather, search_web],
)
```

### Tool Execution Loop

For agentic patterns where the model can call multiple tools:

```python
from flashlite import run_tool_loop

# Run the tool loop - automatically executes tools and continues
result = await run_tool_loop(
    client=client,
    messages=[{"role": "user", "content": "What's the weather in NYC and Tokyo?"}],
    tools=[get_weather, search_web],
    max_iterations=5,  # Maximum tool call rounds
)

print(result.content)  # Final response after tool execution
print(f"Tool calls made: {len(result.tool_calls_made)}")
```

### Pydantic-based Tool Definition

For more complex tools, use Pydantic models for parameter schemas:

```python
from pydantic import BaseModel, Field
from flashlite import tool_from_pydantic

class SearchParams(BaseModel):
    """Search for information."""
    query: str = Field(description="The search query")
    max_results: int = Field(default=10, ge=1, le=100)
    include_images: bool = Field(default=False)

async def do_search(query: str, max_results: int, include_images: bool) -> dict:
    # Implementation
    return {"results": [...], "images": [...]}

search_tool = tool_from_pydantic(SearchParams, do_search)
```

### Tool Call Callbacks

Monitor tool execution with callbacks:

```python
from flashlite.tools import ToolCall, ToolResult

def on_tool_call(tc: ToolCall):
    print(f"Calling tool: {tc.name} with args: {tc.arguments}")

def on_tool_result(tr: ToolResult):
    if tr.error:
        print(f"Tool {tr.name} failed: {tr.error}")
    else:
        print(f"Tool {tr.name} returned: {tr.result}")

result = await run_tool_loop(
    client=client,
    messages=[{"role": "user", "content": "Search for Python tutorials"}],
    tools=[search_web],
    on_tool_call=on_tool_call,
    on_tool_result=on_tool_result,
)
```

## Error Handling

```python
from flashlite import (
    Flashlite,
    FlashliteError,
    CompletionError,
    RateLimitError,
    BudgetExceededError,
)

client = Flashlite(track_costs=True, budget_limit=1.0)

try:
    response = await client.complete(model="gpt-4o", messages="Hello")
except RateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after}s")
except BudgetExceededError as e:
    print(f"Budget exceeded: {e}")
except CompletionError as e:
    print(f"API error: {e.status_code} - {e}")
except FlashliteError as e:
    print(f"Flashlite error: {e}")
```

## Next Steps

- See [DEV.md](DEV.md) for development setup
- See [plan.md](plan.md) for the full implementation roadmap
- Check the `tests/` directory for more usage examples
