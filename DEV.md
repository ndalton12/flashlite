# Flashlite Development Guide

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/flashlite.git
cd flashlite

# Install dependencies (creates virtualenv automatically)
uv sync --all-extras

# Activate the virtual environment (optional, uv run handles this)
source .venv/bin/activate
```

## Common Commands

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_client.py

# Run specific test class or function
uv run pytest tests/test_client.py::TestFlashliteClient::test_complete_with_messages

# Run with coverage
uv run pytest --cov=flashlite --cov-report=html

# Run tests matching a pattern
uv run pytest -k "cache"
```

### Linting & Formatting

```bash
# Run ruff linter
uv run ruff check src/

# Auto-fix linting issues
uv run ruff check src/ --fix

# Format code
uv run ruff format src/

# Check formatting without changing files
uv run ruff format src/ --check
```

### Type Checking

```bash
# Run mypy
uv run mypy src/flashlite
```

### Running All Checks

```bash
# Run linting, type checking, and tests
uv run ruff check src/ && uv run mypy src/flashlite && uv run pytest
```

## Project Structure

```
flashlite/
├── src/flashlite/          # Main package
│   ├── __init__.py         # Public API exports
│   ├── client.py           # Main Flashlite client
│   ├── config.py           # Configuration and .env loading
│   ├── types.py            # Type definitions
│   ├── core/               # Core completion logic
│   ├── middleware/         # Request/response middleware
│   ├── templating/         # Jinja templating
│   ├── cache/              # Caching backends
│   ├── observability/      # Logging, metrics, Inspect compat
│   ├── structured/         # Pydantic structured outputs
│   ├── conversation/       # Multi-turn conversations, context & multi-agent
│   │   ├── manager.py      # Single-agent Conversation class
│   │   ├── context.py      # Context window management
│   │   └── multi_agent.py  # MultiAgentChat for agent-to-agent conversations
│   └── tools/              # Tool/function calling helpers
├── tests/                  # Test suite
├── examples/               # Example scripts
│   ├── basic_example.py    # Basic features demo
│   └── multi_agent_chat.py # Multi-agent conversation example
├── pyproject.toml          # Project configuration
├── plan.md                 # Implementation plan
└── README.md               # User documentation
```

## Running Examples

```bash
# Run basic example
uv run python examples/basic_example.py

# Run multi-agent chat example
uv run python examples/multi_agent_chat.py
```

## Testing Without API Keys

Flashlite tests use litellm's `mock_response` feature to avoid hitting real APIs:

```python
async def test_example():
    client = Flashlite(default_model="gpt-4o")
    
    # mock_response returns a fake response without API calls
    response = await client.complete(
        messages="Hello",
        mock_response="This is a mocked response",
    )
    
    assert response.content == "This is a mocked response"
```

## Adding New Features

1. **Create a branch** for your feature
2. **Write tests first** in the appropriate test file
3. **Implement the feature** in the relevant module
4. **Run linting and tests** to ensure quality
5. **Update documentation** if needed

### Middleware Pattern

New features that process requests/responses should use the middleware pattern:

```python
from flashlite.middleware.base import Middleware
from flashlite.types import CompletionRequest, CompletionResponse

class MyMiddleware(Middleware):
    async def __call__(
        self,
        request: CompletionRequest,
        next_handler: Callable[[CompletionRequest], Awaitable[CompletionResponse]],
    ) -> CompletionResponse:
        # Pre-processing
        print(f"Processing request for {request.model}")
        
        # Call next middleware/handler
        response = await next_handler(request)
        
        # Post-processing
        print(f"Got response with {response.usage.total_tokens} tokens")
        
        return response
```

## Environment Variables

Create a `.env` file for local development:

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Flashlite defaults (optional)
FLASHLITE_DEFAULT_MODEL=gpt-4o
FLASHLITE_LOG_LEVEL=DEBUG
```

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

client = Flashlite(log_level="DEBUG", log_requests=True)
```

### Inspect Middleware Chain

```python
client = Flashlite(...)
print(f"Middleware stack: {client._middleware}")
```

## Release Process

1. Update version in `src/flashlite/__init__.py`
2. Update `pyproject.toml` version
3. Run full test suite
4. Create git tag
5. Build and publish:

```bash
uv build
uv publish
```
