"""Main Flashlite client class."""

import asyncio
import logging
from pathlib import Path
from typing import Any, TypeVar, overload

from pydantic import BaseModel

from ._spinner import Spinner
from .cache import CacheBackend, MemoryCache
from .config import FlashliteConfig, load_env_files
from .conversation import ContextManager, Conversation
from .core.completion import complete as core_complete
from .core.messages import format_messages
from .middleware.base import Middleware, MiddlewareChain
from .middleware.cache import CacheMiddleware
from .middleware.logging import LoggingMiddleware
from .middleware.rate_limit import ConcurrencyLimiter, RateLimitMiddleware
from .middleware.retry import RetryMiddleware
from .observability.callbacks import CallbackManager
from .observability.logging import StructuredLogger
from .observability.metrics import CostTracker
from .structured import (
    StructuredOutputError,
    format_validation_error_for_retry,
    schema_to_prompt,
    validate_response,
)
from .templating.engine import TemplateEngine
from .tools import ToolDefinition, tools_to_anthropic, tools_to_openai
from .types import (
    CompletionRequest,
    CompletionResponse,
    Messages,
    RateLimitConfig,
    RetryConfig,
    ThinkingConfig,
)

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class Flashlite:
    """
    Batteries-included LLM client wrapping litellm.

    Features:
    - Automatic retries with exponential backoff
    - Rate limiting (RPM and TPM)
    - Jinja templating for prompts
    - Async-first with sync wrappers
    - Full passthrough of provider kwargs
    """

    def __init__(
        self,
        # Environment
        env_file: str | Path | None = None,
        env_files: list[str | Path] | None = None,
        # Configuration
        config: FlashliteConfig | None = None,
        default_model: str | None = None,
        # Middleware configs
        retry: RetryConfig | None = None,
        rate_limit: RateLimitConfig | None = None,
        # Caching (disabled by default)
        cache: CacheBackend | None = None,
        cache_ttl: float | None = None,
        # Templating
        template_dir: str | Path | None = None,
        # Logging & Observability
        log_requests: bool = False,
        log_level: str = "WARNING",
        structured_logger: StructuredLogger | None = None,
        # Cost tracking
        track_costs: bool = False,
        budget_limit: float | None = None,
        # Callbacks
        callbacks: CallbackManager | None = None,
        # Defaults
        default_kwargs: dict[str, Any] | None = None,
        timeout: float = 600.0,
    ):
        """
        Initialize the Flashlite client.

        Args:
            env_file: Path to .env file to load
            env_files: Multiple .env files to load (later overrides earlier)
            config: Full configuration object (overrides individual params)
            default_model: Default model to use if not specified per-request
            retry: Retry configuration
            rate_limit: Rate limiting configuration
            cache: Cache backend (None = disabled, pass MemoryCache or DiskCache to enable)
            cache_ttl: Default TTL for cached entries (seconds)
            template_dir: Directory containing prompt templates
            log_requests: Whether to log requests/responses
            log_level: Logging level
            structured_logger: Structured logger for detailed logging to files
            track_costs: Whether to track token costs
            budget_limit: Maximum budget in USD (requires track_costs=True)
            callbacks: Callback manager for event hooks
            default_kwargs: Default kwargs for all completions
            timeout: Request timeout in seconds

        Note:
            Caching is disabled by default. When enabled with temperature > 0 or
            reasoning models, a warning will be emitted since responses may vary.
        """
        # Load environment files
        load_env_files(env_file, env_files)

        # Build configuration
        if config:
            self._config = config
        else:
            # Start with env-based config, then override with explicit params
            self._config = FlashliteConfig.from_env()

            if default_model:
                self._config.default_model = default_model
            if retry:
                self._config.retry = retry
            if rate_limit:
                self._config.rate_limit = rate_limit
            if template_dir:
                self._config.template_dir = template_dir
            if log_requests:
                self._config.log_requests = log_requests
            if log_level != "INFO":
                self._config.log_level = log_level
            if default_kwargs:
                self._config.default_kwargs = default_kwargs
            if timeout != 600.0:
                self._config.timeout = timeout

        # Setup logging
        logging.basicConfig(level=getattr(logging, self._config.log_level))

        # Store observability components
        self._cache = cache
        self._cache_ttl = cache_ttl
        self._structured_logger = structured_logger
        self._callbacks = callbacks

        # Cost tracking
        self._cost_tracker: CostTracker | None = None
        if track_costs or budget_limit is not None:
            self._cost_tracker = CostTracker(budget_limit=budget_limit)

        # Emit info about caching status
        if cache is None:
            logger.info(
                "Caching is disabled. To enable, pass cache=MemoryCache() or "
                "cache=DiskCache('./cache.db') to the Flashlite client."
            )

        # Initialize template engine
        self._template_engine: TemplateEngine | None = None
        if self._config.template_dir:
            self._template_engine = TemplateEngine(self._config.template_dir)

        # Build middleware chain
        self._middleware = self._build_middleware()

    def _build_middleware(self) -> list[Middleware]:
        """Build the middleware stack.

        Middleware order (outermost to innermost):
        1. Logging - tracks timing and emits events
        2. Retry - handles transient failures
        3. Cache - returns cached responses (skips inner middleware on hit)
        4. Rate limiting - controls request rate
        """
        middleware: list[Middleware] = []

        # Logging middleware (outermost - captures full timing)
        if (
            self._config.log_requests
            or self._structured_logger
            or self._cost_tracker
            or self._callbacks
        ):
            middleware.append(
                LoggingMiddleware(
                    structured_logger=self._structured_logger,
                    cost_tracker=self._cost_tracker,
                    callbacks=self._callbacks,
                    log_level=self._config.log_level,
                )
            )

        # Retry middleware
        middleware.append(RetryMiddleware(self._config.retry))

        # Cache middleware (after retry - retries apply to cache misses)
        if self._cache is not None:
            middleware.append(
                CacheMiddleware(
                    backend=self._cache,
                    ttl=self._cache_ttl,
                    warn_non_deterministic=True,
                )
            )

        # Rate limiting middleware (innermost - rate limiting is per-request)
        if self._config.rate_limit.requests_per_minute or self._config.rate_limit.tokens_per_minute:
            middleware.append(RateLimitMiddleware(self._config.rate_limit))

        return middleware

    def _get_chain(self) -> MiddlewareChain:
        """Get the middleware chain with the core completion handler."""
        return MiddlewareChain(self._middleware, self._core_complete)

    async def _core_complete(self, request: CompletionRequest) -> CompletionResponse:
        """Core completion handler - calls litellm."""
        if self._config.log_requests:
            logger.info(f"Completion request: model={request.model}")

        async with Spinner(f"Waiting for {request.model}...", delay=0.2):
            response = await core_complete(request)

        if self._config.log_requests:
            logger.info(
                f"Completion response: model={response.model}, "
                f"tokens={response.usage.total_tokens if response.usage else 'N/A'}"
            )

        return response

    @overload
    async def complete(
        self,
        model: str | None = None,
        messages: Messages | str | None = None,
        *,
        response_model: None = None,
        template: str | None = None,
        variables: dict[str, Any] | None = None,
        system: str | None = None,
        user: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        reasoning_effort: str | None = None,
        thinking: ThinkingConfig | None = None,
        tools: list[ToolDefinition | Any] | None = None,
        structured_retries: int = 1,
        **kwargs: Any,
    ) -> CompletionResponse: ...

    @overload
    async def complete(
        self,
        model: str | None = None,
        messages: Messages | str | None = None,
        *,
        response_model: type[T],
        template: str | None = None,
        variables: dict[str, Any] | None = None,
        system: str | None = None,
        user: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        reasoning_effort: str | None = None,
        thinking: ThinkingConfig | None = None,
        tools: list[ToolDefinition | Any] | None = None,
        structured_retries: int = 1,
        **kwargs: Any,
    ) -> T: ...

    async def complete(
        self,
        model: str | None = None,
        messages: Messages | str | None = None,
        *,
        # Structured outputs
        response_model: type[T] | None = None,
        # Template support
        template: str | None = None,
        variables: dict[str, Any] | None = None,
        # Message building
        system: str | None = None,
        user: str | None = None,
        # Common params
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        # OpenAI reasoning model params (o1, o3)
        reasoning_effort: str | None = None,
        # Anthropic extended thinking params (Claude)
        thinking: ThinkingConfig | None = None,
        # Tool/function calling
        tools: list[ToolDefinition | Any] | None = None,
        # Structured output retries
        structured_retries: int = 1,
        # Additional kwargs passed through to litellm
        **kwargs: Any,
    ) -> CompletionResponse | T:
        """
        Make a completion request.

        Args:
            model: Model identifier (uses default if not specified)
            messages: Messages list, or single string (becomes user message)
            response_model: Pydantic model class for structured output parsing.
                           When provided, returns a validated instance of the model.
            template: Template name to render (from template_dir or registered)
            variables: Variables for template rendering
            system: System prompt (prepended to messages)
            user: User message (appended to messages)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            max_completion_tokens: Max completion tokens (for reasoning models)
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            reasoning_effort: OpenAI reasoning effort level (low/medium/high) for o1/o3
            thinking: Anthropic extended thinking config for Claude models.
                     Use thinking_enabled(budget_tokens) helper or pass dict directly.
            tools: List of tools for function calling. Accepts @tool decorated functions
                  or ToolDefinition objects. Auto-converts to provider format.
            structured_retries: Number of retries for structured output validation (default: 1)
            **kwargs: Additional provider-specific parameters

        Returns:
            CompletionResponse if no response_model, or validated model instance
        """
        # Resolve model
        resolved_model = model or self._config.default_model
        if not resolved_model:
            raise ValueError("No model specified and no default_model configured")

        # Build messages
        if template:
            if not self._template_engine:
                raise ValueError("Template specified but no template_dir configured")
            rendered = self._template_engine.render(template, variables)
            # Template renders to user message content
            final_messages = format_messages(messages=rendered, system=system)
        else:
            final_messages = format_messages(messages=messages, system=system, user=user)

        if not final_messages:
            raise ValueError("No messages provided")

        # Handle structured outputs
        effective_system = system
        if response_model is not None:
            # Inject JSON schema into system prompt
            schema_prompt = schema_to_prompt(response_model)
            if effective_system:
                effective_system = f"{effective_system}\n\n{schema_prompt}"
            else:
                effective_system = schema_prompt

            # Rebuild messages with schema in system prompt
            if template:
                rendered = self._template_engine.render(template, variables)  # type: ignore
                final_messages = format_messages(messages=rendered, system=effective_system)
            else:
                final_messages = format_messages(
                    messages=messages, system=effective_system, user=user
                )

            # Enable JSON mode for supported providers
            if "response_format" not in kwargs:
                # Check if model supports JSON mode
                model_lower = resolved_model.lower()
                if any(
                    p in model_lower for p in ["gpt-4", "gpt-3.5", "claude", "gemini", "mistral"]
                ):
                    kwargs["response_format"] = {"type": "json_object"}

        # Build extra kwargs (merge defaults with per-request)
        extra_kwargs = {**self._config.default_kwargs, **kwargs}

        # Handle tools - convert to provider format if provided
        if tools is not None and "tools" not in extra_kwargs:
            model_lower = resolved_model.lower()
            if "claude" in model_lower or "anthropic" in model_lower:
                extra_kwargs["tools"] = tools_to_anthropic(tools)
            else:
                extra_kwargs["tools"] = tools_to_openai(tools)

        # Build request (template/variables stored for middleware traceability)
        request = CompletionRequest(
            model=resolved_model,
            messages=final_messages,
            template=template,
            variables=variables,
            temperature=temperature,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            stop=stop,
            reasoning_effort=reasoning_effort,  # type: ignore
            thinking=thinking,
            extra_kwargs=extra_kwargs,
        )

        # Execute through middleware chain
        chain = self._get_chain()
        response = await chain(request)

        # If no response model, return raw response
        if response_model is None:
            return response

        # Validate structured output with retries
        last_error: StructuredOutputError | None = None
        current_messages = list(final_messages)

        for attempt in range(structured_retries + 1):
            try:
                return validate_response(response, response_model)
            except StructuredOutputError as e:
                last_error = e
                logger.warning(f"Structured output validation failed (attempt {attempt + 1}): {e}")

                # If we have retries left, ask the model to fix it
                if attempt < structured_retries:
                    # Add the failed response and error feedback
                    error_feedback = format_validation_error_for_retry(e)
                    current_messages.append({"role": "assistant", "content": response.content})
                    current_messages.append({"role": "user", "content": error_feedback})

                    # Make another request
                    retry_request = CompletionRequest(
                        model=resolved_model,
                        messages=current_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        max_completion_tokens=max_completion_tokens,
                        top_p=top_p,
                        stop=stop,
                        reasoning_effort=reasoning_effort,  # type: ignore
                        thinking=thinking,
                        extra_kwargs=extra_kwargs,
                    )
                    response = await chain(retry_request)

        # All retries exhausted
        raise last_error  # type: ignore

    def complete_sync(
        self,
        model: str | None = None,
        messages: Messages | str | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Synchronous version of complete().

        See complete() for full parameter documentation.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're in an async context - use thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.complete(model=model, messages=messages, **kwargs),
                )
                return future.result()
        else:
            return asyncio.run(self.complete(model=model, messages=messages, **kwargs))

    async def complete_many(
        self,
        requests: list[dict[str, Any]],
        max_concurrency: int = 10,
    ) -> list[CompletionResponse | T]:
        """
        Execute multiple completion requests in parallel with concurrency control.

        Each request can use a different model, making this suitable for
        multi-agent scenarios where agents may use different models.

        Note: This is NOT the same as OpenAI/Anthropic "Batch API" which processes
        requests asynchronously over hours. This executes requests immediately
        in parallel with controlled concurrency.

        Args:
            requests: List of kwargs dicts for complete(). Each can specify
                     its own model, messages, response_model, etc.
            max_concurrency: Maximum concurrent requests (default: 10)

        Returns:
            List of responses in same order as requests. Type depends on
            whether response_model was specified in each request.

        Example:
            # Different models in same batch
            responses = await client.complete_many([
                {"model": "gpt-4o", "messages": "Hello from GPT-4"},
                {"model": "claude-sonnet-4-20250514", "messages": "Hello from Claude"},
                {"model": "gpt-4o-mini", "messages": "Hello from mini"},
            ])

            # With structured outputs
            responses = await client.complete_many([
                {"messages": "Analyze: good", "response_model": Sentiment},
                {"messages": "Analyze: bad", "response_model": Sentiment},
            ])
        """
        limiter = ConcurrencyLimiter(max_concurrency)

        async def process_one(req_kwargs: dict[str, Any]) -> CompletionResponse | T:
            async with limiter:
                return await self.complete(**req_kwargs)

        tasks = [process_one(req) for req in requests]
        return await asyncio.gather(*tasks)

    # Template management
    def register_template(self, name: str, template: str) -> None:
        """Register an in-memory template."""
        if not self._template_engine:
            self._template_engine = TemplateEngine()
        self._template_engine.register(name, template)

    def render_template(
        self,
        template: str,
        variables: dict[str, Any] | None = None,
    ) -> str:
        """Render a template without making a completion."""
        if not self._template_engine:
            self._template_engine = TemplateEngine()
        return self._template_engine.render(template, variables)

    # Conversation management
    def conversation(
        self,
        system: str | None = None,
        model: str | None = None,
        max_turns: int | None = None,
    ) -> Conversation:
        """
        Create a new conversation for multi-turn interactions.

        Args:
            system: System prompt for the conversation
            model: Default model (overrides client default)
            max_turns: Maximum turns to keep in history (None = unlimited)

        Returns:
            A new Conversation instance bound to this client

        Example:
            conv = client.conversation(system="You are helpful.")
            response1 = await conv.say("What is Python?")
            response2 = await conv.say("How do I install it?")

            # Fork for exploration
            branch = conv.fork()
            alt = await branch.say("What about JavaScript?")
        """
        return Conversation(
            client=self,
            system=system,
            model=model,
            max_turns=max_turns,
        )

    def context_manager(
        self,
        model: str | None = None,
        max_response_tokens: int = 4096,
        auto_truncate: bool = True,
    ) -> ContextManager:
        """
        Create a context manager for manual context window control.

        Args:
            model: Model to get limits for (uses default if not specified)
            max_response_tokens: Expected max tokens in response
            auto_truncate: Whether to auto-truncate when preparing messages

        Returns:
            A ContextManager for the specified model
        """
        effective_model = model or self._config.default_model
        if not effective_model:
            raise ValueError("No model specified and no default_model configured")
        return ContextManager(
            model=effective_model,
            max_response_tokens=max_response_tokens,
            auto_truncate=auto_truncate,
        )

    # Properties
    @property
    def config(self) -> FlashliteConfig:
        """Get the current configuration."""
        return self._config

    @property
    def template_engine(self) -> TemplateEngine | None:
        """Get the template engine."""
        return self._template_engine

    @property
    def cache(self) -> CacheBackend | None:
        """Get the cache backend (None if caching disabled)."""
        return self._cache

    @property
    def cost_tracker(self) -> CostTracker | None:
        """Get the cost tracker (None if cost tracking disabled)."""
        return self._cost_tracker

    @property
    def total_cost(self) -> float:
        """Get total cost in USD (0.0 if cost tracking disabled)."""
        return self._cost_tracker.total_cost if self._cost_tracker else 0.0

    @property
    def total_tokens(self) -> int:
        """Get total tokens used (0 if cost tracking disabled)."""
        return self._cost_tracker.total_tokens if self._cost_tracker else 0

    def get_cost_report(self) -> dict[str, Any] | None:
        """Get a detailed cost report (None if cost tracking disabled)."""
        return self._cost_tracker.get_report() if self._cost_tracker else None

    async def clear_cache(self) -> int:
        """
        Clear all cached entries.

        Returns:
            Number of entries cleared (0 if caching disabled)
        """
        if self._cache is not None:
            return await self._cache.clear()
        return 0

    async def cache_stats(self) -> dict[str, Any] | None:
        """
        Get cache statistics.

        Returns:
            Cache stats dict, or None if caching disabled
        """
        if self._cache is not None:
            if isinstance(self._cache, MemoryCache):
                return self._cache.stats()
            # For disk cache or other backends
            return {
                "size": await self._cache.size(),
            }
        return None
