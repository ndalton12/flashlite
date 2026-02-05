"""Core completion logic using litellm."""

import asyncio
from typing import Any

import litellm

from ..types import (
    CompletionError,
    CompletionRequest,
    CompletionResponse,
    Messages,
    UsageInfo,
)


async def complete(request: CompletionRequest) -> CompletionResponse:
    """
    Execute a completion request using litellm.

    This is the lowest-level async completion function.
    It handles the litellm API call and response parsing.

    Args:
        request: The completion request

    Returns:
        CompletionResponse with the model's response

    Raises:
        CompletionError: If the API call fails
    """
    kwargs = request.to_litellm_kwargs()

    try:
        # litellm.acompletion is the async version
        response = await litellm.acompletion(**kwargs)

        # Extract content from response
        content = ""
        finish_reason = None

        if response.choices:
            choice = response.choices[0]
            if choice.message and choice.message.content:
                content = choice.message.content
            finish_reason = getattr(choice, "finish_reason", None)

        # Parse usage info
        usage = UsageInfo.from_litellm(
            response.usage.model_dump() if response.usage else None
        )

        return CompletionResponse(
            content=content,
            model=response.model or request.model,
            finish_reason=finish_reason,
            usage=usage,
            raw_response=response,
        )

    except litellm.exceptions.APIConnectionError as e:
        raise CompletionError(f"API connection error: {e}", response=e) from e
    except litellm.exceptions.RateLimitError as e:
        raise CompletionError(
            f"Rate limit exceeded: {e}",
            status_code=429,
            response=e,
        ) from e
    except litellm.exceptions.APIError as e:
        raise CompletionError(
            f"API error: {e}",
            status_code=getattr(e, "status_code", None),
            response=e,
        ) from e
    except Exception as e:
        raise CompletionError(f"Completion failed: {e}", response=e) from e


def complete_sync(request: CompletionRequest) -> CompletionResponse:
    """
    Synchronous version of complete().

    Runs the async completion in an event loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're already in an async context - need to use a new thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, complete(request))
            return future.result()
    else:
        # No running loop, we can use asyncio.run
        return asyncio.run(complete(request))


async def complete_simple(
    model: str,
    messages: Messages,
    **kwargs: Any,
) -> CompletionResponse:
    """
    Simplified completion function for quick usage.

    Args:
        model: Model identifier
        messages: List of messages
        **kwargs: Additional parameters passed to CompletionRequest

    Returns:
        CompletionResponse
    """
    # Separate known kwargs from extra kwargs
    known_params = {
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "top_p",
        "stop",
        "reasoning_effort",
    }

    request_kwargs: dict[str, Any] = {}
    extra_kwargs: dict[str, Any] = {}

    for key, value in kwargs.items():
        if key in known_params:
            request_kwargs[key] = value
        else:
            extra_kwargs[key] = value

    request = CompletionRequest(
        model=model,
        messages=messages,
        extra_kwargs=extra_kwargs,
        **request_kwargs,
    )

    return await complete(request)
