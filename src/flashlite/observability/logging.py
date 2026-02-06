"""Structured logging for flashlite."""

import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from ..types import CompletionRequest, CompletionResponse


@dataclass
class RequestLogEntry:
    """A structured log entry for a completion request."""

    request_id: str
    timestamp: str
    model: str
    messages: list[dict[str, Any]]
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": "request",
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "model": self.model,
            "messages": self.messages,
            "parameters": self.parameters,
        }


@dataclass
class ResponseLogEntry:
    """A structured log entry for a completion response."""

    request_id: str
    timestamp: str
    model: str
    content: str
    finish_reason: str | None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    cached: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": "response",
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "model": self.model,
            "content": self.content,
            "finish_reason": self.finish_reason,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
            },
            "latency_ms": self.latency_ms,
            "cached": self.cached,
            "error": self.error,
        }


class StructuredLogger:
    """
    A structured logger that outputs JSON-formatted log entries.

    Can write to files, stdout, or both. Supports log rotation
    and customizable formatting.

    Example:
        logger = StructuredLogger(
            log_file="./logs/completions.jsonl",
            log_level="INFO",
            include_messages=True,
        )

        # Log a request
        logger.log_request(request, request_id)

        # Log a response
        logger.log_response(response, request_id, latency_ms)
    """

    def __init__(
        self,
        log_file: str | Path | None = None,
        log_level: str = "INFO",
        include_messages: bool = True,
        include_content: bool = True,
        max_content_length: int | None = None,
        redact_patterns: list[str] | None = None,
        stdout: bool = False,
    ):
        """
        Initialize the structured logger.

        Args:
            log_file: Path to log file (JSONL format). None disables file logging.
            log_level: Minimum log level ("DEBUG", "INFO", "WARNING", "ERROR")
            include_messages: Whether to include full message content in logs
            include_content: Whether to include response content in logs
            max_content_length: Max length of content to log (None = unlimited)
            redact_patterns: Patterns to redact from logs (e.g., API keys)
            stdout: Whether to also log to stdout
        """
        self._log_file: Path | None = Path(log_file) if log_file else None
        self._log_level = getattr(logging, log_level.upper())
        self._include_messages = include_messages
        self._include_content = include_content
        self._max_content_length = max_content_length
        self._redact_patterns = redact_patterns or []
        self._stdout = stdout
        self._file_handle: TextIO | None = None

        # Ensure log directory exists
        if self._log_file:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(self._log_file, "a")

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now(UTC).isoformat()

    def _redact(self, text: str) -> str:
        """Redact sensitive patterns from text."""
        for pattern in self._redact_patterns:
            text = text.replace(pattern, "[REDACTED]")
        return text

    def _truncate(self, text: str) -> str:
        """Truncate text if max length is set."""
        if self._max_content_length and len(text) > self._max_content_length:
            return text[: self._max_content_length] + "... [truncated]"
        return text

    def _write_entry(self, entry: dict[str, Any]) -> None:
        """Write a log entry."""
        json_str = json.dumps(entry, default=str)

        if self._file_handle:
            self._file_handle.write(json_str + "\n")
            self._file_handle.flush()

        if self._stdout:
            print(json_str, file=sys.stdout)

    def log_request(
        self,
        request: CompletionRequest,
        request_id: str | None = None,
    ) -> str:
        """
        Log a completion request.

        Args:
            request: The completion request
            request_id: Optional request ID (generated if not provided)

        Returns:
            The request ID
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        # Build parameters dict
        params: dict[str, Any] = {}
        if request.template is not None:
            params["template"] = request.template
        if request.variables is not None:
            params["variables"] = request.variables
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        if request.max_completion_tokens is not None:
            params["max_completion_tokens"] = request.max_completion_tokens
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.stop is not None:
            params["stop"] = request.stop
        if request.reasoning_effort is not None:
            params["reasoning_effort"] = request.reasoning_effort
        if request.thinking is not None:
            params["thinking"] = request.thinking
        params.update(request.extra_kwargs)

        # Build messages
        messages: list[dict[str, Any]] = []
        if self._include_messages:
            for msg in request.messages:
                msg_dict = dict(msg)
                if "content" in msg_dict:
                    content = self._truncate(self._redact(str(msg_dict["content"])))
                    msg_dict["content"] = content
                messages.append(msg_dict)

        entry = RequestLogEntry(
            request_id=request_id,
            timestamp=self._get_timestamp(),
            model=request.model,
            messages=messages,
            parameters=params,
        )

        self._write_entry(entry.to_dict())
        return request_id

    def log_response(
        self,
        response: CompletionResponse,
        request_id: str,
        latency_ms: float,
        cached: bool = False,
    ) -> None:
        """
        Log a completion response.

        Args:
            response: The completion response
            request_id: The corresponding request ID
            latency_ms: Request latency in milliseconds
            cached: Whether the response was from cache
        """
        content = ""
        if self._include_content:
            content = self._truncate(self._redact(response.content))

        entry = ResponseLogEntry(
            request_id=request_id,
            timestamp=self._get_timestamp(),
            model=response.model,
            content=content,
            finish_reason=response.finish_reason,
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
            latency_ms=latency_ms,
            cached=cached,
        )

        self._write_entry(entry.to_dict())

    def log_error(
        self,
        request_id: str,
        error: Exception,
        latency_ms: float,
    ) -> None:
        """
        Log an error response.

        Args:
            request_id: The corresponding request ID
            error: The exception that occurred
            latency_ms: Request latency in milliseconds
        """
        entry = {
            "type": "error",
            "request_id": request_id,
            "timestamp": self._get_timestamp(),
            "error": str(error),
            "error_type": type(error).__name__,
            "latency_ms": latency_ms,
        }

        self._write_entry(entry)

    def close(self) -> None:
        """Close the log file."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None


@dataclass
class RequestContext:
    """Context for tracking a single request through the pipeline."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = field(default_factory=time.perf_counter)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.perf_counter() - self.start_time) * 1000
