"""Tests for observability functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from flashlite.observability import (
    BudgetExceededError,
    CallbackManager,
    CostTracker,
    InspectLogger,
    StructuredLogger,
)
from flashlite.types import CompletionRequest, CompletionResponse, UsageInfo


# Fixtures
@pytest.fixture
def sample_request() -> CompletionRequest:
    """Create a sample completion request."""
    return CompletionRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, world!"}],
        temperature=0.7,
    )


@pytest.fixture
def sample_response() -> CompletionResponse:
    """Create a sample completion response."""
    return CompletionResponse(
        content="Hello! How can I help you today?",
        model="gpt-4o",
        finish_reason="stop",
        usage=UsageInfo(input_tokens=10, output_tokens=20, total_tokens=30),
    )


@pytest.fixture
def temp_log_file() -> Path:
    """Create a temporary log file path."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        return Path(f.name)


@pytest.fixture
def temp_log_dir() -> Path:
    """Create a temporary log directory."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# Cost Tracker Tests
class TestCostTracker:
    """Tests for cost tracking."""

    def test_track_response(self, sample_response: CompletionResponse):
        """Should track response costs."""
        tracker = CostTracker()
        cost = tracker.track(sample_response)

        assert cost > 0
        assert tracker.total_cost > 0
        assert tracker.total_tokens == 30
        assert tracker.total_requests == 1

    def test_accumulates_costs(self, sample_response: CompletionResponse):
        """Should accumulate costs across multiple responses."""
        tracker = CostTracker()
        cost1 = tracker.track(sample_response)
        cost2 = tracker.track(sample_response)

        assert tracker.total_cost == pytest.approx(cost1 + cost2)
        assert tracker.total_requests == 2

    def test_per_model_tracking(self):
        """Should track costs per model."""
        tracker = CostTracker()

        response1 = CompletionResponse(
            content="Hello",
            model="gpt-4o",
            usage=UsageInfo(input_tokens=10, output_tokens=20, total_tokens=30),
        )
        response2 = CompletionResponse(
            content="World",
            model="gpt-4o-mini",
            usage=UsageInfo(input_tokens=10, output_tokens=20, total_tokens=30),
        )

        tracker.track(response1)
        tracker.track(response2)

        report = tracker.get_report()
        assert "gpt-4o" in report["by_model"]
        assert "gpt-4o-mini" in report["by_model"]

    def test_budget_remaining(self, sample_response: CompletionResponse):
        """Should calculate remaining budget."""
        tracker = CostTracker(budget_limit=1.0)
        tracker.track(sample_response)

        assert tracker.budget_remaining is not None
        assert tracker.budget_remaining < 1.0
        assert tracker.budget_remaining > 0

    def test_budget_exceeded_error(self):
        """Should raise error when budget is exceeded."""
        tracker = CostTracker(budget_limit=0.00001)  # Very small budget

        response = CompletionResponse(
            content="Hello",
            model="gpt-4o",
            usage=UsageInfo(input_tokens=1000, output_tokens=1000, total_tokens=2000),
        )

        with pytest.raises(BudgetExceededError):
            tracker.track(response)

    def test_reset(self, sample_response: CompletionResponse):
        """Should reset all metrics."""
        tracker = CostTracker()
        tracker.track(sample_response)
        tracker.reset()

        assert tracker.total_cost == 0
        assert tracker.total_tokens == 0
        assert tracker.total_requests == 0

    def test_report_structure(self, sample_response: CompletionResponse):
        """Report should have expected structure."""
        tracker = CostTracker(budget_limit=10.0)
        tracker.track(sample_response)

        report = tracker.get_report()
        assert "total_cost_usd" in report
        assert "total_requests" in report
        assert "total_tokens" in report
        assert "budget_limit_usd" in report
        assert "budget_remaining_usd" in report
        assert "by_model" in report


# Structured Logger Tests
class TestStructuredLogger:
    """Tests for structured logging."""

    def test_log_request(self, temp_log_file: Path, sample_request: CompletionRequest):
        """Should log requests in JSON format."""
        logger = StructuredLogger(log_file=temp_log_file)
        request_id = logger.log_request(sample_request)
        logger.close()

        assert request_id is not None

        # Read the log file
        with open(temp_log_file) as f:
            line = f.readline()
            entry = json.loads(line)

        assert entry["type"] == "request"
        assert entry["request_id"] == request_id
        assert entry["model"] == "gpt-4o"

    def test_log_response(
        self,
        temp_log_file: Path,
        sample_response: CompletionResponse,
    ):
        """Should log responses in JSON format."""
        logger = StructuredLogger(log_file=temp_log_file)
        logger.log_response(sample_response, "test-id", 100.0)
        logger.close()

        with open(temp_log_file) as f:
            line = f.readline()
            entry = json.loads(line)

        assert entry["type"] == "response"
        assert entry["request_id"] == "test-id"
        assert entry["latency_ms"] == 100.0
        assert entry["usage"]["total_tokens"] == 30

    def test_log_error(self, temp_log_file: Path):
        """Should log errors."""
        logger = StructuredLogger(log_file=temp_log_file)
        logger.log_error("test-id", ValueError("test error"), 50.0)
        logger.close()

        with open(temp_log_file) as f:
            line = f.readline()
            entry = json.loads(line)

        assert entry["type"] == "error"
        assert entry["request_id"] == "test-id"
        assert "test error" in entry["error"]

    def test_content_truncation(self, temp_log_file: Path, sample_request: CompletionRequest):
        """Should truncate long content."""
        logger = StructuredLogger(
            log_file=temp_log_file,
            max_content_length=10,
        )
        logger.log_request(sample_request)
        logger.close()

        with open(temp_log_file) as f:
            line = f.readline()
            entry = json.loads(line)

        # Content should be truncated (10 chars + "... [truncated]" = 25 chars max)
        if entry["messages"]:
            content = entry["messages"][0].get("content", "")
            assert len(content) <= 30  # 10 + "... [truncated]"
            assert "truncated" in content


# Callback Manager Tests
class TestCallbackManager:
    """Tests for callback management."""

    async def test_on_request_callback(self, sample_request: CompletionRequest):
        """Should call request callbacks."""
        manager = CallbackManager()
        called = []

        @manager.on_request
        async def on_req(request: CompletionRequest, request_id: str):
            called.append(("request", request.model, request_id))

        await manager.emit_request(sample_request, "test-id")

        assert len(called) == 1
        assert called[0] == ("request", "gpt-4o", "test-id")

    async def test_on_response_callback(self, sample_response: CompletionResponse):
        """Should call response callbacks."""
        manager = CallbackManager()
        called = []

        @manager.on_response
        async def on_resp(
            response: CompletionResponse,
            request_id: str,
            latency_ms: float,
            cached: bool,
        ):
            called.append(("response", response.model, latency_ms, cached))

        await manager.emit_response(sample_response, "test-id", 100.0, False)

        assert len(called) == 1
        assert called[0] == ("response", "gpt-4o", 100.0, False)

    async def test_on_error_callback(self):
        """Should call error callbacks."""
        manager = CallbackManager()
        called = []

        @manager.on_error
        async def on_err(error: Exception, request_id: str, latency_ms: float):
            called.append(("error", str(error), request_id))

        await manager.emit_error(ValueError("test"), "test-id", 50.0)

        assert len(called) == 1
        assert called[0] == ("error", "test", "test-id")

    async def test_multiple_callbacks(self, sample_request: CompletionRequest):
        """Should support multiple callbacks."""
        manager = CallbackManager()
        call_order = []

        @manager.on_request
        async def callback1(request: CompletionRequest, request_id: str):
            call_order.append(1)

        @manager.on_request
        async def callback2(request: CompletionRequest, request_id: str):
            call_order.append(2)

        await manager.emit_request(sample_request, "test-id")

        assert call_order == [1, 2]

    async def test_sync_callback(self, sample_request: CompletionRequest):
        """Should support synchronous callbacks."""
        manager = CallbackManager()
        called = []

        @manager.on_request
        def sync_callback(request: CompletionRequest, request_id: str):
            called.append("sync")

        await manager.emit_request(sample_request, "test-id")
        assert called == ["sync"]


# Inspect Logger Tests
class TestInspectLogger:
    """Tests for Inspect framework logging."""

    def test_log_entry(
        self,
        temp_log_dir: Path,
        sample_request: CompletionRequest,
        sample_response: CompletionResponse,
    ):
        """Should log entries in Inspect format."""
        logger = InspectLogger(log_dir=temp_log_dir, eval_id="test-eval")
        logger.log(sample_request, sample_response, sample_id=0)
        logger.close()

        # Read the log file
        log_file = temp_log_dir / "test-eval.jsonl"
        with open(log_file) as f:
            line = f.readline()
            entry = json.loads(line)

        assert entry["eval_id"] == "test-eval"
        assert entry["sample_id"] == 0
        assert entry["model"] == "gpt-4o"
        assert "input" in entry
        assert "output" in entry
        assert "tokens" in entry

    def test_auto_sample_id(
        self,
        temp_log_dir: Path,
        sample_request: CompletionRequest,
        sample_response: CompletionResponse,
    ):
        """Should auto-increment sample IDs."""
        logger = InspectLogger(log_dir=temp_log_dir)
        logger.log(sample_request, sample_response)
        logger.log(sample_request, sample_response)
        logger.close()

        with open(logger.log_file) as f:
            lines = f.readlines()

        entry1 = json.loads(lines[0])
        entry2 = json.loads(lines[1])

        assert entry1["sample_id"] == 0
        assert entry2["sample_id"] == 1

    def test_metadata_included(
        self,
        temp_log_dir: Path,
        sample_request: CompletionRequest,
        sample_response: CompletionResponse,
    ):
        """Should include custom metadata."""
        logger = InspectLogger(log_dir=temp_log_dir)
        logger.log(
            sample_request,
            sample_response,
            metadata={"custom_field": "custom_value"},
        )
        logger.close()

        with open(logger.log_file) as f:
            entry = json.loads(f.readline())

        assert entry["metadata"]["custom_field"] == "custom_value"
