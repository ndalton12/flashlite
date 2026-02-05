"""Observability module for flashlite."""

from .callbacks import (
    CallbackManager,
    OnErrorCallback,
    OnRequestCallback,
    OnResponseCallback,
    create_logging_callbacks,
)
from .inspect_compat import (
    FlashliteModelAPI,
    InspectLogEntry,
    InspectLogger,
    convert_flashlite_logs_to_inspect,
)
from .logging import RequestContext, RequestLogEntry, ResponseLogEntry, StructuredLogger
from .metrics import BudgetExceededError, CostMetrics, CostTracker

__all__ = [
    # Logging
    "StructuredLogger",
    "RequestLogEntry",
    "ResponseLogEntry",
    "RequestContext",
    # Metrics
    "CostTracker",
    "CostMetrics",
    "BudgetExceededError",
    # Callbacks
    "CallbackManager",
    "OnRequestCallback",
    "OnResponseCallback",
    "OnErrorCallback",
    "create_logging_callbacks",
    # Inspect
    "InspectLogger",
    "InspectLogEntry",
    "FlashliteModelAPI",
    "convert_flashlite_logs_to_inspect",
]
