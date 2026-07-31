"""Best-effort RAG execution and model-usage telemetry."""

from l1_foundation.observability.client import ObservabilityClient
from l1_foundation.observability.context import (
    InvocationHandle,
    SpanHandle,
    finish_invocation,
    finish_span,
    observation_scope,
    start_invocation,
    start_span,
)
from l1_foundation.observability.contracts import (
    ModelInvocationRecord,
    ObservabilityScope,
    RagExecutionSpanRecord,
)
from l1_foundation.observability.instrumented_model_client import InstrumentedModelClient

__all__ = [
    "InstrumentedModelClient",
    "InvocationHandle",
    "ModelInvocationRecord",
    "ObservabilityClient",
    "ObservabilityScope",
    "RagExecutionSpanRecord",
    "SpanHandle",
    "finish_invocation",
    "finish_span",
    "observation_scope",
    "start_invocation",
    "start_span",
]
