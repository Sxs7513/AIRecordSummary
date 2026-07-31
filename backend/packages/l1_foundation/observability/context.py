from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from uuid import UUID, uuid4

from l1_foundation.observability.client import ObservabilityClient
from l1_foundation.observability.contracts import (
    ModelInvocationRecord,
    ObservabilityScope,
    RagExecutionSpanRecord,
    TerminalStatus,
    UsageSource,
)


@dataclass(frozen=True, slots=True)
class _RuntimeScope:
    value: ObservabilityScope
    client: ObservabilityClient


@dataclass(frozen=True, slots=True)
class SpanHandle:
    record: RagExecutionSpanRecord
    monotonic_started: float
    token: Token[SpanHandle | None]


@dataclass(frozen=True, slots=True)
class InvocationHandle:
    record: ModelInvocationRecord
    monotonic_started: float
    client: ObservabilityClient


_scope: ContextVar[_RuntimeScope | None] = ContextVar("observability_scope", default=None)
_active_span: ContextVar[SpanHandle | None] = ContextVar("observability_active_span", default=None)


@contextmanager
def observation_scope(client: ObservabilityClient, value: ObservabilityScope) -> Generator[None]:
    scope_token = _scope.set(_RuntimeScope(value, client))
    span_token = _active_span.set(None)
    try:
        yield
    finally:
        _active_span.reset(span_token)
        _scope.reset(scope_token)


def current_runtime_scope() -> tuple[ObservabilityScope, ObservabilityClient] | None:
    runtime = _scope.get()
    return None if runtime is None else (runtime.value, runtime.client)


def current_span() -> SpanHandle | None:
    return _active_span.get()


def start_invocation(
    provider: str,
    *,
    invocation_id: UUID | None = None,
    usage_kind: str = "llm",
    stream: bool = False,
) -> InvocationHandle | None:
    runtime = _scope.get()
    if runtime is None:
        return None
    span = _active_span.get()
    record = ModelInvocationRecord(
        id=invocation_id or uuid4(),
        workspace_id=runtime.value.workspace_id,
        generation_run_id=runtime.value.generation_run_id,
        span_id=span.record.id if span is not None else None,
        component=runtime.value.component,
        operation=span.record.operation if span is not None else usage_kind,
        operation_version=span.record.operation_version if span is not None else "1",
        attempt=span.record.attempt if span is not None else 0,
        usage_kind=usage_kind,
        provider=provider,
        stream=stream,
        status="running",
        started_at=datetime.now(UTC),
    )
    runtime.client.publish_model_invocation(record)
    return InvocationHandle(record=record, monotonic_started=perf_counter(), client=runtime.client)


def finish_invocation(
    handle: InvocationHandle | None,
    status: TerminalStatus,
    *,
    model: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    usage_source: UsageSource = "unavailable",
    finish_reason: str | None = None,
    provider_request_id: str | None = None,
    error_type: str | None = None,
) -> None:
    if handle is None:
        return
    handle.client.publish_model_invocation(
        handle.record.model_copy(
            update={
                "model": model,
                "status": status,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "usage_source": usage_source,
                "finish_reason": finish_reason,
                "provider_request_id": provider_request_id,
                "error_type": error_type,
                "finished_at": datetime.now(UTC),
                "elapsed_ms": round((perf_counter() - handle.monotonic_started) * 1_000, 2),
            }
        )
    )


def start_span(
    operation: str,
    *,
    operation_version: str = "1",
    attempt: int = 0,
    metadata: dict[str, object] | None = None,
) -> SpanHandle | None:
    runtime = _scope.get()
    if runtime is None:
        return None
    parent = _active_span.get()
    record = RagExecutionSpanRecord(
        id=uuid4(),
        workspace_id=runtime.value.workspace_id,
        generation_run_id=runtime.value.generation_run_id,
        parent_span_id=parent.record.id if parent is not None else None,
        component=runtime.value.component,
        operation=operation,
        operation_version=operation_version,
        attempt=attempt,
        status="running",
        started_at=datetime.now(UTC),
        metadata=metadata or {},
    )
    token = _active_span.set(None)
    handle = SpanHandle(record=record, monotonic_started=perf_counter(), token=token)
    _active_span.set(handle)
    runtime.client.publish_span(record)
    return handle


def finish_span(
    handle: SpanHandle | None,
    status: TerminalStatus,
    *,
    error_type: str | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    if handle is None:
        return
    runtime = _scope.get()
    elapsed = round((perf_counter() - handle.monotonic_started) * 1_000, 2)
    finished = handle.record.model_copy(
        update={
            "status": status,
            "finished_at": datetime.now(UTC),
            "elapsed_ms": elapsed,
            "error_type": error_type,
            "metadata": {**handle.record.metadata, **(metadata or {})},
        }
    )
    if runtime is not None:
        runtime.client.publish_span(finished)
    _active_span.reset(handle.token)
