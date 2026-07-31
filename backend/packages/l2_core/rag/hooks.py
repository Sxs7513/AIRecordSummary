from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Protocol


def _empty_metadata() -> Mapping[str, object]:
    return {}


@dataclass(frozen=True, slots=True)
class RagNodeCompleted:
    node: str
    attempt: int
    elapsed_ms: float
    metadata: Mapping[str, object] = field(default_factory=_empty_metadata)


@dataclass(frozen=True, slots=True)
class RagOperationCompleted:
    node: str
    operation: str
    output: object
    elapsed_ms: float
    status: str = "succeeded"
    details: Mapping[str, object] = field(default_factory=_empty_metadata)


class RagExecutionHook(Protocol):
    """Run-scoped observer for RAG nodes and their measurable operations."""

    def on_node_completed(self, event: RagNodeCompleted) -> None: ...

    def on_operation_completed(self, event: RagOperationCompleted) -> None: ...


class NoopRagExecutionHook:
    def on_node_completed(self, event: RagNodeCompleted) -> None:
        del event

    def on_operation_completed(self, event: RagOperationCompleted) -> None:
        del event


_NOOP_HOOK = NoopRagExecutionHook()
_CURRENT_HOOK: ContextVar[RagExecutionHook] = ContextVar("rag_execution_hook", default=_NOOP_HOOK)


def current_rag_execution_hook() -> RagExecutionHook:
    return _CURRENT_HOOK.get()


@contextmanager
def rag_execution_hook_scope(hook: RagExecutionHook | None) -> Generator[None]:
    token: Token[RagExecutionHook] | None = None
    if hook is not None:
        token = _CURRENT_HOOK.set(hook)
    try:
        yield
    finally:
        if token is not None:
            _CURRENT_HOOK.reset(token)
