from __future__ import annotations

import asyncio
from typing import Any, cast
from uuid import UUID, uuid4

from l2_core.rag.execution_middleware import raise_if_rag_cancelled
from l2_core.rag.service import RAG_TOKEN_BUDGET_EXCEEDED_MESSAGE, RagService
from l2_core.rag.token_budget import RagTokenBudgetExceeded


class _Sink:
    def __init__(self) -> None:
        self.failed: tuple[str, str, bool] | None = None
        self.cancel_requested = False
        self.cancelled = False

    def start(self) -> None:
        pass

    def cancel_if_requested(self) -> bool:
        self.cancelled = self.cancel_requested
        return self.cancelled

    def prepare_cancel_if_requested(self) -> object | None:
        self.cancelled = self.cancel_requested
        return object() if self.cancelled else None

    def phase(self, _name: str, _label: str, _progress: int | None = None) -> None:
        pass

    def text(self, _value: str) -> None:
        pass

    def fail(self, code: str, message: str, retryable: bool = False) -> None:
        self.failed = (code, message, retryable)

    def prepare_fail(self, code: str, message: str, retryable: bool = False) -> object:
        self.failed = (code, message, retryable)
        return object()


class _GenerationService:
    def __init__(self, sink: _Sink) -> None:
        self._sink = sink

    def event_sink(self, _run_id: UUID) -> _Sink:
        return self._sink

    def is_cancel_requested(self, _run_id: UUID) -> bool:
        return self._sink.cancel_requested


class _BudgetExceededGraph:
    async def run(self, **_kwargs: object) -> tuple[str, list[dict[str, object]], bool, str | None]:
        raise RagTokenBudgetExceeded("RAG token limit reached before answer: used=50000, limit=50000")


class _CooperativelyCancelledGraph:
    def __init__(self, sink: _Sink) -> None:
        self._sink = sink

    async def run(self, **_kwargs: object) -> tuple[str, list[dict[str, object]], bool, str | None]:
        self._sink.cancel_requested = True
        raise_if_rag_cancelled()
        raise AssertionError("cancel boundary should have stopped the graph")


class _ComputeCancelledGraph:
    def __init__(self, sink: _Sink) -> None:
        self._sink = sink

    async def run(self, **_kwargs: object) -> tuple[str, list[dict[str, object]], bool, str | None]:
        self._sink.cancel_requested = True
        raise RuntimeError("compute task cancelled")


class _Retriever:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True

    @staticmethod
    def hydrate_checkpoint_state(state: dict[str, Any]) -> dict[str, Any]:
        return state


def test_token_budget_exceeded_is_returned_as_a_user_facing_generation_error() -> None:
    sink = _Sink()
    retriever = _Retriever()
    service = object.__new__(RagService)
    service._graph = cast(Any, _BudgetExceededGraph())  # pyright: ignore[reportPrivateUsage]
    service._retriever = cast(Any, retriever)  # pyright: ignore[reportPrivateUsage]
    service._observability_client = cast(Any, object())  # pyright: ignore[reportPrivateUsage]
    service._checkpoint_store = cast(Any, object())  # pyright: ignore[reportPrivateUsage]

    asyncio.run(
        service.execute_answer_generation(
            cast(Any, _GenerationService(sink)),
            uuid4(),
            uuid4(),
            "复杂问题",
            10,
        )
    )

    assert sink.failed == ("rag_token_budget_exceeded", RAG_TOKEN_BUDGET_EXCEEDED_MESSAGE, False)
    assert retriever.released


def test_rag_boundary_cancellation_finishes_as_cancelled_without_failing() -> None:
    sink = _Sink()
    retriever = _Retriever()
    service = object.__new__(RagService)
    service._graph = cast(Any, _CooperativelyCancelledGraph(sink))  # pyright: ignore[reportPrivateUsage]
    service._retriever = cast(Any, retriever)  # pyright: ignore[reportPrivateUsage]
    service._observability_client = cast(Any, object())  # pyright: ignore[reportPrivateUsage]
    service._checkpoint_store = cast(Any, object())  # pyright: ignore[reportPrivateUsage]

    asyncio.run(service.execute_answer_generation(cast(Any, _GenerationService(sink)), uuid4(), uuid4(), "问题", 10))

    assert sink.cancelled
    assert sink.failed is None
    assert retriever.released


def test_compute_cancel_error_is_projected_as_generation_cancelled() -> None:
    sink = _Sink()
    retriever = _Retriever()
    service = object.__new__(RagService)
    service._graph = cast(Any, _ComputeCancelledGraph(sink))  # pyright: ignore[reportPrivateUsage]
    service._retriever = cast(Any, retriever)  # pyright: ignore[reportPrivateUsage]
    service._observability_client = cast(Any, object())  # pyright: ignore[reportPrivateUsage]
    service._checkpoint_store = cast(Any, object())  # pyright: ignore[reportPrivateUsage]

    asyncio.run(service.execute_answer_generation(cast(Any, _GenerationService(sink)), uuid4(), uuid4(), "问题", 10))

    assert sink.cancelled
    assert sink.failed is None
    assert retriever.released
