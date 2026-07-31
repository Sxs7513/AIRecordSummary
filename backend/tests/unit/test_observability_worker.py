from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from l1_foundation.messaging import new_event
from l1_foundation.observability.contracts import ModelInvocationRecord, RagExecutionSpanRecord
from l3_app.observability_worker.worker import ObservabilityEventProjector


class _Repository:
    def __init__(self) -> None:
        self.spans: list[RagExecutionSpanRecord] = []
        self.invocations: list[ModelInvocationRecord] = []

    def upsert_span(self, record: RagExecutionSpanRecord) -> None:
        self.spans.append(record)

    def upsert_model_invocation(self, record: ModelInvocationRecord) -> None:
        self.invocations.append(record)


def test_projector_validates_and_routes_observability_events() -> None:
    repository = _Repository()
    projector = ObservabilityEventProjector(repository)
    workspace_id = uuid4()
    generation_id = uuid4()
    now = datetime.now(UTC)
    span = RagExecutionSpanRecord(
        id=uuid4(),
        workspace_id=workspace_id,
        generation_run_id=generation_id,
        operation="answer",
        status="running",
        started_at=now,
    )
    invocation = ModelInvocationRecord(
        id=uuid4(),
        workspace_id=workspace_id,
        generation_run_id=generation_id,
        operation="answer",
        provider="gemini",
        status="running",
        started_at=now,
    )

    async def scenario() -> None:
        await projector.handle(
            new_event(
                "rag.span.recorded",
                "test",
                correlation_id=generation_id,
                generation_id=generation_id,
                payload=span.model_dump(mode="json"),
            )
        )
        await projector.handle(
            new_event(
                "model.invocation.recorded",
                "test",
                correlation_id=generation_id,
                generation_id=generation_id,
                payload=invocation.model_dump(mode="json"),
            )
        )

    asyncio.run(scenario())

    assert repository.spans == [span]
    assert repository.invocations == [invocation]
