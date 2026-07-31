from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from l1_foundation.messaging import EventEnvelope
from l1_foundation.observability.contracts import ModelInvocationRecord, RagExecutionSpanRecord

logger = logging.getLogger("observability")


class ObservabilityProjectionRepository(Protocol):
    def upsert_span(self, record: RagExecutionSpanRecord) -> None: ...

    def upsert_model_invocation(self, record: ModelInvocationRecord) -> None: ...


class ObservabilityEventProjector:
    """Validate Kafka payloads and idempotently update query projections."""

    def __init__(self, repository: ObservabilityProjectionRepository) -> None:
        self._repository = repository

    async def handle(self, event: EventEnvelope) -> None:
        if event.event_type == "rag.span.recorded":
            record = RagExecutionSpanRecord.model_validate(event.payload)
            await asyncio.to_thread(self._repository.upsert_span, record)
            return
        if event.event_type == "model.invocation.recorded":
            record = ModelInvocationRecord.model_validate(event.payload)
            await asyncio.to_thread(self._repository.upsert_model_invocation, record)
            return
        logger.warning("ignoring unsupported observability event type=%s event_id=%s", event.event_type, event.event_id)
