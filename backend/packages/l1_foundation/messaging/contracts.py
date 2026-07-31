from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

JsonObject = dict[str, Any]


class EventEnvelope(BaseModel):
    """Versioned envelope shared by every Kafka command and event."""

    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    event_type: str = Field(min_length=1, max_length=160)
    schema_version: int = Field(default=1, ge=1)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    producer: str = Field(min_length=1, max_length=120)
    correlation_id: UUID
    causation_id: UUID | None = None
    workspace_id: UUID | None = None
    processing_id: UUID | None = None
    generation_id: UUID | None = None
    task_id: UUID | None = None
    trace_id: UUID | None = None
    attempt: int = Field(default=0, ge=0)
    last_error: str | None = Field(default=None, max_length=2000)
    payload: JsonObject = Field(default_factory=dict)


def new_event(
    event_type: str,
    producer: str,
    *,
    correlation_id: UUID | None = None,
    causation_id: UUID | None = None,
    workspace_id: UUID | None = None,
    processing_id: UUID | None = None,
    generation_id: UUID | None = None,
    task_id: UUID | None = None,
    trace_id: UUID | None = None,
    payload: JsonObject | None = None,
) -> EventEnvelope:
    event_id = uuid4()
    return EventEnvelope(
        event_id=event_id,
        event_type=event_type,
        producer=producer,
        correlation_id=correlation_id or event_id,
        causation_id=causation_id,
        workspace_id=workspace_id,
        processing_id=processing_id,
        generation_id=generation_id,
        task_id=task_id,
        trace_id=trace_id,
        payload=payload or {},
    )
