from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from l1_foundation.messaging import KafkaEventProducer, Topics, new_event
from l2_core.generation.contracts import CreateGenerationCommand
from l2_core.rag.contracts import RagHistoryMessage


def _uuid_list() -> list[UUID]:
    return []


def _history_list() -> list[RagHistoryMessage]:
    return []


class RagGenerationWorkItem(BaseModel):
    """Complete, versioned input required by a generation worker."""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    workspace_id: UUID
    query: str = Field(min_length=1, max_length=2_000)
    limit: int = Field(ge=1, le=20)
    scope_recording_ids: list[UUID] = Field(default_factory=_uuid_list)
    history: list[RagHistoryMessage] = Field(default_factory=_history_list)
    conversation_message_id: UUID | None = None
    resume_from_generation_id: UUID | None = None
    generation: CreateGenerationCommand


class SummaryGenerationWorkItem(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_id: UUID
    recording_id: UUID
    generation: CreateGenerationCommand


class GenerationCancelWorkItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    generation_id: UUID
    reason: str = Field(default="user_requested", min_length=1, max_length=100)


class GenerationCommandPublisher:
    """Kafka-first submission boundary; success means the broker acknowledged the command."""

    def __init__(self, producer: KafkaEventProducer) -> None:
        self._kafka_producer = producer

    async def submit_rag(self, item: RagGenerationWorkItem) -> None:
        event = new_event(
            "generation.rag.requested",
            "production-api",
            correlation_id=item.run_id,
            workspace_id=item.workspace_id,
            generation_id=item.run_id,
            payload=item.model_dump(mode="json"),
        )
        await self._kafka_producer.publish(Topics.GENERATION_COMMANDS, str(item.run_id), event)

    async def submit_summary(self, item: SummaryGenerationWorkItem) -> None:
        event = new_event(
            "generation.summary.requested",
            "production-api",
            correlation_id=item.run_id,
            generation_id=item.run_id,
            payload=item.model_dump(mode="json"),
        )
        await self._kafka_producer.publish(Topics.GENERATION_COMMANDS, str(item.run_id), event)

    async def cancel(self, item: GenerationCancelWorkItem) -> None:
        await self._kafka_producer.publish(
            Topics.GENERATION_CANCEL,
            str(item.generation_id),
            new_event(
                "generation.cancel.requested",
                "production-api",
                correlation_id=item.generation_id,
                generation_id=item.generation_id,
                payload=item.model_dump(mode="json"),
            ),
        )
