from __future__ import annotations

import asyncio
import logging
from typing import Protocol
from uuid import UUID

from l1_foundation.messaging import EventEnvelope, KafkaEventProducer, Topics, new_event
from l1_foundation.worker import ComputeCancelRequest, ExecutionScope, execution_scope
from l2_core.audio_processing.contracts import RecordingId
from l2_core.audio_processing.stages.summary.regeneration import RecordingSummaryRegenerationService
from l2_core.generation.contracts import CreateGenerationCommand, GenerationSnapshot, GenerationStatus
from l2_core.generation.service import GenerationService
from l2_core.generation.store import GenerationEventStore
from l2_core.rag.adjudication.contracts import ClaimConfirmationDecision
from l2_core.rag.contracts import RagHistoryMessage
from l2_core.rag.queue import GenerationCancelWorkItem, RagGenerationWorkItem, SummaryGenerationWorkItem

logger = logging.getLogger("generation_worker")


class ConversationGenerationProjection(Protocol):
    def mark_streaming(self, generation_run_id: UUID) -> None: ...

    def sync_generation(self, generation_run_id: UUID) -> None: ...


class RagGenerationExecutor(Protocol):
    async def execute_answer_generation(
        self,
        generation_service: GenerationService,
        run_id: UUID,
        workspace_id: UUID,
        query: str,
        limit: int,
        scope_recording_ids: list[UUID] | None = None,
        history: list[RagHistoryMessage] | None = None,
        resume_from_generation_id: UUID | None = None,
        adjudication_user_decision: ClaimConfirmationDecision | None = None,
    ) -> None: ...


class GenerationCommandHandler:
    """Execute durable generation commands outside the HTTP API process."""

    def __init__(
        self,
        rag_service: RagGenerationExecutor,
        generation_service: GenerationService,
        conversations: ConversationGenerationProjection,
        producer: KafkaEventProducer,
        summary_service: RecordingSummaryRegenerationService,
    ) -> None:
        self._rag_service = rag_service
        self._generation_service = generation_service
        self._conversations = conversations
        self._kafka_producer = producer
        self._summary_service = summary_service

    async def handle(self, event: EventEnvelope) -> None:
        if event.event_type == "generation.summary.requested":
            await self._handle_summary(event)
            return
        if event.event_type != "generation.rag.requested":
            logger.warning("ignoring unsupported generation event type=%s event_id=%s", event.event_type, event.event_id)
            return

        item = RagGenerationWorkItem.model_validate(event.payload)
        snapshot = self._generation_service.ensure(item.run_id, item.generation)
        if snapshot.status in {GenerationStatus.SUCCEEDED, GenerationStatus.FAILED, GenerationStatus.CANCELLED}:
            if item.conversation_message_id is not None:
                self._conversations.sync_generation(item.run_id)
            await self._publish_terminal(event, snapshot)
            return

        if item.conversation_message_id is not None:
            self._conversations.mark_streaming(item.run_id)
        try:
            await self._execute_rag(item)
        finally:
            if item.conversation_message_id is not None:
                self._conversations.sync_generation(item.run_id)
            snapshot = self._generation_service.get(item.run_id)
            if snapshot.status in {GenerationStatus.SUCCEEDED, GenerationStatus.FAILED, GenerationStatus.CANCELLED}:
                await self._publish_terminal(event, snapshot)

    async def _execute_rag(self, item: RagGenerationWorkItem) -> None:
        with execution_scope(ExecutionScope(kind="generation", id=item.run_id)):
            await self._rag_service.execute_answer_generation(
                self._generation_service,
                item.run_id,
                item.workspace_id,
                item.query,
                item.limit,
                item.scope_recording_ids,
                item.history,
                item.resume_from_generation_id,
                item.adjudication_user_decision,
            )

    async def _handle_summary(self, event: EventEnvelope) -> None:
        item = SummaryGenerationWorkItem.model_validate(event.payload)
        snapshot = self._generation_service.ensure(item.run_id, item.generation)
        if not snapshot.status.is_terminal:
            await self._execute_summary(item)
            snapshot = self._generation_service.get(item.run_id)
        await self._publish_terminal(event, snapshot)

    async def _execute_summary(self, item: SummaryGenerationWorkItem) -> None:
        with execution_scope(ExecutionScope(kind="generation", id=item.run_id)):
            await self._summary_service.execute(item.run_id, RecordingId(item.recording_id))

    async def _publish_terminal(self, command: EventEnvelope, snapshot: GenerationSnapshot) -> None:
        event_type = {
            GenerationStatus.SUCCEEDED: "generation.completed",
            GenerationStatus.FAILED: "generation.failed",
            GenerationStatus.CANCELLED: "generation.cancelled",
        }[snapshot.status]
        payload = {"snapshot": snapshot.model_dump(mode="json"), "command": self._generation_service.command(snapshot.id).model_dump(mode="json")}
        terminal = new_event(
            event_type,
            "generation-worker",
            correlation_id=command.correlation_id,
            causation_id=command.event_id,
            workspace_id=command.workspace_id,
            generation_id=snapshot.id,
            payload=payload,
        )
        await self._kafka_producer.publish(Topics.GENERATION_EVENTS, str(snapshot.id), terminal)
        await self._kafka_producer.publish(
            Topics.GENERATION_STATE,
            str(snapshot.id),
            terminal.model_copy(update={"event_type": "generation.state.changed"}),
        )


class GenerationCancelHandler:
    """Project reliable Generation cancellation and propagate it to active Compute tasks."""

    def __init__(
        self,
        generation_service: GenerationService,
        producer: KafkaEventProducer,
        conversations: ConversationGenerationProjection | None = None,
    ) -> None:
        self._generation_service = generation_service
        self._kafka_producer = producer
        self._conversations = conversations

    async def handle(self, event: EventEnvelope) -> None:
        if event.event_type != "generation.cancel.requested":
            return
        item = GenerationCancelWorkItem.model_validate(event.payload)
        snapshot = self._generation_service.get(item.generation_id)
        if snapshot.status.is_terminal:
            logger.info(
                "Generation RAG cancel ignored generation_id=%s status=%s event_id=%s correlation_id=%s",
                item.generation_id,
                snapshot.status.value,
                event.event_id,
                event.correlation_id,
            )
            return
        logger.info(
            "Generation RAG cancel received generation_id=%s status=%s event_id=%s correlation_id=%s",
            item.generation_id,
            snapshot.status.value,
            event.event_id,
            event.correlation_id,
        )
        snapshot = self._generation_service.cancel(item.generation_id)
        if self._conversations is not None:
            self._conversations.sync_generation(item.generation_id)
        await self._publish_terminal(event, snapshot)
        cancel = ComputeCancelRequest(
            execution_scope=ExecutionScope(kind="generation", id=item.generation_id),
            reason="generation_cancelled",
        )
        await self._kafka_producer.publish(
            Topics.COMPUTE_CANCEL,
            str(item.generation_id),
            new_event(
                "compute.cancel.requested",
                "generation-worker",
                correlation_id=item.generation_id,
                causation_id=event.event_id,
                generation_id=item.generation_id,
                payload=cancel.model_dump(mode="json"),
            ),
        )
        logger.info(
            "Generation RAG cancel propagated generation_id=%s execution_scope=generation:%s causation_id=%s",
            item.generation_id,
            item.generation_id,
            event.event_id,
        )

    async def _publish_terminal(self, command: EventEnvelope, snapshot: GenerationSnapshot) -> None:
        payload = {
            "snapshot": snapshot.model_dump(mode="json"),
            "command": self._generation_service.command(snapshot.id).model_dump(mode="json"),
        }
        terminal = new_event(
            "generation.cancelled",
            "generation-worker",
            correlation_id=command.correlation_id,
            causation_id=command.event_id,
            workspace_id=command.workspace_id,
            generation_id=snapshot.id,
            payload=payload,
        )
        await self._kafka_producer.publish(Topics.GENERATION_EVENTS, str(snapshot.id), terminal)
        await self._kafka_producer.publish(
            Topics.GENERATION_STATE,
            str(snapshot.id),
            terminal.model_copy(update={"event_type": "generation.state.changed"}),
        )


class GenerationResultProjector:
    """Idempotently persist terminal Generation events as PostgreSQL query projections."""

    def __init__(self, store: GenerationEventStore) -> None:
        self._postgres_store = store

    async def handle(self, event: EventEnvelope) -> None:
        if event.event_type not in {"generation.completed", "generation.failed", "generation.cancelled"}:
            return
        snapshot = GenerationSnapshot.model_validate(event.payload["snapshot"])
        command = CreateGenerationCommand.model_validate(event.payload["command"])
        await asyncio.to_thread(self._postgres_store.project_terminal, snapshot, command)
