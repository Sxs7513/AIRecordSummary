from __future__ import annotations

import asyncio
import logging
from typing import Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import Connection

from l1_foundation.messaging import EventEnvelope, KafkaEventProducer, OutboxRepository, Topics, new_event
from l1_foundation.worker import ComputeCancelRequest, ExecutionScope, execution_scope
from l2_core.audio_processing.contracts import RecordingId
from l2_core.audio_processing.stages.summary.regeneration import RecordingSummaryRegenerationService
from l2_core.generation.contracts import CreateGenerationCommand, GenerationSnapshot, GenerationStatus
from l2_core.generation.service import GenerationService
from l2_core.rag.adjudication.contracts import ClaimConfirmationDecision
from l2_core.rag.contracts import RagHistoryMessage
from l2_core.rag.queue import GenerationCancelWorkItem, RagGenerationWorkItem, SummaryGenerationWorkItem

logger = logging.getLogger("generation_worker")
ConversationHistoryProjection = tuple[UUID, list[tuple[RagHistoryMessage, RagHistoryMessage]]]


class ConversationGenerationProjection(Protocol):
    def mark_streaming(self, generation_run_id: UUID) -> None: ...

    def sync_generation(self, generation_run_id: UUID) -> None: ...

    def sync_generation_in_transaction(self, connection: Connection, snapshot: GenerationSnapshot) -> ConversationHistoryProjection | None: ...

    def apply_generation_history_cache(self, completed_history: ConversationHistoryProjection | None) -> None: ...


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
        force_correction: bool = False,
        resume_from_generation_id: UUID | None = None,
        adjudication_user_decision: ClaimConfirmationDecision | None = None,
    ) -> GenerationSnapshot: ...


class GenerationTerminalCommitter:
    """Atomically persist one terminal generation, its conversation projection and state outbox event."""

    def __init__(
        self,
        generation_service: GenerationService,
        conversations: ConversationGenerationProjection,
        outbox: OutboxRepository | None = None,
    ) -> None:
        self._generation_service = generation_service
        self._conversations = conversations
        self._outbox = outbox or OutboxRepository()

    async def commit(
        self,
        source: EventEnvelope,
        snapshot: GenerationSnapshot,
        command: CreateGenerationCommand,
    ) -> None:
        state_event = self._state_event(source, snapshot, command)
        completed_history = await asyncio.to_thread(
            self._commit_in_transaction,
            snapshot,
            command,
            state_event,
        )
        if completed_history is not None:
            self._conversations.apply_generation_history_cache(completed_history)

    def _commit_in_transaction(
        self,
        snapshot: GenerationSnapshot,
        command: CreateGenerationCommand,
        state_event: EventEnvelope,
    ) -> ConversationHistoryProjection | None:
        completed_history: ConversationHistoryProjection | None = None
        store = self._generation_service.store
        with store.engine.begin() as connection:
            inserted = store.project_terminal_in_transaction(connection, snapshot, command)
            if not inserted:
                return None
            completed_history = self._conversations.sync_generation_in_transaction(connection, snapshot)
            self._outbox.enqueue(
                connection,
                channel="generation-state",
                topic="redis.generation-terminal",
                partition_key=str(snapshot.id),
                aggregate_type="generation",
                aggregate_id=snapshot.id,
                event=state_event,
            )
        return completed_history

    @staticmethod
    def _state_event(source: EventEnvelope, snapshot: GenerationSnapshot, command: CreateGenerationCommand) -> EventEnvelope:
        terminal_type = {
            GenerationStatus.SUCCEEDED: "generation.completed",
            GenerationStatus.FAILED: "generation.failed",
            GenerationStatus.CANCELLED: "generation.cancelled",
        }[snapshot.status]
        event = new_event(
            "generation.state.changed",
            "generation-worker",
            correlation_id=source.correlation_id,
            causation_id=source.event_id,
            workspace_id=source.workspace_id,
            generation_id=snapshot.id,
            payload={
                "terminal_event_type": terminal_type,
                "snapshot": snapshot.model_dump(mode="json"),
                "command": command.model_dump(mode="json"),
                "preserve_checkpoints": snapshot.status != GenerationStatus.SUCCEEDED or bool((snapshot.output or {}).get("interaction")),
            },
        )
        stable_id = uuid5(
            NAMESPACE_URL,
            f"generation-state:{snapshot.id}:{snapshot.status.value}:{snapshot.updated_at.isoformat()}",
        )
        return event.model_copy(update={"event_id": stable_id})


class GenerationCommandHandler:
    """Execute durable generation commands outside the HTTP API process."""

    def __init__(
        self,
        rag_service: RagGenerationExecutor,
        generation_service: GenerationService,
        conversations: ConversationGenerationProjection,
        summary_service: RecordingSummaryRegenerationService,
        terminal_committer: GenerationTerminalCommitter | None = None,
    ) -> None:
        self._rag_service = rag_service
        self._generation_service = generation_service
        self._conversations = conversations
        self._summary_service = summary_service
        self._terminal_committer = terminal_committer or GenerationTerminalCommitter(generation_service, conversations)

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
            await self._terminal_committer.commit(event, snapshot, item.generation)
            return

        if item.conversation_message_id is not None:
            self._conversations.mark_streaming(item.run_id)
        snapshot = await self._execute_rag(item)
        await self._terminal_committer.commit(event, snapshot, item.generation)

    async def _execute_rag(self, item: RagGenerationWorkItem) -> GenerationSnapshot:
        with execution_scope(ExecutionScope(kind="generation", id=item.run_id)):
            return await self._rag_service.execute_answer_generation(
                generation_service=self._generation_service,
                run_id=item.run_id,
                workspace_id=item.workspace_id,
                query=item.query,
                limit=item.limit,
                scope_recording_ids=item.scope_recording_ids,
                history=item.history,
                force_correction=item.force_correction,
                resume_from_generation_id=item.resume_from_generation_id,
                adjudication_user_decision=item.adjudication_user_decision,
            )

    async def _handle_summary(self, event: EventEnvelope) -> None:
        item = SummaryGenerationWorkItem.model_validate(event.payload)
        snapshot = self._generation_service.ensure(item.run_id, item.generation)
        if not snapshot.status.is_terminal:
            snapshot = await self._execute_summary(item)
        await self._terminal_committer.commit(event, snapshot, item.generation)

    async def _execute_summary(self, item: SummaryGenerationWorkItem) -> GenerationSnapshot:
        with execution_scope(ExecutionScope(kind="generation", id=item.run_id)):
            return await self._summary_service.execute(item.run_id, RecordingId(item.recording_id))


class GenerationCancelHandler:
    """Project reliable Generation cancellation and propagate it to active Compute tasks."""

    def __init__(
        self,
        generation_service: GenerationService,
        producer: KafkaEventProducer,
        conversations: ConversationGenerationProjection | None = None,
        terminal_committer: GenerationTerminalCommitter | None = None,
    ) -> None:
        self._generation_service = generation_service
        self._kafka_producer = producer
        self._conversations = conversations
        self._terminal_committer = terminal_committer or (GenerationTerminalCommitter(generation_service, conversations) if conversations is not None else None)

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
        snapshot = self._generation_service.prepare_cancel(item.generation_id)
        if self._terminal_committer is None:
            raise RuntimeError("Generation terminal committer is required for cancellation")
        await self._terminal_committer.commit(event, snapshot, self._generation_service.command(snapshot.id))
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
