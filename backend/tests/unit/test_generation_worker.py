from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

from l1_foundation.messaging import EventEnvelope, KafkaEventProducer, new_event
from l2_core.generation.contracts import (
    CreateGenerationCommand,
    GenerationKind,
    GenerationPriority,
    GenerationSnapshot,
    GenerationStatus,
)
from l2_core.generation.service import GenerationService
from l2_core.rag.queue import GenerationCancelWorkItem
from l3_app.generation_worker.worker import GenerationCancelHandler


class _GenerationService:
    def __init__(self, generation_id: UUID) -> None:
        self.cancelled: list[UUID] = []
        self.snapshot = _snapshot(generation_id, GenerationStatus.RUNNING)
        self.generation_command = CreateGenerationCommand(
            kind=GenerationKind.TEXT,
            priority=GenerationPriority.INTERACTIVE,
            idempotency_key="test",
        )
        self.terminal_store = _TerminalStore()

    @property
    def store(self) -> _TerminalStore:
        return self.terminal_store

    def get(self, _run_id: UUID) -> object:
        return self.snapshot

    def cancel(self, run_id: UUID) -> object:
        self.cancelled.append(run_id)
        self.snapshot = self.snapshot.model_copy(update={"status": GenerationStatus.CANCELLED})
        return self.snapshot

    def prepare_cancel(self, run_id: UUID) -> object:
        return self.cancel(run_id)

    def command(self, _run_id: UUID) -> CreateGenerationCommand:
        return self.generation_command


class _TerminalStore:
    def __init__(self) -> None:
        self.projected: list[tuple[GenerationSnapshot, CreateGenerationCommand]] = []

    def project_terminal(self, snapshot: GenerationSnapshot, command: CreateGenerationCommand) -> None:
        self.projected.append((snapshot, command))


class _Producer:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, EventEnvelope]] = []

    async def publish(self, topic: str, key: str, event: EventEnvelope) -> None:
        self.messages.append((topic, key, event))


class _Conversations:
    def __init__(self) -> None:
        self.synced: list[UUID] = []

    def sync_generation(self, generation_run_id: UUID) -> None:
        self.synced.append(generation_run_id)

    def mark_streaming(self, generation_run_id: UUID) -> None:
        del generation_run_id
        return

    def sync_generation_in_transaction(self, connection: object, snapshot: GenerationSnapshot) -> None:
        del connection, snapshot
        return None

    def apply_generation_history_cache(self, completed_history: object) -> None:
        del completed_history
        return None


class _TerminalCommitter:
    def __init__(self) -> None:
        self.committed: list[GenerationSnapshot] = []

    async def commit(
        self,
        _source: EventEnvelope,
        snapshot: GenerationSnapshot,
        _command: CreateGenerationCommand,
    ) -> None:
        self.committed.append(snapshot)


def test_generation_cancel_handler_marks_generation_and_propagates_compute_scope() -> None:
    generation_id = uuid4()
    service = _GenerationService(generation_id)
    producer = _Producer()
    conversations = _Conversations()
    terminal_committer = _TerminalCommitter()
    handler = GenerationCancelHandler(
        cast(GenerationService, service),
        cast(KafkaEventProducer, producer),
        conversations,
        terminal_committer,  # type: ignore[arg-type]
    )
    event = new_event(
        "generation.cancel.requested",
        "test",
        generation_id=generation_id,
        payload=GenerationCancelWorkItem(generation_id=generation_id).model_dump(mode="json"),
    )

    asyncio.run(handler.handle(event))

    assert service.cancelled == [generation_id]
    assert conversations.synced == []
    assert [snapshot.status for snapshot in terminal_committer.committed] == [GenerationStatus.CANCELLED]
    assert [message[0] for message in producer.messages] == ["compute.cancel"]
    topic, key, propagated = producer.messages[-1]
    assert topic == "compute.cancel"
    assert key == str(generation_id)
    assert propagated.event_type == "compute.cancel.requested"
    assert propagated.payload["execution_scope"] == {"kind": "generation", "id": str(generation_id)}


def _snapshot(generation_id: UUID, status: GenerationStatus) -> GenerationSnapshot:
    now = datetime.now(UTC)
    return GenerationSnapshot(
        id=generation_id,
        kind=GenerationKind.TEXT,
        priority=GenerationPriority.INTERACTIVE,
        status=status,
        phase=None,
        progress_percent=None,
        blocks=[],
        output=None,
        last_sequence=0,
        cancel_requested=False,
        error_code=None,
        error_message=None,
        created_at=now,
        started_at=now,
        finished_at=None,
        updated_at=now,
    )
