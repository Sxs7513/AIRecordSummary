from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

from l1_foundation.messaging import EventEnvelope, OutboxRepository, new_event
from l2_core.generation.contracts import (
    CreateGenerationCommand,
    GenerationKind,
    GenerationPriority,
    GenerationSnapshot,
    GenerationStatus,
)
from l2_core.generation.service import GenerationService
from l3_app.generation_worker.worker import GenerationTerminalCommitter
from l3_app.outbox_relay.generation_state import GenerationStateProjector


def test_terminal_projection_and_state_outbox_share_one_transaction() -> None:
    connection = object()
    store = _TerminalStore(connection)
    conversations = _Conversations(connection)
    outbox = _Outbox(connection)
    committer = GenerationTerminalCommitter(
        cast(GenerationService, _GenerationService(store)),
        conversations,  # type: ignore[arg-type]
        cast(OutboxRepository, outbox),
    )
    snapshot = _snapshot(uuid4(), GenerationStatus.SUCCEEDED)
    command = _command()
    source = new_event("generation.rag.requested", "test", generation_id=snapshot.id)

    asyncio.run(committer.commit(source, snapshot, command))

    assert store.projected_connection is connection
    assert conversations.projected_connection is connection
    assert outbox.connection is connection
    assert outbox.event is not None
    assert outbox.topic == "redis.generation-terminal"
    assert outbox.event.event_type == "generation.state.changed"
    assert outbox.event.payload["snapshot"]["status"] == "succeeded"


def test_terminal_state_event_id_is_stable_for_command_redelivery() -> None:
    connection = object()
    outbox = _Outbox(connection)
    committer = GenerationTerminalCommitter(
        cast(GenerationService, _GenerationService(_TerminalStore(connection))),
        _Conversations(connection),  # type: ignore[arg-type]
        cast(OutboxRepository, outbox),
    )
    snapshot = _snapshot(uuid4(), GenerationStatus.SUCCEEDED)
    command = _command()
    source = new_event("generation.rag.requested", "test", generation_id=snapshot.id)

    asyncio.run(committer.commit(source, snapshot, command))
    first_event_id = outbox.event.event_id if outbox.event is not None else None
    asyncio.run(committer.commit(source, snapshot, command))

    assert outbox.event is not None
    assert outbox.event.event_id == first_event_id


def test_existing_terminal_row_does_not_create_another_outbox_event() -> None:
    connection = object()
    store = _TerminalStore(connection, inserted=False)
    outbox = _Outbox(connection)
    committer = GenerationTerminalCommitter(
        cast(GenerationService, _GenerationService(store)),
        _Conversations(connection),  # type: ignore[arg-type]
        cast(OutboxRepository, outbox),
    )
    snapshot = _snapshot(uuid4(), GenerationStatus.SUCCEEDED)

    asyncio.run(committer.commit(new_event("generation.rag.requested", "test"), snapshot, _command()))

    assert outbox.event is None


def test_generation_state_projector_maps_success_to_terminal_redis_event() -> None:
    runtime = _RedisProjection()
    projector = GenerationStateProjector(runtime)
    snapshot = _snapshot(uuid4(), GenerationStatus.SUCCEEDED)
    event = new_event(
        "generation.state.changed",
        "test",
        generation_id=snapshot.id,
        payload={"snapshot": snapshot.model_dump(mode="json"), "command": _command().model_dump(mode="json")},
    )

    projector.handle(event)

    assert runtime.event_id == event.event_id
    assert runtime.event_type == "output.final"
    assert runtime.data == {"output": snapshot.output, "sources": snapshot.sources}


class _Engine:
    def __init__(self, connection: object) -> None:
        self._connection = connection

    @contextmanager
    def begin(self) -> Generator[object]:
        yield self._connection


class _TerminalStore:
    def __init__(self, connection: object, *, inserted: bool = True) -> None:
        self.engine = _Engine(connection)
        self.projected_connection: object | None = None
        self._inserted = inserted

    def project_terminal_in_transaction(
        self,
        connection: object,
        snapshot: GenerationSnapshot,
        command: CreateGenerationCommand,
    ) -> bool:
        del snapshot, command
        self.projected_connection = connection
        return self._inserted


class _GenerationService:
    def __init__(self, store: _TerminalStore) -> None:
        self.store = store


class _Conversations:
    def __init__(self, expected_connection: object) -> None:
        self._expected_connection = expected_connection
        self.projected_connection: object | None = None

    def sync_generation_in_transaction(self, connection: object, _snapshot: GenerationSnapshot) -> None:
        assert connection is self._expected_connection
        self.projected_connection = connection

    def apply_generation_history_cache(self, _completed_history: object) -> None:
        pass


class _Outbox:
    def __init__(self, expected_connection: object) -> None:
        self._expected_connection = expected_connection
        self.connection: object | None = None
        self.event: EventEnvelope | None = None
        self.topic = "unset"

    def enqueue(self, connection: object, **kwargs: Any) -> None:
        assert connection is self._expected_connection
        self.connection = connection
        self.event = cast(EventEnvelope, kwargs["event"])
        self.topic = cast(str, kwargs["topic"])


class _RedisProjection:
    def __init__(self) -> None:
        self.event_id: UUID | None = None
        self.event_type: str | None = None
        self.data: dict[str, object] | None = None

    def project_terminal(
        self,
        event_id: UUID,
        snapshot: GenerationSnapshot,
        command: CreateGenerationCommand,
        event_type: str,
        data: dict[str, object],
        *,
        preserve_checkpoints: bool = False,
    ) -> bool:
        del snapshot, command, preserve_checkpoints
        self.event_id = event_id
        self.event_type = event_type
        self.data = data
        return True


def _command() -> CreateGenerationCommand:
    return CreateGenerationCommand(
        kind=GenerationKind.TEXT,
        priority=GenerationPriority.INTERACTIVE,
        idempotency_key="test-generation",
    )


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
        sources=[],
        output={"content_blocks": [], "sources": []},
        last_sequence=0,
        cancel_requested=False,
        error_code=None,
        error_message=None,
        created_at=now,
        started_at=now,
        finished_at=now,
        updated_at=now,
    )
