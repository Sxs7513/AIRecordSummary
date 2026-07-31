from __future__ import annotations

from uuid import UUID

from sqlalchemy import Connection, Engine

from l2_core.generation.contracts import (
    CreateGenerationCommand,
    GenerationNotFoundError,
    GenerationSnapshot,
)
from l2_core.generation.event_sink import GenerationEventSink
from l2_core.generation.redis_runtime import GenerationRedisRuntime
from l2_core.generation.store import GenerationEventStore


class GenerationService:
    """Single application-facing entry point for durable streaming generations."""

    def __init__(self, engine: Engine, redis_runtime: GenerationRedisRuntime) -> None:
        self._postgres_store = GenerationEventStore(engine)
        self._redis_runtime = redis_runtime

    @property
    def store(self) -> GenerationEventStore:
        return self._postgres_store

    def create(self, command: CreateGenerationCommand) -> GenerationSnapshot:
        return self._redis_runtime.create_generation(command)

    def create_in_transaction(self, connection: Connection, command: CreateGenerationCommand) -> GenerationSnapshot:
        """Allocate Redis runtime identity; the connection remains for caller API compatibility."""
        del connection
        return self._redis_runtime.create_generation(command)

    def command(self, run_id: UUID) -> CreateGenerationCommand:
        command = self._redis_runtime.get_command(run_id)
        return command if command is not None else self._postgres_store.get_command(run_id)

    def ensure(self, run_id: UUID, command: CreateGenerationCommand) -> GenerationSnapshot:
        active = self._redis_runtime.get_snapshot(run_id)
        return active[0] if active is not None else self._redis_runtime.create_generation(command, run_id)

    def get(self, run_id: UUID) -> GenerationSnapshot:
        active = self._redis_runtime.get_snapshot(run_id)
        return active[0] if active is not None else self._postgres_store.get_snapshot(run_id)

    def cursor(self, run_id: UUID) -> str:
        active = self._redis_runtime.get_snapshot(run_id)
        return active[1] if active is not None else "0-0"

    def cancel(self, run_id: UUID) -> GenerationSnapshot:
        snapshot = self.get(run_id)
        if snapshot.status.is_terminal:
            return snapshot
        self._redis_runtime.request_cancel(run_id)
        return self.event_sink(run_id).cancel()

    def is_cancel_requested(self, run_id: UUID) -> bool:
        return self._redis_runtime.is_cancel_requested(run_id)

    def event_sink(self, run_id: UUID) -> GenerationEventSink:
        return GenerationEventSink(run_id, self._redis_runtime)

    def delete_runtime_data(self, run_id: UUID) -> None:
        try:
            command = self.command(run_id)
        except GenerationNotFoundError:
            command = None
        self._redis_runtime.delete_generation(run_id, command.idempotency_key if command is not None else None)
