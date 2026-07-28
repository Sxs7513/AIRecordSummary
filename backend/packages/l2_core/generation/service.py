from __future__ import annotations

from uuid import UUID

from sqlalchemy import Connection, Engine

from l2_core.generation.contracts import (
    CreateGenerationCommand,
    GenerationSnapshot,
)
from l2_core.generation.emitter import StreamEmitter
from l2_core.generation.hub import GenerationStreamHub
from l2_core.generation.store import GenerationEventStore


class GenerationService:
    """Single application-facing entry point for durable streaming generations."""

    def __init__(self, engine: Engine, hub: GenerationStreamHub | None = None) -> None:
        self._store = GenerationEventStore(engine)
        self._hub = hub

    @property
    def store(self) -> GenerationEventStore:
        return self._store

    def create(self, command: CreateGenerationCommand) -> GenerationSnapshot:
        return self._store.create(command)

    def create_in_transaction(self, connection: Connection, command: CreateGenerationCommand) -> GenerationSnapshot:
        """Create a run as part of the caller's larger business transaction."""
        return self._store.create_in_transaction(connection, command)

    def get(self, run_id: UUID) -> GenerationSnapshot:
        return self._store.get_snapshot(run_id)

    def cancel(self, run_id: UUID) -> GenerationSnapshot:
        return self._store.request_cancel(run_id)

    def emitter(self, run_id: UUID) -> StreamEmitter:
        return StreamEmitter(run_id, self._store, self._hub)
