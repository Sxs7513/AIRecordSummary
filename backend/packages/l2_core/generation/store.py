from __future__ import annotations

import json
from datetime import datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy import Engine, RowMapping, text

from l2_core.generation.contracts import (
    ContentBlock,
    CreateGenerationCommand,
    GenerationAccessScope,
    GenerationKind,
    GenerationNotFoundError,
    GenerationPriority,
    GenerationSnapshot,
    GenerationStatus,
)


class GenerationEventStore:
    """PostgreSQL projection for generation identity, access and terminal results."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @property
    def engine(self) -> Engine:
        """Database engine used by this durable generation store."""
        return self._engine

    def get_snapshot(self, run_id: UUID) -> GenerationSnapshot:
        with self._engine.connect() as connection:
            row = connection.execute(text("select * from generation_runs where id = :run_id"), {"run_id": run_id}).mappings().one_or_none()
        if row is None:
            raise GenerationNotFoundError(str(run_id))
        return self._snapshot(row)

    def get_access_scope(self, run_id: UUID) -> GenerationAccessScope:
        with self._engine.connect() as connection:
            row = (
                connection.execute(text("select owner_user_id, subject_type, subject_id from generation_runs where id = :run_id"), {"run_id": run_id})
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise GenerationNotFoundError(str(run_id))
        return GenerationAccessScope(
            owner_user_id=cast(UUID | None, row["owner_user_id"]),
            subject_type=cast(str | None, row["subject_type"]),
            subject_id=cast(UUID | None, row["subject_id"]),
        )

    def get_command(self, run_id: UUID) -> CreateGenerationCommand:
        with self._engine.connect() as connection:
            row = connection.execute(text("select * from generation_runs where id = :run_id"), {"run_id": run_id}).mappings().one_or_none()
        if row is None:
            raise GenerationNotFoundError(str(run_id))
        return CreateGenerationCommand(
            kind=GenerationKind(cast(str, row["kind"])),
            priority=GenerationPriority(cast(str, row["priority"])),
            idempotency_key=cast(str, row["idempotency_key"]),
            parent_type=cast(str | None, row["parent_type"]),
            parent_id=cast(str | None, row["parent_id"]),
            access_scope=GenerationAccessScope(
                owner_user_id=cast(UUID | None, row["owner_user_id"]),
                subject_type=cast(str | None, row["subject_type"]),
                subject_id=cast(UUID | None, row["subject_id"]),
            ),
            input=cast(dict[str, Any], row["input_payload"]),
        )

    def project_terminal(self, snapshot: GenerationSnapshot, command: CreateGenerationCommand) -> None:
        """Idempotently materialize one terminal Kafka event for long-term queries."""
        if not snapshot.status.is_terminal:
            raise ValueError("Only terminal generation snapshots can be projected")
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    insert into generation_runs (
                        id, kind, priority, idempotency_key, parent_type, parent_id, owner_user_id,
                        subject_type, subject_id, status, input_payload, output_payload,
                        error_code, error_message, created_at, started_at, finished_at, updated_at
                    ) values (
                        :id, :kind, :priority, :idempotency_key, :parent_type, :parent_id, :owner_user_id,
                        :subject_type, :subject_id, :status, cast(:input_payload as jsonb), cast(:output_payload as jsonb),
                        :error_code, :error_message, :created_at, :started_at, :finished_at, :updated_at
                    )
                    on conflict (id) do update set
                        status = excluded.status, output_payload = excluded.output_payload,
                        error_code = excluded.error_code, error_message = excluded.error_message,
                        started_at = excluded.started_at, finished_at = excluded.finished_at, updated_at = excluded.updated_at
                    """
                ),
                {
                    "id": snapshot.id,
                    "kind": snapshot.kind.value,
                    "priority": snapshot.priority.value,
                    "idempotency_key": command.idempotency_key,
                    "parent_type": command.parent_type,
                    "parent_id": command.parent_id,
                    "owner_user_id": command.access_scope.owner_user_id,
                    "subject_type": command.access_scope.subject_type,
                    "subject_id": command.access_scope.subject_id,
                    "status": snapshot.status.value,
                    "input_payload": json.dumps(command.input),
                    "output_payload": json.dumps(snapshot.output),
                    "error_code": snapshot.error_code,
                    "error_message": snapshot.error_message,
                    "created_at": snapshot.created_at,
                    "started_at": snapshot.started_at,
                    "finished_at": snapshot.finished_at,
                    "updated_at": snapshot.updated_at,
                },
            )

    @staticmethod
    def _snapshot(row: RowMapping) -> GenerationSnapshot:
        output = cast(dict[str, Any] | None, row["output_payload"])
        raw_sources: object = output.get("sources") if output is not None else []
        raw_blocks: object = output.get("content_blocks") if output is not None else []
        sources: list[object] = cast(list[object], raw_sources) if isinstance(raw_sources, list) else []
        return GenerationSnapshot(
            id=cast(UUID, row["id"]),
            kind=GenerationKind(cast(str, row["kind"])),
            priority=GenerationPriority(cast(str, row["priority"])),
            status=GenerationStatus(cast(str, row["status"])),
            phase=None,
            progress_percent=None,
            blocks=[ContentBlock.model_validate(item) for item in cast(list[dict[str, Any]], raw_blocks)],
            sources=[item for item in sources if isinstance(item, dict)],
            output=output,
            last_sequence=0,
            cancel_requested=False,
            error_code=cast(str | None, row["error_code"]),
            error_message=cast(str | None, row["error_message"]),
            created_at=cast(datetime, row["created_at"]),
            started_at=cast(datetime | None, row["started_at"]),
            finished_at=cast(datetime | None, row["finished_at"]),
            updated_at=cast(datetime, row["updated_at"]),
        )
