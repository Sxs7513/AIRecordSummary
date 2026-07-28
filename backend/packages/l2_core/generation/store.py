from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy import Connection, Engine, RowMapping, text

from l2_core.generation.contracts import (
    ContentBlock,
    CreateGenerationCommand,
    GenerationAccessScope,
    GenerationEvent,
    GenerationKind,
    GenerationNotFoundError,
    GenerationPhase,
    GenerationPriority,
    GenerationSnapshot,
    GenerationStatus,
)


class GenerationEventStore:
    """PostgreSQL source of truth for generation snapshots and resumable events."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @property
    def engine(self) -> Engine:
        """Database engine used by this durable generation store."""
        return self._engine

    def create(self, command: CreateGenerationCommand) -> GenerationSnapshot:
        with self._engine.begin() as connection:
            return self.create_in_transaction(connection, command)

    def create_in_transaction(self, connection: Connection, command: CreateGenerationCommand) -> GenerationSnapshot:
        row = (
            connection.execute(
                text(
                    """
                    insert into generation_runs (
                        kind, priority, idempotency_key, parent_type, parent_id,
                        owner_user_id, subject_type, subject_id, input_payload
                    ) values (
                        :kind, :priority, :idempotency_key, :parent_type, :parent_id,
                        :owner_user_id, :subject_type, :subject_id, cast(:input_payload as jsonb)
                    )
                    on conflict (idempotency_key) do update set updated_at = generation_runs.updated_at
                    returning *
                    """
                ),
                {
                    "kind": command.kind.value,
                    "priority": command.priority.value,
                    "idempotency_key": command.idempotency_key,
                    "parent_type": command.parent_type,
                    "parent_id": command.parent_id,
                    "owner_user_id": command.access_scope.owner_user_id,
                    "subject_type": command.access_scope.subject_type,
                    "subject_id": command.access_scope.subject_id,
                    "input_payload": json.dumps(command.input),
                },
            )
            .mappings()
            .one()
        )
        return self._snapshot(row)

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

    def events_after(self, run_id: UUID, sequence: int) -> list[GenerationEvent]:
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    select generation_run_id, sequence, event_type, payload, created_at
                    from generation_events
                    where generation_run_id = :run_id and sequence > :sequence
                    order by sequence
                    """
                ),
                {"run_id": run_id, "sequence": max(0, sequence)},
            ).mappings()
            return [self._event(row) for row in rows]

    def start(self, run_id: UUID) -> GenerationEvent:
        return self._append(run_id, "run.status", {"status": GenerationStatus.RUNNING.value}, status=GenerationStatus.RUNNING)

    def set_phase(self, run_id: UUID, phase: GenerationPhase, progress_percent: int | None = None) -> GenerationEvent:
        return self._append(
            run_id,
            "phase",
            phase.model_dump(mode="json"),
            phase=phase,
            progress_percent=progress_percent,
        )

    def append_blocks(self, run_id: UUID, blocks: Sequence[ContentBlock]) -> GenerationEvent | None:
        if not blocks:
            return None
        serialized_blocks = [block.model_dump(mode="json") for block in blocks]
        with self._engine.begin() as connection:
            row = self._locked_run(connection, run_id)
            status = GenerationStatus(cast(str, row["status"]))
            if status in (GenerationStatus.CANCELLED, GenerationStatus.FAILED, GenerationStatus.SUCCEEDED):
                return None
            existing = cast(list[dict[str, Any]], row["output_blocks"])
            sequence = cast(int, row["last_sequence"]) + 1
            event = self._insert_event(connection, run_id, sequence, "content.delta", {"blocks": serialized_blocks})
            connection.execute(
                text(
                    """
                    update generation_runs
                    set output_blocks = cast(:blocks as jsonb), last_sequence = :sequence,
                        first_token_at = coalesce(first_token_at, now()), updated_at = now()
                    where id = :run_id
                    """
                ),
                {"run_id": run_id, "blocks": json.dumps([*existing, *serialized_blocks]), "sequence": sequence},
            )
            return event

    def succeed(self, run_id: UUID, output: dict[str, Any], sources: Sequence[dict[str, object]] = ()) -> GenerationEvent:
        final_output = {**output, "sources": list(sources)}
        return self._append(
            run_id,
            "output.final",
            {"output": final_output, "sources": list(sources)},
            status=GenerationStatus.SUCCEEDED,
            output=final_output,
        )

    def fail(self, run_id: UUID, code: str, message: str, retryable: bool = False) -> GenerationEvent:
        return self._append(
            run_id,
            "run.error",
            {"code": code, "message": message, "retryable": retryable},
            status=GenerationStatus.FAILED,
            error_code=code,
            error_message=message,
        )

    def request_cancel(self, run_id: UUID) -> GenerationSnapshot:
        with self._engine.begin() as connection:
            row = (
                connection.execute(
                    text(
                        """
                    update generation_runs set cancel_requested = true, updated_at = now()
                    where id = :run_id and status in ('queued', 'running')
                    returning *
                    """
                    ),
                    {"run_id": run_id},
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            return self.get_snapshot(run_id)
        return self._snapshot(row)

    def cancel_if_requested(self, run_id: UUID, reason: str = "user_requested") -> GenerationEvent | None:
        with self._engine.connect() as connection:
            requested = connection.execute(text("select cancel_requested from generation_runs where id = :run_id"), {"run_id": run_id}).scalar_one_or_none()
        if requested is None:
            raise GenerationNotFoundError(str(run_id))
        if not requested:
            return None
        return self._append(run_id, "run.cancelled", {"reason": reason}, status=GenerationStatus.CANCELLED)

    def _append(
        self,
        run_id: UUID,
        event_type: str,
        payload: dict[str, Any],
        *,
        status: GenerationStatus | None = None,
        phase: GenerationPhase | None = None,
        progress_percent: int | None = None,
        output: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> GenerationEvent:
        with self._engine.begin() as connection:
            row = self._locked_run(connection, run_id)
            sequence = cast(int, row["last_sequence"]) + 1
            event = self._insert_event(connection, run_id, sequence, event_type, payload)
            fields = ["last_sequence = :sequence", "updated_at = now()"]
            parameters: dict[str, Any] = {"run_id": run_id, "sequence": sequence}
            if status is not None:
                fields.append("status = :status")
                parameters["status"] = status.value
                if status == GenerationStatus.RUNNING:
                    fields.append("started_at = coalesce(started_at, now())")
                if status in (GenerationStatus.SUCCEEDED, GenerationStatus.FAILED, GenerationStatus.CANCELLED):
                    fields.append("finished_at = now()")
            if phase is not None:
                fields.append("phase = cast(:phase as jsonb)")
                parameters["phase"] = json.dumps(phase.model_dump(mode="json"))
            if progress_percent is not None:
                fields.append("progress_percent = :progress_percent")
                parameters["progress_percent"] = max(0, min(100, progress_percent))
            if output is not None:
                fields.append("output_payload = cast(:output as jsonb)")
                parameters["output"] = json.dumps(output)
            if error_code is not None:
                fields.append("error_code = :error_code")
                parameters["error_code"] = error_code
            if error_message is not None:
                fields.append("error_message = :error_message")
                parameters["error_message"] = error_message[:2000]
            connection.execute(text(f"update generation_runs set {', '.join(fields)} where id = :run_id"), parameters)
            return event

    @staticmethod
    def _locked_run(connection: Any, run_id: UUID) -> RowMapping:
        row = connection.execute(text("select * from generation_runs where id = :run_id for update"), {"run_id": run_id}).mappings().one_or_none()
        if row is None:
            raise GenerationNotFoundError(str(run_id))
        return row

    @staticmethod
    def _insert_event(connection: Any, run_id: UUID, sequence: int, event_type: str, payload: dict[str, Any]) -> GenerationEvent:
        row = (
            connection.execute(
                text(
                    """
                insert into generation_events (generation_run_id, sequence, event_type, payload)
                values (:run_id, :sequence, :event_type, cast(:payload as jsonb))
                returning generation_run_id, sequence, event_type, payload, created_at
                """
                ),
                {"run_id": run_id, "sequence": sequence, "event_type": event_type, "payload": json.dumps(payload)},
            )
            .mappings()
            .one()
        )
        return GenerationEventStore._event(row)

    @staticmethod
    def _event(row: RowMapping) -> GenerationEvent:
        return GenerationEvent(
            run_id=cast(UUID, row["generation_run_id"]),
            seq=cast(int, row["sequence"]),
            type=cast(str, row["event_type"]),
            at=cast(datetime, row["created_at"]),
            data=cast(dict[str, Any], row["payload"]),
        )

    @staticmethod
    def _snapshot(row: RowMapping) -> GenerationSnapshot:
        phase_data = cast(dict[str, Any] | None, row["phase"])
        output = cast(dict[str, Any] | None, row["output_payload"])
        raw_sources: object = output.get("sources") if output is not None else []
        sources: list[object] = cast(list[object], raw_sources) if isinstance(raw_sources, list) else []
        return GenerationSnapshot(
            id=cast(UUID, row["id"]),
            kind=GenerationKind(cast(str, row["kind"])),
            priority=GenerationPriority(cast(str, row["priority"])),
            status=GenerationStatus(cast(str, row["status"])),
            phase=GenerationPhase.model_validate(phase_data) if phase_data else None,
            progress_percent=cast(int | None, row["progress_percent"]),
            blocks=[ContentBlock.model_validate(item) for item in cast(list[dict[str, Any]], row["output_blocks"])],
            sources=[item for item in sources if isinstance(item, dict)],
            output=output,
            last_sequence=cast(int, row["last_sequence"]),
            cancel_requested=cast(bool, row["cancel_requested"]),
            error_code=cast(str | None, row["error_code"]),
            error_message=cast(str | None, row["error_message"]),
            created_at=cast(datetime, row["created_at"]),
            started_at=cast(datetime | None, row["started_at"]),
            finished_at=cast(datetime | None, row["finished_at"]),
            updated_at=cast(datetime, row["updated_at"]),
        )
