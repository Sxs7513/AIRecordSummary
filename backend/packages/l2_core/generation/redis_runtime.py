from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

from l1_foundation.streaming import SyncRedisStreamStore
from l2_core.generation.contracts import CreateGenerationCommand, GenerationSnapshot, GenerationStatus, parse_content_block

GENERATION_TERMINAL_DELIVERY_TTL_SECONDS = 300


def generation_state_key(run_id: UUID) -> str:
    return f"generation:{run_id}"


def generation_stream_key(run_id: UUID) -> str:
    return f"generation:{run_id}:events"


def redis_stream_sequence(cursor: str) -> int:
    milliseconds, ordinal = cursor.split("-", maxsplit=1)
    return int(milliseconds) * 1_000 + int(ordinal)


class GenerationRedisRuntime:
    """Generation active-state and SSE-event projection stored in Redis."""

    def __init__(self, redis_store: SyncRedisStreamStore) -> None:
        self._redis_store = redis_store

    def append_event(self, run_id: UUID, event_type: str, data: dict[str, Any]) -> tuple[str, int]:
        cursor = self._redis_store.append(generation_stream_key(run_id), event_type, data)
        return cursor, redis_stream_sequence(cursor)

    def project_terminal(
        self,
        event_id: UUID,
        snapshot: GenerationSnapshot,
        command: CreateGenerationCommand,
        event_type: str,
        data: dict[str, Any],
        *,
        preserve_checkpoints: bool = False,
    ) -> bool:
        if not snapshot.status.is_terminal:
            raise ValueError("Only terminal generation snapshots can be projected")
        _cursor, projected = self._redis_store.project_terminal(
            generation_state_key(snapshot.id),
            generation_stream_key(snapshot.id),
            event_id=str(event_id),
            event_type=event_type,
            data=data,
            state=snapshot.model_dump(mode="json"),
            command=command.model_dump(mode="json"),
            ttl_seconds=GENERATION_TERMINAL_DELIVERY_TTL_SECONDS,
        )
        self.expire_terminal_generation(snapshot.id, preserve_checkpoints=preserve_checkpoints)
        return projected

    def create_generation(self, command: CreateGenerationCommand, run_id: UUID | None = None) -> GenerationSnapshot:
        existing = self._redis_store.get_state(f"generation:idempotency:{command.idempotency_key}")
        if existing is not None:
            active = self.get_snapshot(UUID(str(existing["run_id"])))
            if active is not None:
                return active[0]
        now = datetime.now(UTC)
        resume_blocks = command.input.get("resume_content_blocks", [])
        serialized_blocks = cast(list[object], resume_blocks) if isinstance(resume_blocks, list) else []
        snapshot = GenerationSnapshot(
            id=run_id or uuid4(),
            kind=command.kind,
            priority=command.priority,
            status=GenerationStatus.QUEUED,
            phase=None,
            progress_percent=None,
            blocks=[parse_content_block(item) for item in serialized_blocks],
            output=None,
            last_sequence=0,
            cancel_requested=False,
            error_code=None,
            error_message=None,
            created_at=now,
            started_at=None,
            finished_at=None,
            updated_at=now,
        )
        self._redis_store.set_state(f"generation:idempotency:{command.idempotency_key}", {"run_id": str(snapshot.id)})
        value = snapshot.model_dump(mode="json")
        value["cursor"] = "0-0"
        value["command"] = command.model_dump(mode="json")
        self._redis_store.set_state(generation_state_key(snapshot.id), value)
        return snapshot

    def save_snapshot(self, snapshot: GenerationSnapshot, cursor: str = "0-0") -> bool:
        current = self._redis_store.get_state(generation_state_key(snapshot.id)) or {}
        current_status = current.get("status")
        if current_status in {status.value for status in GenerationStatus if status.is_terminal} and current_status != snapshot.status.value:
            return False
        value = snapshot.model_dump(mode="json")
        value["cursor"] = cursor
        if "command" in current:
            value["command"] = current["command"]
        self._redis_store.set_state(generation_state_key(snapshot.id), value)
        return True

    def get_snapshot(self, run_id: UUID) -> tuple[GenerationSnapshot, str] | None:
        value = self._redis_store.get_state(generation_state_key(run_id))
        if value is None:
            return None
        cursor = str(value.pop("cursor", "0-0"))
        value.pop("command", None)
        return GenerationSnapshot.model_validate(value), cursor

    def get_command(self, run_id: UUID) -> CreateGenerationCommand | None:
        value = self._redis_store.get_state(generation_state_key(run_id))
        if value is None or not isinstance(value.get("command"), dict):
            return None
        return CreateGenerationCommand.model_validate(value["command"])

    def request_cancel(self, run_id: UUID) -> None:
        self._redis_store.request_cancel(str(run_id))

    def is_cancel_requested(self, run_id: UUID) -> bool:
        return self._redis_store.is_cancel_requested(str(run_id))

    def expire_terminal_generation(self, run_id: UUID, *, preserve_checkpoints: bool = False) -> None:
        """Keep terminal delivery data briefly, then remove Redis runtime state automatically."""
        command = self.get_command(run_id)
        self._redis_store.finish(
            generation_state_key(run_id),
            generation_stream_key(run_id),
            ttl_seconds=GENERATION_TERMINAL_DELIVERY_TTL_SECONDS,
        )
        keys = [
            f"task:{run_id}:cancel",
            f"task:execution:generation:{run_id}:cancel",
        ]
        if command is not None:
            keys.append(f"generation:idempotency:{command.idempotency_key}")
        self._redis_store.expire(*keys, ttl_seconds=GENERATION_TERMINAL_DELIVERY_TTL_SECONDS)
        if not preserve_checkpoints:
            self._redis_store.expire_pattern(
                f"generation:{run_id}:rag-checkpoint:*",
                ttl_seconds=GENERATION_TERMINAL_DELIVERY_TTL_SECONDS,
            )

    def delete_generation(self, run_id: UUID, idempotency_key: str | None = None) -> None:
        """Delete all Redis runtime data owned by one Generation."""
        if idempotency_key is None:
            command = self.get_command(run_id)
            idempotency_key = command.idempotency_key if command is not None else None
        keys = [
            generation_state_key(run_id),
            generation_stream_key(run_id),
            f"task:{run_id}:cancel",
            f"task:execution:generation:{run_id}:cancel",
        ]
        if idempotency_key is not None:
            keys.append(f"generation:idempotency:{idempotency_key}")
        self._redis_store.delete(*keys)
        self._redis_store.delete_pattern(f"generation:{run_id}:rag-checkpoint:*")
