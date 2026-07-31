from __future__ import annotations

from datetime import UTC, datetime
from time import monotonic
from uuid import UUID

from l2_core.generation.contracts import ContentBlock, GenerationPhase, GenerationSnapshot, GenerationStatus
from l2_core.generation.redis_runtime import GenerationRedisRuntime


class GenerationEventSink:
    """Write generation live state and resumable events to Redis."""

    def __init__(self, run_id: UUID, redis_runtime: GenerationRedisRuntime) -> None:
        self._run_id = run_id
        self._redis_runtime = redis_runtime
        self._pending_text = ""
        self._last_flush_at = monotonic()
        active = redis_runtime.get_snapshot(run_id)
        if active is None:
            raise LookupError(f"Generation runtime state not found: {run_id}")
        self._snapshot = active[0]
        self._blocks: list[ContentBlock] = list(self._snapshot.blocks)
        self._cursor = "0-0"

    def start(self) -> None:
        if self._is_fenced():
            return
        now = datetime.now(UTC)
        self._snapshot = self._snapshot.model_copy(update={"status": GenerationStatus.RUNNING, "started_at": now, "updated_at": now})
        self._publish("run.status", {"status": GenerationStatus.RUNNING.value})

    def phase(self, name: str, label: str, progress_percent: int | None = None) -> None:
        if self._is_fenced():
            return
        self.flush()
        phase = GenerationPhase(name=name, label=label)
        self._snapshot = self._snapshot.model_copy(update={"phase": phase, "progress_percent": progress_percent, "updated_at": datetime.now(UTC)})
        self._publish("phase", {**phase.model_dump(mode="json"), "progress_percent": progress_percent})

    def text(self, value: str) -> None:
        if not value or self._is_fenced():
            return
        self._pending_text += value
        if len(self._pending_text) >= 500 or monotonic() - self._last_flush_at >= 0.5:
            self.flush()

    def flush(self) -> None:
        if not self._pending_text or self._is_fenced():
            self._pending_text = ""
            return
        block = ContentBlock(value=self._pending_text)
        self._pending_text = ""
        self._last_flush_at = monotonic()
        self._blocks.append(block)
        self._snapshot = self._snapshot.model_copy(update={"blocks": list(self._blocks), "updated_at": datetime.now(UTC)})
        self._publish("content.delta", {"blocks": [block.model_dump(mode="json")]})

    def succeed(
        self,
        output: dict[str, object],
        sources: list[dict[str, object]] | None = None,
        *,
        final_text: str | None = None,
    ) -> None:
        if self._is_fenced():
            return
        if final_text is None:
            self.flush()
        else:
            self._pending_text = ""
            self._blocks = [ContentBlock(value=final_text)] if final_text else []
        now = datetime.now(UTC)
        final_output = {**output, "content_blocks": [block.model_dump(mode="json") for block in self._blocks], "sources": sources or []}
        self._snapshot = self._snapshot.model_copy(
            update={
                "status": GenerationStatus.SUCCEEDED,
                "blocks": list(self._blocks),
                "sources": sources or [],
                "output": final_output,
                "finished_at": now,
                "updated_at": now,
            }
        )
        self._publish("output.final", {"output": self._snapshot.output, "sources": self._snapshot.sources})
        self._redis_runtime.expire_terminal_generation(self._run_id)

    def fail(self, code: str, message: str, retryable: bool = False) -> None:
        if self._is_fenced():
            return
        self.flush()
        now = datetime.now(UTC)
        self._snapshot = self._snapshot.model_copy(
            update={
                "status": GenerationStatus.FAILED,
                "blocks": list(self._blocks),
                "output": {"content_blocks": [block.model_dump(mode="json") for block in self._blocks], "retryable": retryable},
                "error_code": code,
                "error_message": message[:2000],
                "finished_at": now,
                "updated_at": now,
            }
        )
        self._publish("run.error", {"code": code, "message": message, "retryable": retryable})
        self._redis_runtime.expire_terminal_generation(self._run_id, preserve_checkpoints=True)

    def cancel_if_requested(self) -> bool:
        active = self._redis_runtime.get_snapshot(self._run_id)
        if active is not None and active[0].status == GenerationStatus.CANCELLED:
            return True
        if not self._redis_runtime.is_cancel_requested(self._run_id):
            return False
        self.cancel()
        return True

    def cancel(self, reason: str = "user_requested") -> GenerationSnapshot:
        """Immediately project an irreversible cancelled terminal state to Redis."""
        active = self._redis_runtime.get_snapshot(self._run_id)
        if active is not None and active[0].status.is_terminal:
            self._snapshot = active[0]
            return self._snapshot
        self._pending_text = ""
        now = datetime.now(UTC)
        self._snapshot = self._snapshot.model_copy(
            update={
                "status": GenerationStatus.CANCELLED,
                "blocks": list(self._blocks),
                "output": {"content_blocks": [block.model_dump(mode="json") for block in self._blocks]},
                "finished_at": now,
                "updated_at": now,
            }
        )
        self._publish("run.cancelled", {"reason": reason}, allow_cancel_projection=True)
        self._redis_runtime.expire_terminal_generation(self._run_id, preserve_checkpoints=True)
        return self._snapshot

    def _publish(self, event_type: str, data: dict[str, object], *, allow_cancel_projection: bool = False) -> None:
        if not allow_cancel_projection and self._is_fenced():
            return
        self._cursor, sequence = self._redis_runtime.append_event(self._run_id, event_type, data)
        self._snapshot = self._snapshot.model_copy(update={"last_sequence": sequence, "cancel_requested": False})
        self._redis_runtime.save_snapshot(self._snapshot, self._cursor)

    def _is_fenced(self) -> bool:
        active = self._redis_runtime.get_snapshot(self._run_id)
        return active is None or active[0].status.is_terminal or self._redis_runtime.is_cancel_requested(self._run_id)
