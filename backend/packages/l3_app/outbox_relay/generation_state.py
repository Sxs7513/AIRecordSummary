from __future__ import annotations

from typing import Protocol
from uuid import UUID

from l1_foundation.messaging import EventEnvelope
from l2_core.generation.contracts import CreateGenerationCommand, GenerationSnapshot, GenerationStatus


class GenerationTerminalRedisProjection(Protocol):
    def project_terminal(
        self,
        event_id: UUID,
        snapshot: GenerationSnapshot,
        command: CreateGenerationCommand,
        event_type: str,
        data: dict[str, object],
        *,
        preserve_checkpoints: bool = False,
    ) -> bool: ...


class GenerationStateProjector:
    """Idempotently project durable generation terminal events into Redis and the SSE stream."""

    def __init__(self, runtime: GenerationTerminalRedisProjection) -> None:
        self._runtime = runtime

    def handle(self, event: EventEnvelope) -> None:
        if event.event_type != "generation.state.changed":
            return
        snapshot = GenerationSnapshot.model_validate(event.payload["snapshot"])
        command = CreateGenerationCommand.model_validate(event.payload["command"])
        event_type, data = self._terminal_event(snapshot)
        self._runtime.project_terminal(
            event.event_id,
            snapshot,
            command,
            event_type,
            data,
            preserve_checkpoints=bool(event.payload.get("preserve_checkpoints", False)),
        )

    @staticmethod
    def _terminal_event(snapshot: GenerationSnapshot) -> tuple[str, dict[str, object]]:
        if snapshot.status == GenerationStatus.SUCCEEDED:
            return "output.final", {"output": snapshot.output, "sources": snapshot.sources}
        if snapshot.status == GenerationStatus.FAILED:
            retryable = bool((snapshot.output or {}).get("retryable", False))
            return "run.error", {
                "code": snapshot.error_code or "generation_failed",
                "message": snapshot.error_message or "Generation failed",
                "retryable": retryable,
            }
        if snapshot.status == GenerationStatus.CANCELLED:
            return "run.cancelled", {"reason": "user_requested"}
        raise ValueError(f"Generation state event is not terminal: {snapshot.status.value}")
