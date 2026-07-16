from __future__ import annotations

from time import monotonic
from uuid import UUID

from generation.contracts import ContentBlock, GenerationEvent, GenerationPhase
from generation.hub import GenerationStreamHub
from generation.store import GenerationEventStore


class StreamEmitter:
    """Convert execution callbacks into durable, ordered events and live notifications."""

    def __init__(self, run_id: UUID, store: GenerationEventStore, hub: GenerationStreamHub | None) -> None:
        self._run_id = run_id
        self._store = store
        self._hub = hub
        self._pending_text = ""
        self._last_flush_at = monotonic()

    def start(self) -> None:
        self._publish(self._store.start(self._run_id))

    def phase(self, name: str, label: str, progress_percent: int | None = None) -> None:
        self.flush()
        self._publish(self._store.set_phase(self._run_id, GenerationPhase(name=name, label=label), progress_percent))

    def text(self, value: str) -> None:
        if not value:
            return
        self._pending_text += value
        if len(self._pending_text) >= 500 or monotonic() - self._last_flush_at >= 0.5:
            self.flush()

    def flush(self) -> None:
        if not self._pending_text:
            return
        event = self._store.append_blocks(self._run_id, [ContentBlock(value=self._pending_text)])
        self._pending_text = ""
        self._last_flush_at = monotonic()
        if event is not None:
            self._publish(event)

    def succeed(self, output: dict[str, object], sources: list[dict[str, object]] | None = None) -> None:
        self.flush()
        self._publish(self._store.succeed(self._run_id, output, sources or []))

    def fail(self, code: str, message: str, retryable: bool = False) -> None:
        self.flush()
        self._publish(self._store.fail(self._run_id, code, message, retryable))

    def cancel_if_requested(self) -> bool:
        event = self._store.cancel_if_requested(self._run_id)
        if event is None:
            return False
        self._publish(event)
        return True

    def _publish(self, event: GenerationEvent) -> None:
        if self._hub is not None:
            self._hub.publish(event)
