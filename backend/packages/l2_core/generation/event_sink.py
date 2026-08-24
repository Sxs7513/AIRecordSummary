from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from time import monotonic
from typing import Literal
from uuid import UUID

from l2_core.generation.contracts import (
    AggregateSubMessage,
    AggreMessageBlock,
    ContentBlock,
    GenerationPhase,
    GenerationSnapshot,
    GenerationStatus,
    MessageGroup,
    SubMessage,
    TextBlock,
)
from l2_core.generation.redis_runtime import GenerationRedisRuntime


class GenerationEventSink:
    """Write generation live state and resumable events to Redis."""

    def __init__(self, run_id: UUID, redis_runtime: GenerationRedisRuntime) -> None:
        self._run_id = run_id
        self._redis_runtime = redis_runtime
        self._pending_text = ""
        self._pending_aggregate_text: dict[str, str] = {}
        self._last_flush_at = monotonic()
        active = redis_runtime.get_snapshot(run_id)
        if active is None:
            raise LookupError(f"Generation runtime state not found: {run_id}")
        self._snapshot = active[0]
        self._blocks: list[ContentBlock] = list(self._snapshot.blocks)
        self._cursor = "0-0"

    @property
    def has_aggregate_message(self) -> bool:
        return self._aggregate_block() is not None

    @property
    def snapshot(self) -> GenerationSnapshot:
        return self._snapshot

    def start_aggregate_message(self) -> None:
        if self._is_fenced():
            return
        self.flush()
        existing = self._aggregate_block()
        if existing is None:
            self._blocks = [item for item in self._blocks if not isinstance(item, TextBlock)]
            group = MessageGroup(
                id="answer-comparison",
                sub_message_ids=["original-answer", "corrected-answer"],
                primary_sub_message_id="corrected-answer",
            )
            block = AggreMessageBlock(
                id="answer-comparison",
                sub_message=AggregateSubMessage(
                    message_group=group,
                    sub_message_list=[
                        SubMessage(id="original-answer", variant="original", title="原始转写", status="streaming"),
                        SubMessage(id="corrected-answer", variant="corrected", title="纠偏后", status="streaming"),
                    ],
                ),
            )
        else:
            block = existing.model_copy(
                update={
                    "sub_message": existing.sub_message.model_copy(
                        update={
                            "sub_message_list": [
                                item.model_copy(update={"status": "streaming", "error": None}) for item in existing.sub_message.sub_message_list
                            ]
                        }
                    )
                }
            )
        self._store_aggregate_block(block)
        self._publish("content.delta", {"operation": "replace", "blocks": [block.model_dump(mode="json")]})

    def aggregate_text(self, variant: str, value: str) -> None:
        if not value or self._is_fenced():
            return
        sub_message_id = self._sub_message_id(variant)
        self._pending_aggregate_text[sub_message_id] = self._pending_aggregate_text.get(sub_message_id, "") + value
        if len(self._pending_aggregate_text[sub_message_id]) >= 500 or monotonic() - self._last_flush_at >= 0.5:
            self._flush_aggregate_text(sub_message_id)

    def complete_aggregate_variant(
        self,
        variant: str,
        text: str,
        sources: list[dict[str, object]],
    ) -> None:
        if self._is_fenced():
            return
        sub_message_id = self._sub_message_id(variant)
        self._flush_aggregate_text(sub_message_id)
        self._update_aggregate_sub_message(
            sub_message_id,
            lambda item: item.model_copy(
                update={
                    "status": "completed",
                    "blocks": [TextBlock(value=text)] if text else [],
                    "sources": sources,
                    "error": None,
                }
            ),
        )

    def fail_aggregate_variant(self, variant: str, error: str) -> None:
        if self._is_fenced():
            return
        sub_message_id = self._sub_message_id(variant)
        self._flush_aggregate_text(sub_message_id)
        self._update_aggregate_sub_message(
            sub_message_id,
            lambda item: item.model_copy(update={"status": "failed", "error": error[:2000]}),
        )

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

    def block(self, block: ContentBlock) -> None:
        if self._is_fenced():
            return
        self.flush()
        self._blocks.append(block)
        self._snapshot = self._snapshot.model_copy(update={"blocks": list(self._blocks), "updated_at": datetime.now(UTC)})
        self._publish("content.delta", {"blocks": [block.model_dump(mode="json")]})

    def flush(self) -> None:
        if self._is_fenced():
            self._pending_text = ""
            self._pending_aggregate_text.clear()
            return
        if self._pending_text:
            block = TextBlock(value=self._pending_text)
            self._pending_text = ""
            self._blocks.append(block)
            self._snapshot = self._snapshot.model_copy(update={"blocks": list(self._blocks), "updated_at": datetime.now(UTC)})
            self._publish("content.delta", {"blocks": [block.model_dump(mode="json")]})
        for sub_message_id in list(self._pending_aggregate_text):
            self._flush_aggregate_text(sub_message_id)
        self._last_flush_at = monotonic()

    def succeed(
        self,
        output: dict[str, object],
        sources: list[dict[str, object]] | None = None,
        *,
        final_text: str | None = None,
        preserve_checkpoints: bool = False,
    ) -> None:
        snapshot = self.prepare_succeed(output, sources, final_text=final_text)
        if snapshot is None:
            return
        self._publish("output.final", {"output": snapshot.output, "sources": snapshot.sources})
        self._redis_runtime.expire_terminal_generation(self._run_id, preserve_checkpoints=preserve_checkpoints)

    def prepare_succeed(
        self,
        output: dict[str, object],
        sources: list[dict[str, object]] | None = None,
        *,
        final_text: str | None = None,
    ) -> GenerationSnapshot | None:
        """Build a succeeded terminal snapshot without making it visible in Redis."""
        if self._is_fenced():
            return None
        if final_text is None:
            self.flush()
        else:
            self._pending_text = ""
            self._pending_aggregate_text.clear()
            self._blocks = [TextBlock(value=final_text)] if final_text else []
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
        return self._snapshot

    def fail(self, code: str, message: str, retryable: bool = False) -> None:
        snapshot = self.prepare_fail(code, message, retryable)
        if snapshot is None:
            return
        self._publish("run.error", {"code": code, "message": message, "retryable": retryable})
        self._redis_runtime.expire_terminal_generation(self._run_id, preserve_checkpoints=True)

    def prepare_fail(self, code: str, message: str, retryable: bool = False) -> GenerationSnapshot | None:
        """Build a failed terminal snapshot without making it visible in Redis."""
        if self._is_fenced():
            return None
        self.flush()
        self._mark_streaming_aggregate_variants("failed")
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
        return self._snapshot

    def prepare_cancel_if_requested(self) -> GenerationSnapshot | None:
        active = self._redis_runtime.get_snapshot(self._run_id)
        if active is not None and active[0].status == GenerationStatus.CANCELLED:
            self._snapshot = active[0]
            return self._snapshot
        if not self._redis_runtime.is_cancel_requested(self._run_id):
            return None
        return self.prepare_cancel()

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
        snapshot = self.prepare_cancel(reason)
        if snapshot.status != GenerationStatus.CANCELLED:
            return snapshot
        self._publish("run.cancelled", {"reason": reason}, allow_cancel_projection=True)
        self._redis_runtime.expire_terminal_generation(self._run_id, preserve_checkpoints=True)
        return self._snapshot

    def prepare_cancel(self, reason: str = "user_requested") -> GenerationSnapshot:
        """Build a cancelled terminal snapshot without making it visible in Redis."""
        del reason
        active = self._redis_runtime.get_snapshot(self._run_id)
        if active is not None and active[0].status.is_terminal:
            self._snapshot = active[0]
            return self._snapshot
        self._pending_text = ""
        self._pending_aggregate_text.clear()
        self._mark_streaming_aggregate_variants("cancelled")
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
        return self._snapshot

    def _publish(self, event_type: str, data: dict[str, object], *, allow_cancel_projection: bool = False) -> None:
        if not allow_cancel_projection and self._is_fenced():
            return
        self._cursor, sequence = self._redis_runtime.append_event(self._run_id, event_type, data)
        self._snapshot = self._snapshot.model_copy(update={"last_sequence": sequence, "cancel_requested": False})
        self._redis_runtime.save_snapshot(self._snapshot, self._cursor)

    def _flush_aggregate_text(self, sub_message_id: str) -> None:
        value = self._pending_aggregate_text.pop(sub_message_id, "")
        if not value:
            return
        block = self._require_aggregate_block()
        current = self._require_sub_message(block, sub_message_id)
        updated = current.model_copy(update={"status": "streaming", "blocks": [*current.blocks, TextBlock(value=value)]})
        self._store_aggregate_sub_message(block, updated)
        patch = AggreMessageBlock(
            id=block.id,
            sub_message=AggregateSubMessage(
                message_group=block.sub_message.message_group,
                sub_message_list=[updated.model_copy(update={"blocks": [TextBlock(value=value)]})],
            ),
        )
        self._publish("content.delta", {"operation": "append", "blocks": [patch.model_dump(mode="json")]})

    def _update_aggregate_sub_message(
        self,
        sub_message_id: str,
        update: Callable[[SubMessage], SubMessage],
    ) -> None:
        block = self._require_aggregate_block()
        current = self._require_sub_message(block, sub_message_id)
        updated = update(current)
        self._store_aggregate_sub_message(block, updated)
        patch = AggreMessageBlock(
            id=block.id,
            sub_message=AggregateSubMessage(
                message_group=block.sub_message.message_group,
                sub_message_list=[updated],
            ),
        )
        self._publish("content.delta", {"operation": "replace", "blocks": [patch.model_dump(mode="json")]})

    def _store_aggregate_sub_message(self, block: AggreMessageBlock, updated: SubMessage) -> None:
        sub_messages = [updated if item.id == updated.id else item for item in block.sub_message.sub_message_list]
        self._store_aggregate_block(block.model_copy(update={"sub_message": block.sub_message.model_copy(update={"sub_message_list": sub_messages})}))

    def _store_aggregate_block(self, block: AggreMessageBlock) -> None:
        replaced = False
        blocks: list[ContentBlock] = []
        for current in self._blocks:
            if isinstance(current, AggreMessageBlock) and current.id == block.id:
                blocks.append(block)
                replaced = True
            else:
                blocks.append(current)
        if not replaced:
            blocks.append(block)
        self._blocks = blocks
        self._snapshot = self._snapshot.model_copy(update={"blocks": list(self._blocks), "updated_at": datetime.now(UTC)})

    def _aggregate_block(self) -> AggreMessageBlock | None:
        return next((item for item in self._blocks if isinstance(item, AggreMessageBlock)), None)

    def _require_aggregate_block(self) -> AggreMessageBlock:
        block = self._aggregate_block()
        if block is None:
            raise RuntimeError("aggregate answer stream has not started")
        return block

    @staticmethod
    def _require_sub_message(block: AggreMessageBlock, sub_message_id: str) -> SubMessage:
        item = next((value for value in block.sub_message.sub_message_list if value.id == sub_message_id), None)
        if item is None:
            raise RuntimeError(f"aggregate sub-message is missing: {sub_message_id}")
        return item

    @staticmethod
    def _sub_message_id(variant: str) -> str:
        if variant == "original":
            return "original-answer"
        if variant == "corrected":
            return "corrected-answer"
        raise ValueError(f"unknown answer variant: {variant}")

    def _mark_streaming_aggregate_variants(self, status: Literal["failed", "cancelled"]) -> None:
        block = self._aggregate_block()
        if block is None:
            return
        updated = block.model_copy(
            update={
                "sub_message": block.sub_message.model_copy(
                    update={
                        "sub_message_list": [
                            item.model_copy(update={"status": status}) if item.status == "streaming" else item for item in block.sub_message.sub_message_list
                        ]
                    }
                )
            }
        )
        self._store_aggregate_block(updated)

    def _is_fenced(self) -> bool:
        active = self._redis_runtime.get_snapshot(self._run_id)
        return active is None or active[0].status.is_terminal or self._redis_runtime.is_cancel_requested(self._run_id)
