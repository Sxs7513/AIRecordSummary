from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID, uuid4

from generations_routes import encode_sse

from l1_foundation.messaging import EventEnvelope
from l2_core.audio_processing.stages.summary.stage import SafeTextStream
from l2_core.generation.contracts import (
    AggreMessageBlock,
    GenerationEvent,
    GenerationKind,
    GenerationPriority,
    GenerationSnapshot,
    GenerationStatus,
    TextBlock,
)
from l2_core.generation.event_sink import GenerationEventSink
from l2_core.generation.redis_runtime import GenerationRedisRuntime, redis_stream_sequence
from l2_core.rag.adjudication.contracts import (
    AdjudicationConfirmationBlock,
    AdjudicationConfirmationCandidate,
    AdjudicationConfirmationItem,
)
from l2_core.rag.queue import GenerationCancelWorkItem, GenerationCommandPublisher


class _Producer:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, EventEnvelope]] = []

    async def publish(self, topic: str, key: str, event: EventEnvelope) -> None:
        self.messages.append((topic, key, event))


def test_safe_text_stream_never_emits_a_leading_thinking_block() -> None:
    visible: list[str] = []
    stream = SafeTextStream(visible.append)

    stream.feed("<thi")
    stream.feed("nk>内部推理")
    stream.feed("内容</think>最终")
    stream.feed("总结")
    stream.finish()

    assert "".join(visible) == "最终总结"


def test_sse_event_uses_redis_stream_cursor_as_the_browser_reconnect_id() -> None:
    encoded = encode_sse(_event(uuid4(), redis_stream_sequence("1720000000000-9")), "1720000000000-9")

    assert encoded.startswith("id: 1720000000000-9\nevent: content.delta\ndata: ")
    assert encoded.endswith("\n\n")


def test_generation_cancel_is_published_as_a_reliable_kafka_command() -> None:
    run_id = uuid4()
    producer = _Producer()
    publisher = GenerationCommandPublisher(producer)  # type: ignore[arg-type]

    asyncio.run(publisher.cancel(GenerationCancelWorkItem(generation_id=run_id)))

    [(topic, key, event)] = producer.messages
    assert topic == "generation.cancel"
    assert key == str(run_id)
    assert event.event_type == "generation.cancel.requested"
    assert event.payload == {"generation_id": str(run_id), "reason": "user_requested"}


def test_generation_runtime_cleanup_removes_state_events_cancel_flags_and_checkpoints() -> None:
    run_id = uuid4()
    redis_store = _GenerationRedisStore(run_id)
    runtime = GenerationRedisRuntime(redis_store)  # type: ignore[arg-type]

    runtime.delete_generation(run_id, "conversation-message:test")

    assert redis_store.deleted_keys == (
        f"generation:{run_id}",
        f"generation:{run_id}:events",
        f"task:{run_id}:cancel",
        f"task:execution:generation:{run_id}:cancel",
        "generation:idempotency:conversation-message:test",
    )
    assert redis_store.deleted_pattern == f"generation:{run_id}:rag-checkpoint:*"


def test_cancel_projects_terminal_event_immediately_and_fences_a_stale_sink() -> None:
    run_id = uuid4()
    runtime = _SinkRuntime(_snapshot(run_id))
    stale_sink = GenerationEventSink(run_id, runtime)  # type: ignore[arg-type]
    cancel_sink = GenerationEventSink(run_id, runtime)  # type: ignore[arg-type]

    cancelled = cancel_sink.cancel()
    stale_sink.phase("answer", "正在回答", 90)
    stale_sink.text("迟到正文")
    stale_sink.succeed({"message": None})

    assert cancelled.status == GenerationStatus.CANCELLED
    assert runtime.snapshot.status == GenerationStatus.CANCELLED
    assert [event_type for event_type, _ in runtime.events] == ["run.cancelled"]
    assert runtime.expired


def test_final_output_replaces_streamed_blocks_with_canonical_text() -> None:
    run_id = uuid4()
    runtime = _SinkRuntime(_snapshot(run_id))
    sink = GenerationEventSink(run_id, runtime)  # type: ignore[arg-type]

    sink.text("原始引用[3]")
    sink.flush()
    sink.succeed({"message": None}, [{"index": 1}], final_text="最终引用[1]")

    assert [block.value for block in runtime.snapshot.blocks if isinstance(block, TextBlock)] == ["最终引用[1]"]
    event_type, data = runtime.events[-1]
    assert event_type == "output.final"
    output = data["output"]
    assert isinstance(output, dict)
    assert output["content_blocks"] == [{"type": "text", "value": "最终引用[1]"}]


def test_prepare_success_builds_terminal_snapshot_without_publishing_redis_terminal() -> None:
    run_id = uuid4()
    runtime = _SinkRuntime(_snapshot(run_id))
    sink = GenerationEventSink(run_id, runtime)  # type: ignore[arg-type]

    terminal = sink.prepare_succeed({"message": None}, final_text="最终回答")

    assert terminal is not None
    assert terminal.status == GenerationStatus.SUCCEEDED
    assert runtime.snapshot.status == GenerationStatus.RUNNING
    assert runtime.events == []
    assert not runtime.expired


def test_aggregate_message_streams_two_variants_and_persists_one_final_block() -> None:
    run_id = uuid4()
    runtime = _SinkRuntime(_snapshot(run_id))
    sink = GenerationEventSink(run_id, runtime)  # type: ignore[arg-type]

    sink.start_aggregate_message()
    sink.aggregate_text("original", "原始回答")
    sink.aggregate_text("corrected", "纠偏回答")
    sink.flush()
    sink.complete_aggregate_variant("original", "原始回答", [{"index": 1}])
    sink.complete_aggregate_variant("corrected", "纠偏回答", [{"index": 2}])
    sink.succeed({"message": None}, [{"index": 2}])

    assert sink.has_aggregate_message
    assert len(runtime.snapshot.blocks) == 1
    aggregate = runtime.snapshot.blocks[0]
    assert isinstance(aggregate, AggreMessageBlock)
    by_variant = {item.variant: item for item in aggregate.sub_message.sub_message_list}
    assert [block.value for block in by_variant["original"].blocks] == ["原始回答"]
    assert [block.value for block in by_variant["corrected"].blocks] == ["纠偏回答"]
    assert by_variant["original"].status == "completed"
    assert by_variant["corrected"].status == "completed"
    content_events = [data for event_type, data in runtime.events if event_type == "content.delta"]
    assert [event["operation"] for event in content_events] == ["replace", "append", "append", "replace", "replace"]
    assert runtime.events[-1][0] == "output.final"


def test_confirmation_block_is_streamed_and_succeeds_while_preserving_checkpoints() -> None:
    run_id = uuid4()
    runtime = _SinkRuntime(_snapshot(run_id))
    sink = GenerationEventSink(run_id, runtime)  # type: ignore[arg-type]
    block = AdjudicationConfirmationBlock(
        request_id=uuid4(),
        source_generation_id=run_id,
        items=[
            AdjudicationConfirmationItem(
                id="p1",
                evidence_index=1,
                recording_id=uuid4(),
                chunk_id=uuid4(),
                start_ms=1_000,
                end_ms=2_000,
                original_expression="RF",
                candidates=[
                    AdjudicationConfirmationCandidate(
                        id="p1",
                        expression="I²C",
                        confidence=0.8,
                    )
                ],
            )
        ],
    )

    sink.block(block)
    sink.succeed({"interaction": {"type": block.type}}, preserve_checkpoints=True)

    assert runtime.snapshot.status == GenerationStatus.SUCCEEDED
    assert runtime.snapshot.blocks == [block]
    assert [event_type for event_type, _data in runtime.events] == ["content.delta", "output.final"]
    assert runtime.preserve_checkpoints


class _GenerationRedisStore:
    def __init__(self, _run_id: UUID) -> None:
        self.deleted_keys: tuple[str, ...] = ()
        self.deleted_pattern = ""

    def delete(self, *keys: str) -> int:
        self.deleted_keys = keys
        return len(keys)

    def delete_pattern(self, pattern: str) -> int:
        self.deleted_pattern = pattern
        return 2


class _SinkRuntime:
    def __init__(self, snapshot: GenerationSnapshot) -> None:
        self.snapshot = snapshot
        self.events: list[tuple[str, dict[str, object]]] = []
        self.expired = False
        self.preserve_checkpoints = False

    def get_snapshot(self, _run_id: UUID) -> tuple[GenerationSnapshot, str]:
        return self.snapshot, f"{len(self.events)}-0"

    def is_cancel_requested(self, _run_id: UUID) -> bool:
        return False

    def append_event(self, _run_id: UUID, event_type: str, data: dict[str, object]) -> tuple[str, int]:
        self.events.append((event_type, data))
        sequence = len(self.events)
        return f"{sequence}-0", sequence

    def save_snapshot(self, snapshot: GenerationSnapshot, _cursor: str) -> bool:
        if self.snapshot.status.is_terminal and self.snapshot.status != snapshot.status:
            return False
        self.snapshot = snapshot
        return True

    def expire_terminal_generation(self, _run_id: UUID, *, preserve_checkpoints: bool = False) -> None:
        self.expired = True
        self.preserve_checkpoints = preserve_checkpoints


def _snapshot(run_id: UUID) -> GenerationSnapshot:
    now = datetime.now(UTC)
    return GenerationSnapshot(
        id=run_id,
        kind=GenerationKind.TEXT,
        priority=GenerationPriority.INTERACTIVE,
        status=GenerationStatus.RUNNING,
        phase=None,
        progress_percent=None,
        blocks=[],
        output=None,
        last_sequence=0,
        cancel_requested=False,
        error_code=None,
        error_message=None,
        created_at=now,
        started_at=now,
        finished_at=None,
        updated_at=now,
    )


def _event(run_id: UUID, sequence: int) -> GenerationEvent:
    return GenerationEvent(
        run_id=run_id,
        seq=sequence,
        type="content.delta",
        at=datetime.now(UTC),
        data={"blocks": [{"type": "text", "value": "测试"}]},
    )
