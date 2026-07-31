from __future__ import annotations

import asyncio
from typing import Any, cast
from uuid import UUID, uuid4

from l1_foundation.messaging import EventEnvelope, KafkaEventProducer, new_event
from l1_foundation.pipeline.contracts import ArtifactRef, RetryPolicy, StageContext, StageResult
from l1_foundation.pipeline.definitions.graph import ArtifactBinding, PipelineDefinition, PipelineNode
from l1_foundation.pipeline.registry import StageRegistry
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l1_foundation.streaming import SyncRedisStreamStore
from l2_core.application.processing_queue import EmbeddingReindexWorkItem, ProcessingCancelWorkItem, ProcessingWorkItem
from l2_core.audio_processing.hooks import RecordingProcessingHooks
from l3_app.processing_worker.worker import ProcessingCancelHandler, ProcessingCommandHandler, processing_state_key


class _Redis:
    def __init__(self) -> None:
        self.states: dict[str, dict[str, Any]] = {}
        self.cancelled: set[str] = set()
        self.finished: list[tuple[str, str]] = []

    def get_state(self, key: str) -> dict[str, Any] | None:
        value = self.states.get(key)
        return dict(value) if value is not None else None

    def set_state(self, key: str, state: dict[str, Any]) -> None:
        self.states[key] = dict(state)

    def append(self, _stream: str, _event_type: str, _data: dict[str, Any]) -> str:
        return "1-0"

    def request_cancel(self, task_id: str) -> None:
        self.cancelled.add(task_id)

    def is_cancel_requested(self, task_id: str) -> bool:
        return task_id in self.cancelled

    def finish(self, state_key: str, stream_key: str) -> None:
        self.finished.append((state_key, stream_key))


class _Producer:
    def __init__(self) -> None:
        self.events: list[EventEnvelope] = []

    async def publish(self, _topic: str, _key: str, event: EventEnvelope) -> None:
        self.events.append(event)


class _Hooks:
    def __init__(self) -> None:
        self.statuses: list[str] = []
        self.stage_outputs: list[object] = []

    def stage_succeeded(self, _subject_id: UUID, _stage_name: str, output: object) -> None:
        self.stage_outputs.append(output)

    def run_state_changed(self, _subject_id: UUID, status: str, _error_message: str | None) -> None:
        self.statuses.append(status)


class _Stage:
    name = "test_stage"
    version = "1"
    retry_policy = RetryPolicy(max_attempts=1)

    def __init__(self, *, block: bool = False, restored: StageResult[dict[str, object]] | None = None) -> None:
        self.calls = 0
        self.block = block
        self.restored = restored
        self.started = asyncio.Event()
        self.was_cancelled = False

    async def try_restore(self, _context: StageContext, _input: object) -> StageResult[dict[str, object]] | None:
        return self.restored

    async def run(self, _context: StageContext, _input: object) -> StageResult[dict[str, object]]:
        self.calls += 1
        self.started.set()
        if self.block:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.was_cancelled = True
                raise
        return StageResult(output={})


class _Registry:
    def __init__(self, stage: _Stage) -> None:
        self._stage = stage

    def get(self, _name: str, _version: str) -> _Stage:
        return self._stage


def _command(processing_id: UUID, subject_id: UUID) -> EventEnvelope:
    item = ProcessingWorkItem(
        processing_id=processing_id,
        subject_type="recording",
        subject_id=subject_id,
        pipeline_name="test_pipeline",
        pipeline_version="1",
        initial_artifacts=[],
    )
    return new_event(
        "processing.requested",
        "test",
        processing_id=processing_id,
        correlation_id=processing_id,
        payload=item.model_dump(mode="json"),
    )


def _cancel(subject_id: UUID) -> EventEnvelope:
    item = ProcessingCancelWorkItem(subject_type="recording", subject_id=subject_id)
    return new_event(
        "processing.cancel.requested",
        "test",
        correlation_id=subject_id,
        payload=item.model_dump(mode="json"),
    )


def _embedding_reindex(processing_id: UUID, subject_id: UUID, chunks: ArtifactRef) -> EventEnvelope:
    item = EmbeddingReindexWorkItem(processing_id=processing_id, subject_id=subject_id, chunks=chunks)
    return new_event(
        "processing.embedding-index.requested",
        "test",
        processing_id=processing_id,
        correlation_id=processing_id,
        payload=item.model_dump(mode="json"),
    )


def _handler(redis: _Redis, producer: _Producer, hooks: _Hooks, stage: _Stage) -> ProcessingCommandHandler:
    definition = PipelineDefinition(
        name="test_pipeline",
        version="1",
        nodes=(PipelineNode("test_node", stage.name, stage.version, stage.retry_policy),),
    )
    return ProcessingCommandHandler(
        definition,
        cast(StageRegistry, _Registry(stage)),
        cast(ArtifactStore, object()),
        cast(SyncRedisStreamStore, redis),
        cast(KafkaEventProducer, producer),
        cast(RecordingProcessingHooks, hooks),
    )


def test_terminal_processing_redelivery_is_skipped() -> None:
    processing_id = uuid4()
    redis = _Redis()
    redis.states[processing_state_key(processing_id)] = {"status": "succeeded"}
    producer = _Producer()
    hooks = _Hooks()
    stage = _Stage()

    asyncio.run(_handler(redis, producer, hooks, stage).handle(_command(processing_id, uuid4())))

    assert stage.calls == 0
    assert producer.events == []
    assert hooks.statuses == []


def test_restored_stage_skips_execution_and_replays_projection() -> None:
    processing_id = uuid4()
    redis = _Redis()
    producer = _Producer()
    hooks = _Hooks()
    stage = _Stage(restored=StageResult(output={"restored": True}))

    asyncio.run(_handler(redis, producer, hooks, stage).handle(_command(processing_id, uuid4())))

    assert stage.calls == 0
    assert hooks.stage_outputs == [{"restored": True}]
    assert redis.states[processing_state_key(processing_id)]["stages"]["test_node"]["reused"] is True


def test_cancelled_processing_stops_before_the_first_stage() -> None:
    processing_id = uuid4()
    subject_id = uuid4()
    redis = _Redis()
    producer = _Producer()
    hooks = _Hooks()
    stage = _Stage()

    async def scenario() -> None:
        await ProcessingCancelHandler(cast(SyncRedisStreamStore, redis)).handle(_cancel(subject_id))
        await _handler(redis, producer, hooks, stage).handle(_command(processing_id, subject_id))

    asyncio.run(scenario())

    assert stage.calls == 0
    assert str(subject_id) in redis.cancelled
    assert redis.states[processing_state_key(processing_id)]["status"] == "cancelled"
    assert hooks.statuses == ["cancelled"]
    assert redis.finished


def test_cancellation_interrupts_the_active_async_stage() -> None:
    async def scenario() -> None:
        processing_id = uuid4()
        redis = _Redis()
        producer = _Producer()
        hooks = _Hooks()
        stage = _Stage(block=True)
        task = asyncio.create_task(_handler(redis, producer, hooks, stage).handle(_command(processing_id, uuid4())))
        await asyncio.wait_for(stage.started.wait(), timeout=1)
        redis.request_cancel(str(processing_id))
        await asyncio.wait_for(task, timeout=1)

        assert stage.was_cancelled
        assert redis.states[processing_state_key(processing_id)]["status"] == "cancelled"
        assert redis.states[processing_state_key(processing_id)]["stages"]["test_node"]["status"] == "cancelled"

    asyncio.run(scenario())


def test_embedding_reindex_runs_only_the_embedding_node() -> None:
    processing_id = uuid4()
    subject_id = uuid4()
    chunks = ArtifactRef(
        artifact_type="search.chunks",
        artifact_version="1",
        producer_stage="build_search_chunks",
        uri="artifacts/search-chunks.json",
    )
    redis = _Redis()
    redis.states[processing_state_key(processing_id)] = {
        "processing_id": str(processing_id),
        "subject_id": str(subject_id),
        "pipeline_name": "test_pipeline",
        "pipeline_version": "1",
        "status": "partial_failed",
        "stages": {
            "build_search_chunks": {"status": "succeeded", "artifacts": [chunks.model_dump(mode="json")]},
            "embedding_indexing": {"status": "failed", "error": "previous failure"},
        },
        "started_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    producer = _Producer()
    hooks = _Hooks()
    stage = _Stage()
    stage.name = "embedding_indexing"
    definition = PipelineDefinition(
        name="test_pipeline",
        version="1",
        nodes=(
            PipelineNode("build_search_chunks", "unused", "1", RetryPolicy(max_attempts=1)),
            PipelineNode(
                "embedding_indexing",
                stage.name,
                stage.version,
                stage.retry_policy,
                depends_on=("build_search_chunks",),
                required=False,
                input_artifacts=(ArtifactBinding("chunks", "search.chunks", "build_search_chunks"),),
            ),
        ),
    )
    handler = ProcessingCommandHandler(
        definition,
        cast(StageRegistry, _Registry(stage)),
        cast(ArtifactStore, object()),
        cast(SyncRedisStreamStore, redis),
        cast(KafkaEventProducer, producer),
        cast(RecordingProcessingHooks, hooks),
    )

    asyncio.run(handler.handle(_embedding_reindex(processing_id, subject_id, chunks)))

    state = redis.states[processing_state_key(processing_id)]
    assert stage.calls == 1
    assert state["stages"]["build_search_chunks"]["status"] == "succeeded"
    assert state["stages"]["embedding_indexing"]["status"] == "succeeded"
    assert state["status"] == "succeeded"
    assert hooks.statuses == ["succeeded"]
