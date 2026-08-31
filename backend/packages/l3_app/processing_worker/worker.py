from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid5

from pydantic import BaseModel

from l1_foundation.messaging import EventEnvelope, KafkaEventProducer, Topics, new_event
from l1_foundation.pipeline.contracts import ArtifactPayload, ArtifactRef, PipelineRunId, PipelineSubjectId, StageContext, StageResult, StageRunId
from l1_foundation.pipeline.definitions.graph import PipelineDefinition, PipelineNode
from l1_foundation.pipeline.registry import StageRegistry
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore, build_input_fingerprint
from l1_foundation.streaming import SyncRedisStreamStore
from l1_foundation.worker import ExecutionScope, execution_scope
from l2_core.application.processing_queue import EmbeddingReindexWorkItem, ProcessingCancelWorkItem, ProcessingWorkItem
from l2_core.audio_processing.hooks import RecordingProcessingHooks

logger = logging.getLogger("audio_processing")
TERMINAL_PROCESSING_STATUSES = frozenset({"succeeded", "partial_failed", "failed", "cancelled"})
PROCESSING_CANCEL_POLL_SECONDS = 0.2


class ProcessingCancelledError(Exception):
    """Stop the remaining DAG after a recording requests cancellation."""


class ProcessingCancelHandler:
    """Project durable Processing cancellation requests into fast Redis flags."""

    def __init__(self, redis: SyncRedisStreamStore) -> None:
        self._redis_event_store = redis

    async def handle(self, event: EventEnvelope) -> None:
        if event.event_type != "processing.cancel.requested":
            return
        item = ProcessingCancelWorkItem.model_validate(event.payload)
        self._redis_event_store.request_cancel(str(item.subject_id))


def processing_state_key(processing_id: UUID) -> str:
    return f"processing:{processing_id}:state"


def processing_stream_key(processing_id: UUID) -> str:
    return f"processing:{processing_id}:events"


class _ProgressReporter:
    def __init__(self, redis: SyncRedisStreamStore, item: ProcessingWorkItem, node: str, state: dict[str, Any]) -> None:
        self._redis_event_store = redis
        self._item = item
        self._node = node
        self._state = state

    def report(self, percent: int, message: str) -> None:
        bounded = max(0, min(100, percent))
        stage = cast(dict[str, Any], self._state["stages"][self._node])
        stage.update(progress_percent=bounded, progress_message=message, progress_updated_at=datetime.now(UTC).isoformat())
        self._state["updated_at"] = datetime.now(UTC).isoformat()
        self._redis_event_store.set_state(processing_state_key(self._item.processing_id), self._state)
        self._redis_event_store.append(
            processing_stream_key(self._item.processing_id),
            "stage.progress",
            {"node": self._node, "percent": bounded, "message": message},
        )


class ProcessingCommandHandler:
    """Execute a versioned DAG whose durable command and lifecycle live in Kafka."""

    def __init__(
        self,
        definition: PipelineDefinition,
        registry: StageRegistry,
        artifacts: ArtifactStore,
        redis: SyncRedisStreamStore,
        producer: KafkaEventProducer,
        hooks: RecordingProcessingHooks,
    ) -> None:
        self._definition = definition
        self._registry = registry
        self._artifacts = artifacts
        self._redis_event_store = redis
        self._kafka_producer = producer
        self._hooks = hooks

    async def handle(self, event: EventEnvelope) -> None:
        if event.event_type == "processing.embedding-index.requested":
            await self._handle_embedding_reindex(event)
            return
        if event.event_type not in {"processing.requested", "processing.retry.requested"}:
            return
        is_retry = event.event_type == "processing.retry.requested"
        item = ProcessingWorkItem.model_validate(event.payload)
        existing = self._redis_event_store.get_state(processing_state_key(item.processing_id))
        existing_is_same_subject = existing is not None and str(existing.get("subject_id")) == str(item.subject_id)
        retry_is_redelivery = is_retry and existing is not None and existing.get("retry_event_id") == str(event.event_id)
        if existing is not None and existing_is_same_subject and existing.get("status") in TERMINAL_PROCESSING_STATUSES and not is_retry:
            logger.info(
                "processing：跳过已终止的重复任务 processing_id=%s status=%s",
                item.processing_id,
                existing["status"],
            )
            return
        if is_retry:
            if existing is None or not existing_is_same_subject:
                raise ValueError(f"Processing retry requires a matching run: {item.processing_id}")
            if retry_is_redelivery and existing.get("status") in TERMINAL_PROCESSING_STATUSES:
                logger.info("processing：跳过已完成的重试消息 processing_id=%s event_id=%s", item.processing_id, event.event_id)
                return
            if existing.get("status") not in TERMINAL_PROCESSING_STATUSES and not (
                retry_is_redelivery and existing.get("status") == "running"
            ):
                raise ValueError(f"Processing retry requires a terminal run: {item.processing_id}")
        if existing is not None and not existing_is_same_subject:
            # A deterministic processing ID may be reused after an old recording is deleted.
            # Remove stale live state before projecting this command.
            self._redis_event_store.delete(
                processing_state_key(item.processing_id),
                processing_stream_key(item.processing_id),
                f"task:{item.processing_id}:cancel",
            )
        previous_stages = cast(dict[str, dict[str, Any]], existing.get("stages", {})) if is_retry and existing is not None else {}
        now = datetime.now(UTC).isoformat()
        state: dict[str, Any] = {
            "processing_id": str(item.processing_id),
            "subject_type": item.subject_type,
            "subject_id": str(item.subject_id),
            "pipeline_name": item.pipeline_name,
            "pipeline_version": item.pipeline_version,
            "status": "running",
            "stages": {},
            "started_at": existing.get("started_at") if is_retry and existing is not None else now,
        }
        if state["started_at"] is None:
            state["started_at"] = now
        if existing is not None and existing.get("created_at") is not None:
            state["created_at"] = existing["created_at"]
        if is_retry and existing is not None:
            state["retry_event_id"] = str(event.event_id)
            raw_retry_count = existing.get("retry_count", 0)
            previous_retry_count = raw_retry_count if isinstance(raw_retry_count, int) else 0
            state["retry_count"] = previous_retry_count if retry_is_redelivery else previous_retry_count + 1
        if self._is_cancel_requested(item):
            await self._complete_cancelled(event, item, state)
            return
        if (item.pipeline_name, item.pipeline_version) != (self._definition.name, self._definition.version):
            raise ValueError(f"Unsupported pipeline definition: {item.pipeline_name}@{item.pipeline_version}")
        refs: dict[tuple[str | None, str], ArtifactRef] = {(None, ref.artifact_type): ref for ref in item.initial_artifacts}
        await self._state(event, item, state, "processing.started")
        required_failure: str | None = None
        optional_failure = False
        allow_legacy_restore = is_retry
        for node in self._definition.topologically_sorted_nodes():
            if self._is_cancel_requested(item):
                await self._complete_cancelled(event, item, state)
                return
            try:
                with execution_scope(ExecutionScope(kind="processing", id=item.processing_id)):
                    reused = await self._run_node(
                        event,
                        item,
                        node,
                        refs,
                        state,
                        attempt_offset=self._previous_attempt_count(node, previous_stages),
                        allow_legacy_restore=allow_legacy_restore,
                    )
                    if not reused:
                        allow_legacy_restore = False
            except ProcessingCancelledError:
                await self._complete_cancelled(event, item, state, node.name)
                return
            except Exception as error:
                previous_stage = cast(dict[str, Any], state["stages"].get(node.name, {}))
                state["stages"][node.name] = {
                    "status": "failed",
                    "attempt": previous_stage.get("attempt", 0),
                    "error": str(error)[:2000],
                }
                await self._state(event, item, state, "processing.stage.failed")
                if node.required:
                    required_failure = str(error) or type(error).__name__
                    break
                optional_failure = True
        if self._is_cancel_requested(item):
            await self._complete_cancelled(event, item, state)
            return
        if required_failure is not None:
            state.update(status="failed", error_message=required_failure, finished_at=datetime.now(UTC).isoformat())
        else:
            state.update(status="partial_failed" if optional_failure else "succeeded", finished_at=datetime.now(UTC).isoformat())
        self._hooks.run_state_changed(item.subject_id, cast(str, state["status"]), cast(str | None, state.get("error_message")))
        await self._state(event, item, state, "processing.completed")
        self._redis_event_store.finish(processing_state_key(item.processing_id), processing_stream_key(item.processing_id))

    async def _handle_embedding_reindex(self, event: EventEnvelope) -> None:
        request = EmbeddingReindexWorkItem.model_validate(event.payload)
        state = self._redis_event_store.get_state(processing_state_key(request.processing_id))
        if state is None:
            raise LookupError(f"Processing state not found: {request.processing_id}")
        stages = cast(dict[str, dict[str, Any]], state.get("stages", {}))
        if stages.get("embedding_indexing", {}).get("status") == "succeeded":
            logger.info("processing：跳过已成功的向量索引重试 processing_id=%s", request.processing_id)
            return
        item = ProcessingWorkItem(
            processing_id=request.processing_id,
            subject_type="recording",
            subject_id=request.subject_id,
            pipeline_name=self._definition.name,
            pipeline_version=self._definition.version,
            initial_artifacts=[request.chunks],
        )
        if self._is_cancel_requested(item):
            await self._complete_cancelled(event, item, state)
            return
        node = next((candidate for candidate in self._definition.nodes if candidate.name == "embedding_indexing"), None)
        if node is None:
            raise LookupError("Pipeline has no embedding_indexing node")
        state.update(status="running", error_message=None, finished_at=None)
        await self._state(event, item, state, "processing.started")
        refs: dict[tuple[str | None, str], ArtifactRef] = {
            (request.chunks.producer_stage or "build_search_chunks", request.chunks.artifact_type): request.chunks
        }
        try:
            with execution_scope(ExecutionScope(kind="processing", id=item.processing_id)):
                await self._run_node(event, item, node, refs, state)
        except ProcessingCancelledError:
            await self._complete_cancelled(event, item, state, node.name)
            return
        except Exception as error:
            state["stages"][node.name] = {"status": "failed", "error": str(error)[:2000]}
            await self._state(event, item, state, "processing.stage.failed")
            state.update(status="partial_failed", finished_at=datetime.now(UTC).isoformat())
        else:
            remaining_failures = any(
                name != node.name and stage.get("status") in {"failed", "cancelled"} for name, stage in cast(dict[str, dict[str, Any]], state["stages"]).items()
            )
            state.update(status="partial_failed" if remaining_failures else "succeeded", finished_at=datetime.now(UTC).isoformat())
        self._hooks.run_state_changed(request.subject_id, cast(str, state["status"]), cast(str | None, state.get("error_message")))
        await self._state(event, item, state, "processing.completed")
        self._redis_event_store.finish(processing_state_key(request.processing_id), processing_stream_key(request.processing_id))

    async def _run_node(
        self,
        event: EventEnvelope,
        item: ProcessingWorkItem,
        node: PipelineNode,
        refs: dict[tuple[str | None, str], ArtifactRef],
        state: dict[str, Any],
        *,
        attempt_offset: int = 0,
        allow_legacy_restore: bool = False,
    ) -> bool:
        stage = self._registry.get(node.stage_name, node.stage_version)
        retry_attempt = 0
        while True:
            if self._is_cancel_requested(item):
                raise ProcessingCancelledError
            retry_attempt += 1
            attempt = attempt_offset + retry_attempt
            state["stages"][node.name] = {"status": "running", "attempt": attempt}
            await self._state(event, item, state, "processing.stage.started")
            try:
                payload: dict[str, Any] = dict(node.input_payload or {})
                for binding in node.input_artifacts:
                    payload[binding.name] = refs[(binding.from_node, binding.artifact_type)]
                cache_config_value = getattr(stage, "cache_config", None)
                cache_config = cache_config_value() if callable(cache_config_value) else cache_config_value
                input_fingerprint = build_input_fingerprint(node.stage_name, node.stage_version, payload, cache_config)
                stage_input: object = payload
                input_model = getattr(stage, "input_model", None)
                if isinstance(input_model, type) and issubclass(input_model, BaseModel):
                    stage_input = input_model.model_validate(payload)
                stage_run_id = StageRunId(uuid5(item.processing_id, node.name))
                context = StageContext(
                    subject_id=PipelineSubjectId(item.subject_id),
                    pipeline_run_id=PipelineRunId(item.processing_id),
                    stage_run_id=stage_run_id,
                    attempt_count=attempt,
                    input_fingerprint=input_fingerprint,
                    allow_legacy_restore=allow_legacy_restore,
                    progress_reporter=_ProgressReporter(self._redis_event_store, item, node.name, state),
                )
                restore_value = getattr(stage, "try_restore", None)
                restore = cast(Callable[[StageContext, object], Awaitable[StageResult[Any] | None]] | None, restore_value if callable(restore_value) else None)
                restored_result = await restore(context, stage_input) if restore is not None else None
                reused = restored_result is not None
                if restored_result is None:
                    stage_task = asyncio.create_task(
                        stage.run(context, stage_input),
                        name=f"processing-{item.processing_id}-{node.name}",
                    )
                    while not stage_task.done():
                        await asyncio.wait((stage_task,), timeout=PROCESSING_CANCEL_POLL_SECONDS)
                        if self._is_cancel_requested(item):
                            stage_task.cancel()
                            await asyncio.gather(stage_task, return_exceptions=True)
                            raise ProcessingCancelledError
                    result = stage_task.result()
                    payloads = tuple(cast(ArtifactPayload, artifact) for artifact in result.artifacts)
                    written = tuple(
                        self._artifacts.write_json(
                            item.subject_id,
                            PipelineRunId(item.processing_id),
                            stage_run_id,
                            node.name,
                            artifact,
                            stage_version=node.stage_version,
                            input_fingerprint=input_fingerprint,
                        )
                        for artifact in payloads
                    )
                else:
                    result = restored_result
                    written = tuple(cast(ArtifactRef, artifact) for artifact in result.artifacts)
                if self._is_cancel_requested(item):
                    raise ProcessingCancelledError
                for artifact in written:
                    refs[(node.name, artifact.artifact_type)] = artifact
                output: object = result.output
                if isinstance(output, BaseModel):
                    output = output.model_dump(mode="json")
                elif is_dataclass(output):
                    output = asdict(cast(Any, output))
                self._hooks.stage_succeeded(item.subject_id, node.stage_name, output)
                state["stages"][node.name] = {
                    "status": "succeeded",
                    "attempt": attempt,
                    "reused": reused,
                    "artifacts": [artifact.model_dump(mode="json") for artifact in written],
                }
                await self._state(event, item, state, "processing.stage.succeeded")
                return reused
            except ProcessingCancelledError:
                raise
            except Exception:
                max_attempts = stage.retry_policy.max_attempts or 3
                if retry_attempt >= max_attempts:
                    raise
                await asyncio.sleep(stage.retry_policy.retry_delay_seconds(retry_attempt))

    @staticmethod
    def _previous_attempt_count(node: PipelineNode, previous_stages: dict[str, dict[str, Any]]) -> int:
        previous = previous_stages.get(node.name)
        if previous is None:
            return 0
        raw_attempt = previous.get("attempt")
        if isinstance(raw_attempt, int) and raw_attempt >= 0:
            return raw_attempt
        if previous.get("status") == "failed":
            return node.retry_policy.max_attempts or 3
        return 0

    def _is_cancel_requested(self, item: ProcessingWorkItem) -> bool:
        return self._redis_event_store.is_cancel_requested(str(item.processing_id)) or self._redis_event_store.is_cancel_requested(str(item.subject_id))

    async def _complete_cancelled(
        self,
        command: EventEnvelope,
        item: ProcessingWorkItem,
        state: dict[str, Any],
        active_node: str | None = None,
    ) -> None:
        if active_node is not None:
            stage = cast(dict[str, Any], state["stages"].get(active_node, {}))
            stage.update(status="cancelled", error="Processing cancelled")
            state["stages"][active_node] = stage
        state.update(status="cancelled", error_message=None, finished_at=datetime.now(UTC).isoformat())
        self._hooks.run_state_changed(item.subject_id, "cancelled", None)
        await self._state(command, item, state, "processing.completed")
        self._redis_event_store.finish(processing_state_key(item.processing_id), processing_stream_key(item.processing_id))

    async def _state(self, command: EventEnvelope, item: ProcessingWorkItem, state: dict[str, Any], event_type: str) -> None:
        state["updated_at"] = datetime.now(UTC).isoformat()
        self._redis_event_store.set_state(processing_state_key(item.processing_id), state)
        self._redis_event_store.append(processing_stream_key(item.processing_id), event_type, state)
        envelope = new_event(
            event_type,
            "processing-worker",
            correlation_id=command.correlation_id,
            causation_id=command.event_id,
            processing_id=item.processing_id,
            payload=state,
        )
        await self._kafka_producer.publish(Topics.PROCESSING_EVENTS, str(item.processing_id), envelope)
