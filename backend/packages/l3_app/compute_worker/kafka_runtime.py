from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from l1_foundation.files import FileStore
from l1_foundation.messaging import EventEnvelope, KafkaEventProducer, new_event
from l1_foundation.streaming import RedisStreamStore
from l1_foundation.worker.contracts import (
    ComputeCancelledEvent,
    ComputeCancelRequest,
    ComputeCompletedEvent,
    ComputeFailedEvent,
    ComputeQueuedEvent,
    ComputeResultLocator,
    ComputeTaskReply,
    ComputeTaskRequest,
    ComputeTaskStatus,
    ExecutionScope,
    JsonObject,
)
from l3_app.compute_worker.runtime import ComputeWorkerRuntime


def compute_state_key(task_id: object) -> str:
    return f"compute:{task_id}:state"


def compute_stream_key(task_id: object) -> str:
    return f"compute:{task_id}:events"


def compute_result_key(task_id: object) -> str:
    return f"compute:{task_id}:result"


def execution_scope_cancel_id(scope: ExecutionScope) -> str:
    return f"execution:{scope.kind}:{scope.id}"


class KafkaComputeTaskHandler:
    """Bridge durable Kafka commands into the resource-aware compute runtime."""

    def __init__(
        self,
        runtime: ComputeWorkerRuntime,
        kafka_producer: KafkaEventProducer,
        redis_event_store: RedisStreamStore,
        file_store: FileStore,
        *,
        inline_result_limit_bytes: int = 256 * 1024,
        result_ttl_seconds: int = 86_400,
    ) -> None:
        if inline_result_limit_bytes < 1:
            raise ValueError("inline_result_limit_bytes must be positive")
        self._runtime = runtime
        self._kafka_producer = kafka_producer
        self._redis_event_store = redis_event_store
        self._file_store = file_store
        self._inline_result_limit_bytes = inline_result_limit_bytes
        self._result_ttl_seconds = result_ttl_seconds

    async def handle(self, envelope: EventEnvelope) -> None:
        if envelope.event_type != "compute.task.requested":
            return
        request = ComputeTaskRequest.model_validate(envelope.payload)
        if request.reply_to is None:
            raise ValueError("Compute task is missing reply_to")
        await self._runtime.submit(request)
        if await self._is_cancel_requested(request):
            self._runtime.cancel(request.task_id)
        await self._publish_state(request, "queued")
        if request.wait_for_subscriber:
            await self._publish_reply(
                envelope,
                request,
                ComputeTaskReply(
                    task_id=request.task_id,
                    requester_id=request.reply_to.requester_id,
                    status=ComputeTaskStatus.QUEUED,
                    stream_key=compute_stream_key(request.task_id),
                ),
            )
        cancel_monitor = asyncio.create_task(self._monitor_cancel(request), name=f"compute-cancel-{request.task_id}")
        try:
            async for event in self._runtime.events(request.task_id):
                event_data = event.model_dump(mode="json")
                if request.wait_for_subscriber:
                    await self._redis_event_store.append(compute_stream_key(request.task_id), event.type, event_data)
                if event.type not in {"delta", "heartbeat"}:
                    snapshot = self._runtime.status(request.task_id).model_dump(mode="json")
                    snapshot["result"] = None
                    await self._redis_event_store.set_state(compute_state_key(request.task_id), snapshot)
                if not request.wait_for_subscriber and isinstance(event, ComputeCompletedEvent | ComputeFailedEvent | ComputeCancelledEvent):
                    await self._publish_reply(envelope, request, await self._reply_for_event(request, event))
                if event.type in {"completed", "failed", "cancelled"}:
                    await self._redis_event_store.finish(compute_state_key(request.task_id), compute_stream_key(request.task_id))
                    return
        finally:
            cancel_monitor.cancel()
            await asyncio.gather(cancel_monitor, return_exceptions=True)

    async def _reply_for_event(self, request: ComputeTaskRequest, event: object) -> ComputeTaskReply:
        reply_to = request.reply_to
        if reply_to is None:
            raise ValueError("Compute task is missing reply_to")
        snapshot = self._runtime.status(request.task_id)
        locator: ComputeResultLocator | None = None
        error = None
        if isinstance(event, ComputeCompletedEvent):
            locator = await self._persist_result(request.task_id, event.result)
        elif isinstance(event, ComputeFailedEvent):
            error = event.error
        elif isinstance(event, ComputeCancelledEvent):
            error = None
        return ComputeTaskReply(
            task_id=request.task_id,
            requester_id=reply_to.requester_id,
            status=snapshot.status,
            result=locator,
            error=error,
        )

    async def _persist_result(self, task_id: object, result: JsonObject) -> ComputeResultLocator:
        serialized = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(serialized) <= self._inline_result_limit_bytes:
            key = compute_result_key(task_id)
            await self._redis_event_store.set_state(key, result, ttl_seconds=self._result_ttl_seconds)
            return ComputeResultLocator(storage="redis", key=key, size_bytes=len(serialized))
        key = f"compute-results/{task_id}.json"
        with TemporaryDirectory(prefix="compute-result-") as temporary_directory:
            source = Path(temporary_directory) / "result.json"
            source.write_bytes(serialized)
            await asyncio.to_thread(self._file_store.put_file, source, key=key)
        return ComputeResultLocator(storage="file", key=key, size_bytes=len(serialized))

    async def _publish_reply(
        self,
        envelope: EventEnvelope,
        request: ComputeTaskRequest,
        reply: ComputeTaskReply,
    ) -> None:
        reply_to = request.reply_to
        if reply_to is None:
            raise ValueError("Compute task is missing reply_to")
        await self._kafka_producer.publish(
            reply_to.topic,
            reply_to.requester_id,
            new_event(
                "compute.task.reply",
                "compute-worker",
                correlation_id=envelope.correlation_id,
                causation_id=envelope.event_id,
                task_id=request.task_id,
                payload=reply.model_dump(mode="json"),
            ),
        )

    async def _monitor_cancel(self, request: ComputeTaskRequest) -> None:
        while True:
            if await self._is_cancel_requested(request):
                self._runtime.cancel(request.task_id)
                return
            await asyncio.sleep(0.2)

    async def _is_cancel_requested(self, request: ComputeTaskRequest) -> bool:
        if await self._redis_event_store.is_cancel_requested(str(request.task_id)):
            return True
        return request.execution_scope is not None and await self._redis_event_store.is_cancel_requested(execution_scope_cancel_id(request.execution_scope))

    async def _publish_state(self, request: ComputeTaskRequest, status: str) -> None:
        snapshot = self._runtime.status(request.task_id).model_dump(mode="json")
        snapshot["result"] = None
        await self._redis_event_store.set_state(compute_state_key(request.task_id), snapshot)
        if request.wait_for_subscriber:
            queued = ComputeQueuedEvent(task_id=request.task_id, at=datetime.now(UTC))
            await self._redis_event_store.append(compute_stream_key(request.task_id), status, queued.model_dump(mode="json"))


class KafkaComputeCancelHandler:
    """Project reliable cancellation commands into Redis and the local runtime."""

    def __init__(self, runtime: ComputeWorkerRuntime, redis_event_store: RedisStreamStore) -> None:
        self._runtime = runtime
        self._redis_event_store = redis_event_store

    async def handle(self, envelope: EventEnvelope) -> None:
        if envelope.event_type != "compute.cancel.requested":
            return
        request = ComputeCancelRequest.model_validate(envelope.payload)
        if request.task_id is not None:
            await self._redis_event_store.request_cancel(str(request.task_id))
            try:
                self._runtime.cancel(request.task_id)
            except LookupError:
                return
            return
        scope = request.execution_scope
        if scope is None:
            raise ValueError("Compute scope cancellation is missing execution_scope")
        await self._redis_event_store.request_cancel(execution_scope_cancel_id(scope))
        self._runtime.cancel_scope(scope)
