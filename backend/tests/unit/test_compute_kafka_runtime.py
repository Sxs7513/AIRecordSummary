from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

from l1_foundation.infrastructure.storage.local import LocalStorage
from l1_foundation.messaging import EventEnvelope, KafkaEventProducer, new_event
from l1_foundation.streaming import RedisStreamStore
from l1_foundation.task_runtime.resources import ResourceQueue
from l1_foundation.worker import (
    ComputeCancelRequest,
    ComputeCompletedEvent,
    ComputeProgressEvent,
    ComputeReplyAddress,
    ComputeResultLocator,
    ComputeStartedEvent,
    ComputeTaskReply,
    ComputeTaskRequest,
    ComputeTaskSnapshot,
    ComputeTaskStatus,
    ExecutionScope,
)
from l1_foundation.worker.contracts import JsonObject
from l3_app.compute_worker.kafka_runtime import (
    KafkaComputeCancelHandler,
    KafkaComputeTaskHandler,
    compute_result_key,
    execution_scope_cancel_id,
)
from l3_app.compute_worker.runtime import ComputeWorkerRuntime


class _Runtime:
    def __init__(self) -> None:
        self.cancelled_scopes: list[ExecutionScope] = []

    def cancel_scope(self, scope: ExecutionScope) -> list[object]:
        self.cancelled_scopes.append(scope)
        return []


class _Redis:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    async def request_cancel(self, task_id: str) -> None:
        self.cancelled.append(task_id)


class _ResultRedis:
    def __init__(self) -> None:
        self.values: dict[str, tuple[dict[str, object], int | None]] = {}

    async def set_state(self, key: str, value: dict[str, object], *, ttl_seconds: int | None = None) -> None:
        self.values[key] = (value, ttl_seconds)


class _Producer:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, EventEnvelope]] = []

    async def publish(self, topic: str, key: str, event: EventEnvelope) -> None:
        self.messages.append((topic, key, event))


class _HandlerRedis:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, object]] = {}
        self.finished: list[tuple[str, str]] = []

    async def set_state(self, key: str, value: dict[str, object], *, ttl_seconds: int | None = None) -> None:
        del ttl_seconds
        self.values[key] = value

    async def append(self, _key: str, _event_type: str, _value: dict[str, object]) -> str:
        return "1-0"

    async def finish(self, state_key: str, stream_key: str) -> None:
        self.finished.append((state_key, stream_key))

    async def is_cancel_requested(self, _task_id: str) -> bool:
        return False


class _HandlerRuntime:
    def __init__(self, request: ComputeTaskRequest) -> None:
        now = datetime.now(UTC)
        self._snapshot = ComputeTaskSnapshot(
            task_id=request.task_id,
            operation=request.operation,
            operation_version=request.operation_version,
            resource_queue=request.resource_queue,
            status=ComputeTaskStatus.QUEUED,
            created_at=now,
        )

    async def submit(self, _request: ComputeTaskRequest) -> ComputeTaskSnapshot:
        return self._snapshot

    def status(self, _task_id: UUID) -> ComputeTaskSnapshot:
        return self._snapshot

    def cancel(self, _task_id: UUID) -> ComputeTaskSnapshot:
        return self._snapshot

    async def events(self, task_id: UUID) -> AsyncIterator[ComputeStartedEvent | ComputeProgressEvent | ComputeCompletedEvent]:
        now = datetime.now(UTC)
        self._snapshot = self._snapshot.model_copy(update={"status": ComputeTaskStatus.RUNNING, "started_at": now})
        yield ComputeStartedEvent(task_id=task_id, at=now)
        self._snapshot = self._snapshot.model_copy(update={"progress": 0.5, "message": "half"})
        yield ComputeProgressEvent(task_id=task_id, at=now, progress=0.5, message="half")
        result: JsonObject = {"value": "done"}
        self._snapshot = self._snapshot.model_copy(
            update={"status": ComputeTaskStatus.SUCCEEDED, "progress": 1.0, "result": result, "finished_at": now}
        )
        yield ComputeCompletedEvent(task_id=task_id, at=now, result=result)


def test_compute_cancel_handler_projects_scope_and_cancels_local_runtime() -> None:
    scope = ExecutionScope(kind="generation", id=uuid4())
    runtime = _Runtime()
    redis = _Redis()
    handler = KafkaComputeCancelHandler(cast(ComputeWorkerRuntime, runtime), cast(RedisStreamStore, redis))
    event = new_event(
        "compute.cancel.requested",
        "test",
        payload=ComputeCancelRequest(execution_scope=scope).model_dump(mode="json"),
    )

    asyncio.run(handler.handle(event))

    assert redis.cancelled == [execution_scope_cancel_id(scope)]
    assert runtime.cancelled_scopes == [scope]


def test_small_compute_result_uses_redis_locator(tmp_path: Path) -> None:
    task_id = uuid4()
    redis = _ResultRedis()
    storage = LocalStorage(tmp_path)
    handler = KafkaComputeTaskHandler(
        cast(ComputeWorkerRuntime, object()),
        cast(KafkaEventProducer, _Producer()),
        cast(RedisStreamStore, redis),
        storage,
        inline_result_limit_bytes=256 * 1024,
        result_ttl_seconds=123,
    )

    locator = asyncio.run(handler._persist_result(task_id, {"value": "small"}))

    assert locator.storage == "redis"
    assert locator.key == compute_result_key(task_id)
    assert redis.values[locator.key] == ({"value": "small"}, 123)


def test_large_compute_result_uses_file_locator(tmp_path: Path) -> None:
    task_id = uuid4()
    storage = LocalStorage(tmp_path)
    handler = KafkaComputeTaskHandler(
        cast(ComputeWorkerRuntime, object()),
        cast(KafkaEventProducer, _Producer()),
        cast(RedisStreamStore, _ResultRedis()),
        storage,
        inline_result_limit_bytes=16,
    )

    locator = asyncio.run(handler._persist_result(task_id, {"value": "x" * 100}))

    assert locator.storage == "file"
    assert locator.key == f"compute-results/{task_id}.json"
    assert json.loads(storage.get_file_by_key(locator.key).read_text(encoding="utf-8")) == {"value": "x" * 100}


def test_kafka_reply_contains_only_result_locator(tmp_path: Path) -> None:
    task_id = uuid4()
    producer = _Producer()
    handler = KafkaComputeTaskHandler(
        cast(ComputeWorkerRuntime, object()),
        cast(KafkaEventProducer, producer),
        cast(RedisStreamStore, _ResultRedis()),
        LocalStorage(tmp_path),
    )
    request = ComputeTaskRequest(
        task_id=task_id,
        operation="test",
        operation_version="1",
        resource_queue=ResourceQueue.CPU,
        reply_to=ComputeReplyAddress(topic="compute.results", requester_id="requester-a"),
    )
    envelope = new_event("compute.task.requested", "test", task_id=task_id, payload=request.model_dump(mode="json"))
    reply = ComputeTaskReply(
        task_id=task_id,
        requester_id="requester-a",
        status=ComputeTaskStatus.SUCCEEDED,
        result=ComputeResultLocator(storage="redis", key=compute_result_key(task_id), size_bytes=20),
    )

    asyncio.run(handler._publish_reply(envelope, request, reply))

    [(topic, key, event)] = producer.messages
    assert topic == "compute.results"
    assert key == "requester-a"
    assert event.payload["result"] == {"storage": "redis", "key": compute_result_key(task_id), "size_bytes": 20}
    assert "value" not in event.payload


def test_non_streaming_task_publishes_only_one_terminal_reply(tmp_path: Path) -> None:
    task_id = uuid4()
    request = ComputeTaskRequest(
        task_id=task_id,
        operation="test",
        operation_version="1",
        resource_queue=ResourceQueue.CPU,
        reply_to=ComputeReplyAddress(topic="compute.results", requester_id="requester-a"),
    )
    producer = _Producer()
    redis = _HandlerRedis()
    handler = KafkaComputeTaskHandler(
        cast(ComputeWorkerRuntime, _HandlerRuntime(request)),
        cast(KafkaEventProducer, producer),
        cast(RedisStreamStore, redis),
        LocalStorage(tmp_path),
    )
    envelope = new_event("compute.task.requested", "test", task_id=task_id, payload=request.model_dump(mode="json"))

    asyncio.run(handler.handle(envelope))

    assert len(producer.messages) == 1
    [(topic, key, event)] = producer.messages
    assert topic == "compute.results"
    assert key == "requester-a"
    assert event.event_type == "compute.task.reply"
    assert event.payload["status"] == "succeeded"
    assert "progress" not in event.payload
    assert "message" not in event.payload
