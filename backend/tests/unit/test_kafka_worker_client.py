from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from threading import Event as ThreadEvent
from threading import Thread
from typing import cast
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from l1_foundation.messaging import EventEnvelope, KafkaEventProducer, SyncKafkaEventProducer, new_event
from l1_foundation.streaming import RedisStreamStore, SyncRedisStreamStore
from l1_foundation.streaming.redis import RedisStreamEvent
from l1_foundation.task_runtime.resources import ResourceQueue
from l1_foundation.worker.contracts import (
    ComputeCommand,
    ComputeResultLocator,
    ComputeTaskReply,
    ComputeTaskStatus,
    ExecutionScope,
    execution_scope,
)
from l1_foundation.worker.errors import ComputeReplyTimeoutError
from l1_foundation.worker.kafka_client import KafkaWorkerClient, SyncKafkaWorkerClient


class _Input(BaseModel):
    value: str


class _Result(BaseModel):
    value: str


class _Producer:
    def __init__(self) -> None:
        self.published = asyncio.Event()
        self.messages: list[tuple[str, str, EventEnvelope]] = []

    async def publish(self, topic: str, key: str, event: EventEnvelope) -> None:
        self.messages.append((topic, key, event))
        self.published.set()


class _AsyncState:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, object]] = {}
        self.cancelled: list[str] = []
        self.deleted: list[str] = []
        self.stream_events: list[RedisStreamEvent] = []

    async def set_state_if_absent(self, key: str, state: dict[str, object]) -> bool:
        if key in self.values:
            return False
        self.values[key] = state
        return True

    async def get_state(self, key: str) -> dict[str, object] | None:
        return self.values.get(key)

    async def delete(self, *keys: str) -> int:
        self.deleted.extend(keys)
        for key in keys:
            self.values.pop(key, None)
        return len(keys)

    async def request_cancel(self, task_id: str) -> None:
        self.cancelled.append(task_id)

    async def read(self, _stream: str, _cursor: str) -> list[RedisStreamEvent]:
        events, self.stream_events = self.stream_events, []
        return events


class _SyncProducer:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, EventEnvelope]] = []
        self.on_publish: Callable[[], None] | None = None

    def publish(self, topic: str, key: str, event: EventEnvelope) -> None:
        self.messages.append((topic, key, event))
        if self.on_publish is not None:
            self.on_publish()


class _SyncState:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, object]] = {}
        self.deleted: list[str] = []
        self.on_state_initialized: Callable[[str], None] | None = None

    def set_state_if_absent(self, key: str, state: dict[str, object]) -> bool:
        if key in self.values:
            return False
        self.values[key] = state
        if self.on_state_initialized is not None:
            self.on_state_initialized(key)
        return True

    def get_state(self, key: str) -> dict[str, object] | None:
        return self.values.get(key)

    def delete(self, *keys: str) -> int:
        self.deleted.extend(keys)
        for key in keys:
            self.values.pop(key, None)
        return len(keys)

    def request_cancel(self, _task_id: str) -> None:
        return None


def _command(task_id: UUID) -> ComputeCommand[_Input]:
    return ComputeCommand(
        task_id=task_id,
        operation="test",
        operation_version="1",
        resource_queue=ResourceQueue.CPU,
        input=_Input(value="value"),
    )


def _reply(task_id: UUID, requester_id: str, status: ComputeTaskStatus, locator: ComputeResultLocator | None = None) -> EventEnvelope:
    return new_event(
        "compute.task.reply",
        "test",
        task_id=task_id,
        payload=ComputeTaskReply(task_id=task_id, requester_id=requester_id, status=status, result=locator).model_dump(mode="json"),
    )


def test_async_stream_times_out_when_no_locator_reply_arrives() -> None:
    async def scenario() -> None:
        client = KafkaWorkerClient(
            cast(KafkaEventProducer, _Producer()),
            cast(RedisStreamStore, _AsyncState()),
            requester_id="requester",
            reply_wait_timeout_seconds=0.01,
        )
        with pytest.raises(ComputeReplyTimeoutError, match="did not receive a Kafka reply"):
            await client.execute_streaming(_command(uuid4()), result_type=_Result)

    asyncio.run(scenario())


def test_async_execute_routes_reply_and_consumes_redis_result() -> None:
    async def scenario() -> None:
        task_id = uuid4()
        producer = _Producer()
        state = _AsyncState()
        client = KafkaWorkerClient(
            cast(KafkaEventProducer, producer),
            cast(RedisStreamStore, state),
            requester_id="requester-a",
        )
        execution = asyncio.create_task(client.execute(_command(task_id), result_type=_Result))
        await asyncio.wait_for(producer.published.wait(), timeout=1)
        await client._handle_reply(_reply(task_id, "another-requester", ComputeTaskStatus.SUCCEEDED))
        result_key = f"compute:{task_id}:result"
        state.values[result_key] = {"value": "done"}
        await client._handle_reply(
            _reply(task_id, "requester-a", ComputeTaskStatus.SUCCEEDED, ComputeResultLocator(storage="redis", key=result_key, size_bytes=16))
        )

        assert await execution == _Result(value="done")
        assert state.deleted == [result_key]

    asyncio.run(scenario())


def test_async_execute_reads_progress_from_redis_state() -> None:
    async def scenario() -> None:
        task_id = uuid4()
        producer = _Producer()
        state = _AsyncState()
        client = KafkaWorkerClient(
            cast(KafkaEventProducer, producer),
            cast(RedisStreamStore, state),
            requester_id="requester-progress",
            progress_poll_interval_seconds=0.001,
        )
        reported = asyncio.Event()
        progress: list[tuple[float, str | None]] = []

        def on_progress(value: float, message: str | None) -> None:
            progress.append((value, message))
            reported.set()

        execution = asyncio.create_task(client.execute(_command(task_id), result_type=_Result, on_progress=on_progress))
        await asyncio.wait_for(producer.published.wait(), timeout=1)
        state.values[f"compute:{task_id}:state"].update(status="running", progress=0.5, message="half")
        await asyncio.wait_for(reported.wait(), timeout=1)
        result_key = f"compute:{task_id}:result"
        state.values[result_key] = {"value": "done"}
        await client._handle_reply(
            _reply(
                task_id,
                "requester-progress",
                ComputeTaskStatus.SUCCEEDED,
                ComputeResultLocator(storage="redis", key=result_key, size_bytes=16),
            )
        )

        assert await execution == _Result(value="done")
        assert progress == [(0.5, "half")]

    asyncio.run(scenario())


def test_async_compute_client_forwards_parent_cancellation() -> None:
    async def scenario() -> None:
        task_id = uuid4()
        producer = _Producer()
        state = _AsyncState()
        client = KafkaWorkerClient(cast(KafkaEventProducer, producer), cast(RedisStreamStore, state), requester_id="requester")
        execution = asyncio.create_task(client.execute(_command(task_id), result_type=_Result))
        await asyncio.wait_for(producer.published.wait(), timeout=1)
        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution
        assert state.cancelled == [str(task_id)]

    asyncio.run(scenario())


def test_async_submit_adds_reply_address_and_current_execution_scope() -> None:
    async def scenario() -> None:
        task_id = uuid4()
        scope = ExecutionScope(kind="generation", id=uuid4())
        producer = _Producer()
        state = _AsyncState()
        client = KafkaWorkerClient(
            cast(KafkaEventProducer, producer),
            cast(RedisStreamStore, state),
            requester_id="requester-a",
        )
        with execution_scope(scope):
            snapshot = await client.submit(_command(task_id))

        [(_, _, event)] = producer.messages
        assert event.payload["execution_scope"] == scope.model_dump(mode="json")
        assert event.payload["reply_to"] == {"topic": "compute.results", "requester_id": "requester-a"}
        assert state.values[f"compute:{task_id}:state"] == snapshot.model_dump(mode="json")

    asyncio.run(scenario())


def test_async_streaming_waits_for_stream_locator_then_reads_redis_stream() -> None:
    async def scenario() -> None:
        task_id = uuid4()
        producer = _Producer()
        state = _AsyncState()
        now = datetime.now(UTC)
        state.stream_events = [
            RedisStreamEvent("1-0", "delta", {"type": "delta", "task_id": str(task_id), "at": now.isoformat(), "text": "hello"}),
            RedisStreamEvent(
                "2-0",
                "completed",
                {"type": "completed", "task_id": str(task_id), "at": now.isoformat(), "result": {"value": "done"}},
            ),
        ]
        client = KafkaWorkerClient(
            cast(KafkaEventProducer, producer),
            cast(RedisStreamStore, state),
            requester_id="requester-stream",
        )
        deltas: list[str] = []
        execution = asyncio.create_task(client.execute_streaming(_command(task_id), result_type=_Result, on_delta=deltas.append))
        await asyncio.wait_for(producer.published.wait(), timeout=1)
        stream_reply = ComputeTaskReply(
            task_id=task_id,
            requester_id="requester-stream",
            status=ComputeTaskStatus.QUEUED,
            stream_key=f"compute:{task_id}:events",
        )
        await client._handle_reply(
            new_event("compute.task.reply", "test", task_id=task_id, payload=stream_reply.model_dump(mode="json"))
        )

        assert await execution == _Result(value="done")
        assert deltas == ["hello"]

    asyncio.run(scenario())


def test_sync_execute_routes_reply_and_consumes_redis_result() -> None:
    task_id = uuid4()
    producer = _SyncProducer()
    state = _SyncState()
    client = SyncKafkaWorkerClient(
        cast(SyncKafkaEventProducer, producer),
        cast(SyncRedisStreamStore, state),
        requester_id="requester-sync",
    )
    result_key = f"compute:{task_id}:result"
    state.values[result_key] = {"value": "done"}

    def publish_replies() -> None:
        producer.on_publish = None
        client._handle_reply(
            _reply(task_id, "requester-sync", ComputeTaskStatus.SUCCEEDED, ComputeResultLocator(storage="redis", key=result_key, size_bytes=16))
        )

    producer.on_publish = publish_replies

    assert client.execute(_command(task_id), result_type=_Result) == _Result(value="done")
    assert state.deleted == [result_key]


def test_sync_execute_reads_progress_from_redis_state() -> None:
    task_id = uuid4()
    producer = _SyncProducer()
    state = _SyncState()
    client = SyncKafkaWorkerClient(
        cast(SyncKafkaEventProducer, producer),
        cast(SyncRedisStreamStore, state),
        requester_id="requester-sync-progress",
        progress_poll_interval_seconds=0.001,
    )
    progress_reported = ThreadEvent()
    progress: list[tuple[float, str | None]] = []
    result_key = f"compute:{task_id}:result"

    def deliver_result(state_key: str) -> None:
        def run() -> None:
            state.values[state_key].update(status="running", progress=0.25, message="quarter")
            assert progress_reported.wait(timeout=1)
            state.values[result_key] = {"value": "done"}
            client._handle_reply(
                _reply(
                    task_id,
                    "requester-sync-progress",
                    ComputeTaskStatus.SUCCEEDED,
                    ComputeResultLocator(storage="redis", key=result_key, size_bytes=16),
                )
            )

        Thread(target=run, daemon=True).start()

    state.on_state_initialized = deliver_result

    def on_progress(value: float, message: str | None) -> None:
        progress.append((value, message))
        progress_reported.set()

    assert client.execute(_command(task_id), result_type=_Result, on_progress=on_progress) == _Result(value="done")
    assert progress == [(0.25, "quarter")]
