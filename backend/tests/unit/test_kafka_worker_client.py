from __future__ import annotations

import asyncio
from typing import cast
from uuid import uuid4

import pytest
from pydantic import BaseModel

from l1_foundation.messaging import EventEnvelope, KafkaEventProducer, SyncKafkaEventProducer
from l1_foundation.streaming import RedisStreamStore, SyncRedisStreamStore
from l1_foundation.task_runtime.resources import ResourceQueue
from l1_foundation.worker.contracts import ComputeCommand, ComputeTaskSnapshot, ExecutionScope, execution_scope
from l1_foundation.worker.errors import ComputeStateTimeoutError
from l1_foundation.worker.kafka_client import KafkaWorkerClient, SyncKafkaWorkerClient


class _MissingAsyncState:
    async def get_state(self, _key: str) -> None:
        return None


class _MissingSyncState:
    def get_state(self, _key: str) -> None:
        return None


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

    async def set_state_if_absent(self, key: str, state: dict[str, object]) -> bool:
        if key in self.values:
            return False
        self.values[key] = state
        return True


class _SyncProducer:
    def __init__(self) -> None:
        self.published = False

    def publish(self, *_args: object) -> None:
        self.published = True


class _SyncState:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, object]] = {}

    def set_state_if_absent(self, key: str, state: dict[str, object]) -> bool:
        if key in self.values:
            return False
        self.values[key] = state
        return True


class _CancellableState:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    async def get_state(self, _key: str) -> None:
        return None

    async def set_state_if_absent(self, _key: str, _state: dict[str, object]) -> bool:
        return True

    async def request_cancel(self, task_id: str) -> None:
        self.cancelled.append(task_id)


class _TestKafkaWorkerClient(KafkaWorkerClient):
    async def wait_for_state(self) -> ComputeTaskSnapshot:
        return await self._wait_for_state(uuid4())


class _TestSyncKafkaWorkerClient(SyncKafkaWorkerClient):
    def wait_for_state(self) -> ComputeTaskSnapshot:
        return self._wait_for_state(uuid4())


def test_async_compute_client_times_out_when_initial_state_never_appears() -> None:
    client = _TestKafkaWorkerClient(
        cast(KafkaEventProducer, object()),
        cast(RedisStreamStore, _MissingAsyncState()),
        poll_interval_seconds=0.001,
        state_wait_timeout_seconds=0.01,
    )

    with pytest.raises(ComputeStateTimeoutError, match="did not enter Redis state within 0.01 seconds"):
        asyncio.run(client.wait_for_state())


def test_sync_compute_client_times_out_when_initial_state_never_appears() -> None:
    client = _TestSyncKafkaWorkerClient(
        cast(SyncKafkaEventProducer, object()),
        cast(SyncRedisStreamStore, _MissingSyncState()),
        poll_interval_seconds=0.001,
        state_wait_timeout_seconds=0.01,
    )

    with pytest.raises(ComputeStateTimeoutError, match="did not enter Redis state within 0.01 seconds"):
        client.wait_for_state()


def test_async_compute_client_forwards_parent_cancellation() -> None:
    async def scenario() -> None:
        task_id = uuid4()
        producer = _Producer()
        state = _CancellableState()
        client = KafkaWorkerClient(
            cast(KafkaEventProducer, producer),
            cast(RedisStreamStore, state),
            poll_interval_seconds=0.001,
            state_wait_timeout_seconds=30,
        )
        execution = asyncio.create_task(
            client.execute(
                ComputeCommand(
                    task_id=task_id,
                    operation="test",
                    operation_version="1",
                    resource_queue=ResourceQueue.CPU,
                    input=_Input(value="value"),
                ),
                result_type=_Result,
            )
        )
        await asyncio.wait_for(producer.published.wait(), timeout=1)
        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution

        assert state.cancelled == [str(task_id)]

    asyncio.run(scenario())


def test_async_submit_projects_queued_state_after_kafka_acknowledgement() -> None:
    async def scenario() -> None:
        task_id = uuid4()
        producer = _Producer()
        state = _AsyncState()
        client = KafkaWorkerClient(cast(KafkaEventProducer, producer), cast(RedisStreamStore, state))

        snapshot = await client.submit(
            ComputeCommand(
                task_id=task_id,
                operation="test",
                operation_version="1",
                resource_queue=ResourceQueue.CPU,
                input=_Input(value="value"),
            )
        )

        assert producer.published.is_set()
        assert state.values[f"compute:{task_id}:state"] == snapshot.model_dump(mode="json")

    asyncio.run(scenario())


def test_compute_submission_serializes_the_current_execution_scope() -> None:
    async def scenario() -> None:
        task_id = uuid4()
        scope = ExecutionScope(kind="generation", id=uuid4())
        producer = _Producer()
        state = _AsyncState()
        client = KafkaWorkerClient(cast(KafkaEventProducer, producer), cast(RedisStreamStore, state))

        with execution_scope(scope):
            await client.submit(
                ComputeCommand(
                    task_id=task_id,
                    operation="test",
                    operation_version="1",
                    resource_queue=ResourceQueue.CPU,
                    input=_Input(value="value"),
                )
            )

        [(_, _, event)] = producer.messages
        assert event.payload["execution_scope"] == scope.model_dump(mode="json")

    asyncio.run(scenario())


def test_sync_submit_projects_queued_state_after_kafka_acknowledgement() -> None:
    task_id = uuid4()
    producer = _SyncProducer()
    state = _SyncState()
    client = SyncKafkaWorkerClient(cast(SyncKafkaEventProducer, producer), cast(SyncRedisStreamStore, state))

    snapshot = client.submit(
        ComputeCommand(
            task_id=task_id,
            operation="test",
            operation_version="1",
            resource_queue=ResourceQueue.CPU,
            input=_Input(value="value"),
        )
    )

    assert producer.published
    assert state.values[f"compute:{task_id}:state"] == snapshot.model_dump(mode="json")
