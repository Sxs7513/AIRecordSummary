from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import replace
from datetime import UTC, datetime
from time import monotonic, sleep
from typing import TypeVar
from uuid import UUID

from pydantic import BaseModel

from l1_foundation.messaging import KafkaEventProducer, SyncKafkaEventProducer, Topics, new_event
from l1_foundation.streaming import RedisStreamStore, SyncRedisStreamStore
from l1_foundation.task_runtime.resources import ResourceQueue
from l1_foundation.worker.client import SyncWorkerClient, WorkerClient
from l1_foundation.worker.contracts import (
    ComputeCancelledEvent,
    ComputeCancelRequest,
    ComputeCommand,
    ComputeCompletedEvent,
    ComputeDeltaEvent,
    ComputeEvent,
    ComputeFailedEvent,
    ComputeProgressEvent,
    ComputeTaskError,
    ComputeTaskSnapshot,
    ComputeTaskStatus,
    parse_compute_event,
)
from l1_foundation.worker.errors import ComputeRemoteError, ComputeStateTimeoutError, ComputeStreamDisconnectedError, ComputeTaskNotFoundError

InputT = TypeVar("InputT", bound=BaseModel)
ResultT = TypeVar("ResultT", bound=BaseModel)
DEFAULT_STATE_WAIT_TIMEOUT_SECONDS = 30.0


def _task_topic(queue: ResourceQueue) -> str:
    return {
        ResourceQueue.IO: Topics.COMPUTE_TASKS_IO,
        ResourceQueue.CPU: Topics.COMPUTE_TASKS_CPU,
        ResourceQueue.GPU_HIGH: Topics.COMPUTE_TASKS_GPU_HIGH,
        ResourceQueue.GPU_NORMAL: Topics.COMPUTE_TASKS_GPU_NORMAL,
    }[queue]


def _state_key(task_id: UUID) -> str:
    return f"compute:{task_id}:state"


def _stream_key(task_id: UUID) -> str:
    return f"compute:{task_id}:events"


def _queued_snapshot[CommandInputT: BaseModel](command: ComputeCommand[CommandInputT]) -> ComputeTaskSnapshot:
    return ComputeTaskSnapshot(
        task_id=command.task_id,
        operation=command.operation,
        operation_version=command.operation_version,
        resource_queue=command.resource_queue,
        status=ComputeTaskStatus.QUEUED,
        created_at=datetime.now(UTC),
    )


def _result_or_raise[OutputT: BaseModel](snapshot: ComputeTaskSnapshot, result_type: type[OutputT]) -> OutputT:
    if snapshot.status == ComputeTaskStatus.SUCCEEDED and snapshot.result is not None:
        return result_type.model_validate(snapshot.result)
    if snapshot.error is not None:
        raise ComputeRemoteError(snapshot.error)
    raise ComputeRemoteError(
        ComputeTaskError(code=snapshot.status.value, message=f"Compute task ended with status={snapshot.status.value}", retryable=False)
    )


class KafkaWorkerClient(WorkerClient):
    """Async compute RPC facade backed by Kafka commands and Redis live state."""

    def __init__(
        self,
        producer: KafkaEventProducer,
        redis: RedisStreamStore,
        *,
        poll_interval_seconds: float = 0.2,
        state_wait_timeout_seconds: float = DEFAULT_STATE_WAIT_TIMEOUT_SECONDS,
    ) -> None:
        if state_wait_timeout_seconds <= 0:
            raise ValueError("state_wait_timeout_seconds must be positive")
        self._kafka_producer = producer
        self._redis_store = redis
        self._kafka_poll_interval = poll_interval_seconds
        self._state_wait_timeout = state_wait_timeout_seconds

    async def __aenter__(self) -> KafkaWorkerClient:
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None

    async def close(self) -> None:
        return None

    async def ready(self) -> None:
        await self._redis_store.ping()

    async def submit(self, command: ComputeCommand[InputT]) -> ComputeTaskSnapshot:
        request = command.to_request()
        event = new_event(
            "compute.task.requested",
            "application",
            correlation_id=command.task_id,
            task_id=command.task_id,
            payload=request.model_dump(mode="json"),
        )
        await self._kafka_producer.publish(_task_topic(command.resource_queue), str(command.task_id), event)
        snapshot = _queued_snapshot(command)
        # Kafka is the durable command source. Once it has acknowledged the
        # publish, make the accepted/queued projection visible immediately;
        # the Compute worker will subsequently own all state transitions.
        await self._redis_store.set_state_if_absent(_state_key(command.task_id), snapshot.model_dump(mode="json"))
        return snapshot

    async def status(self, task_id: UUID) -> ComputeTaskSnapshot:
        state = await self._redis_store.get_state(_state_key(task_id))
        if state is None:
            raise ComputeTaskNotFoundError(f"Compute task not found: {task_id}")
        return ComputeTaskSnapshot.model_validate(state)

    async def cancel(self, task_id: UUID) -> ComputeTaskSnapshot:
        await self._request_cancel(task_id)
        return await self.status(task_id)

    async def _request_cancel(self, task_id: UUID) -> None:
        request = ComputeCancelRequest(task_id=task_id)
        await self._kafka_producer.publish(
            Topics.COMPUTE_CANCEL,
            str(task_id),
            new_event(
                "compute.cancel.requested",
                "application",
                correlation_id=task_id,
                task_id=task_id,
                payload=request.model_dump(mode="json"),
            ),
        )
        await self._redis_store.request_cancel(str(task_id))

    async def execute(
        self,
        command: ComputeCommand[InputT],
        *,
        result_type: type[ResultT],
        on_progress: Callable[[float, str | None], None] | None = None,
    ) -> ResultT:
        try:
            await self.submit(replace(command, wait_for_subscriber=False))
            snapshot = await self._wait_for_state(command.task_id)
            while not snapshot.status.is_terminal:
                if on_progress is not None and snapshot.progress is not None:
                    on_progress(snapshot.progress, snapshot.message)
                await asyncio.sleep(self._kafka_poll_interval)
                snapshot = await self.status(command.task_id)
            return _result_or_raise(snapshot, result_type)
        except asyncio.CancelledError:
            await self._request_cancel(command.task_id)
            raise

    async def stream(self, command: ComputeCommand[InputT]) -> AsyncIterator[ComputeEvent]:
        await self.submit(replace(command, wait_for_subscriber=True))
        cursor = "0-0"
        try:
            while True:
                events = await self._redis_store.read(_stream_key(command.task_id), cursor)
                for raw in events:
                    cursor = raw.id
                    event = parse_compute_event(raw.data)
                    yield event
                    if isinstance(event, ComputeCompletedEvent | ComputeFailedEvent | ComputeCancelledEvent):
                        return
        except asyncio.CancelledError:
            await self._request_cancel(command.task_id)
            raise

    async def execute_streaming(
        self,
        command: ComputeCommand[InputT],
        *,
        result_type: type[ResultT],
        on_progress: Callable[[float, str | None], None] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> ResultT:
        async for event in self.stream(command):
            if isinstance(event, ComputeProgressEvent) and on_progress is not None:
                on_progress(event.progress, event.message)
            elif isinstance(event, ComputeDeltaEvent) and on_delta is not None:
                on_delta(event.text)
            elif isinstance(event, ComputeCompletedEvent):
                return result_type.model_validate(event.result)
            elif isinstance(event, ComputeFailedEvent):
                raise ComputeRemoteError(event.error)
            elif isinstance(event, ComputeCancelledEvent):
                raise ComputeRemoteError(ComputeTaskError(code="cancelled", message="Compute task was cancelled", retryable=True))
        raise ComputeStreamDisconnectedError("Compute Redis stream ended without a terminal event")

    async def _wait_for_state(self, task_id: UUID) -> ComputeTaskSnapshot:
        deadline = monotonic() + self._state_wait_timeout
        while True:
            state = await self._redis_store.get_state(_state_key(task_id))
            if state is not None:
                return ComputeTaskSnapshot.model_validate(state)
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise ComputeStateTimeoutError(
                    f"Compute task {task_id} did not enter Redis state within {self._state_wait_timeout:g} seconds"
                )
            await asyncio.sleep(min(self._kafka_poll_interval, remaining))


class SyncKafkaWorkerClient(SyncWorkerClient):
    """Sync compute RPC facade for thread-based stages."""

    def __init__(
        self,
        producer: SyncKafkaEventProducer,
        redis: SyncRedisStreamStore,
        *,
        poll_interval_seconds: float = 0.2,
        state_wait_timeout_seconds: float = DEFAULT_STATE_WAIT_TIMEOUT_SECONDS,
    ) -> None:
        if state_wait_timeout_seconds <= 0:
            raise ValueError("state_wait_timeout_seconds must be positive")
        self._kafka_producer = producer
        self._redis_store = redis
        self._kafka_poll_interval = poll_interval_seconds
        self._state_wait_timeout = state_wait_timeout_seconds

    def close(self) -> None:
        return None

    def ready(self) -> None:
        self._redis_store.ping()

    def submit(self, command: ComputeCommand[InputT]) -> ComputeTaskSnapshot:
        request = command.to_request()
        self._kafka_producer.publish(
            _task_topic(command.resource_queue),
            str(command.task_id),
            new_event(
                "compute.task.requested",
                "application",
                correlation_id=command.task_id,
                task_id=command.task_id,
                payload=request.model_dump(mode="json"),
            ),
        )
        snapshot = _queued_snapshot(command)
        self._redis_store.set_state_if_absent(_state_key(command.task_id), snapshot.model_dump(mode="json"))
        return snapshot

    def status(self, task_id: UUID) -> ComputeTaskSnapshot:
        state = self._redis_store.get_state(_state_key(task_id))
        if state is None:
            raise ComputeTaskNotFoundError(f"Compute task not found: {task_id}")
        return ComputeTaskSnapshot.model_validate(state)

    def cancel(self, task_id: UUID) -> ComputeTaskSnapshot:
        self._request_cancel(task_id)
        return self.status(task_id)

    def _request_cancel(self, task_id: UUID) -> None:
        request = ComputeCancelRequest(task_id=task_id)
        self._kafka_producer.publish(
            Topics.COMPUTE_CANCEL,
            str(task_id),
            new_event(
                "compute.cancel.requested",
                "application",
                correlation_id=task_id,
                task_id=task_id,
                payload=request.model_dump(mode="json"),
            ),
        )
        self._redis_store.request_cancel(str(task_id))

    def execute(
        self,
        command: ComputeCommand[InputT],
        *,
        result_type: type[ResultT],
        on_progress: Callable[[float, str | None], None] | None = None,
    ) -> ResultT:
        self.submit(replace(command, wait_for_subscriber=False))
        snapshot = self._wait_for_state(command.task_id)
        while not snapshot.status.is_terminal:
            if on_progress is not None and snapshot.progress is not None:
                on_progress(snapshot.progress, snapshot.message)
            sleep(self._kafka_poll_interval)
            snapshot = self.status(command.task_id)
        return _result_or_raise(snapshot, result_type)

    def stream(self, command: ComputeCommand[InputT]) -> Iterator[ComputeEvent]:
        self.submit(replace(command, wait_for_subscriber=True))
        cursor = "0-0"
        while True:
            events = self._redis_store.read(_stream_key(command.task_id), cursor)
            for raw in events:
                cursor = raw.id
                event = parse_compute_event(raw.data)
                yield event
                if isinstance(event, ComputeCompletedEvent | ComputeFailedEvent | ComputeCancelledEvent):
                    return

    def execute_streaming(
        self,
        command: ComputeCommand[InputT],
        *,
        result_type: type[ResultT],
        on_progress: Callable[[float, str | None], None] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> ResultT:
        for event in self.stream(command):
            if isinstance(event, ComputeProgressEvent) and on_progress is not None:
                on_progress(event.progress, event.message)
            elif isinstance(event, ComputeDeltaEvent) and on_delta is not None:
                on_delta(event.text)
            elif isinstance(event, ComputeCompletedEvent):
                return result_type.model_validate(event.result)
            elif isinstance(event, ComputeFailedEvent):
                raise ComputeRemoteError(event.error)
            elif isinstance(event, ComputeCancelledEvent):
                raise ComputeRemoteError(ComputeTaskError(code="cancelled", message="Compute task was cancelled", retryable=True))
        raise ComputeStreamDisconnectedError("Compute Redis stream ended without a terminal event")

    def _wait_for_state(self, task_id: UUID) -> ComputeTaskSnapshot:
        deadline = monotonic() + self._state_wait_timeout
        while True:
            state = self._redis_store.get_state(_state_key(task_id))
            if state is not None:
                return ComputeTaskSnapshot.model_validate(state)
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise ComputeStateTimeoutError(
                    f"Compute task {task_id} did not enter Redis state within {self._state_wait_timeout:g} seconds"
                )
            sleep(min(self._kafka_poll_interval, remaining))
