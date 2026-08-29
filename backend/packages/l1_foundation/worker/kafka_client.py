from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import replace
from datetime import UTC, datetime
from queue import Empty, Queue
from threading import Lock
from typing import TypeVar, cast
from uuid import UUID, uuid4

from pydantic import BaseModel

from l1_foundation.files import FileStore
from l1_foundation.messaging import (
    EventEnvelope,
    KafkaEventConsumer,
    KafkaEventProducer,
    SyncKafkaEventConsumer,
    SyncKafkaEventProducer,
    Topics,
    new_event,
)
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
    ComputeReplyAddress,
    ComputeResultLocator,
    ComputeTaskError,
    ComputeTaskReply,
    ComputeTaskSnapshot,
    ComputeTaskStatus,
    JsonObject,
    parse_compute_event,
)
from l1_foundation.worker.errors import ComputeRemoteError, ComputeReplyTimeoutError, ComputeStreamDisconnectedError, ComputeTaskNotFoundError

InputT = TypeVar("InputT", bound=BaseModel)
ResultT = TypeVar("ResultT", bound=BaseModel)
DEFAULT_REPLY_WAIT_TIMEOUT_SECONDS = 30.0
DEFAULT_PROGRESS_POLL_INTERVAL_SECONDS = 0.2
logger = logging.getLogger("worker")


def _task_topic(queue: ResourceQueue) -> str:
    return {
        ResourceQueue.IO: Topics.COMPUTE_TASKS_IO,
        ResourceQueue.CPU: Topics.COMPUTE_TASKS_CPU,
        ResourceQueue.GPU_HIGH: Topics.COMPUTE_TASKS_GPU_HIGH,
        ResourceQueue.GPU_NORMAL: Topics.COMPUTE_TASKS_GPU_NORMAL,
    }[queue]


def _state_key(task_id: UUID) -> str:
    return f"compute:{task_id}:state"


def _queued_snapshot[CommandInputT: BaseModel](command: ComputeCommand[CommandInputT]) -> ComputeTaskSnapshot:
    return ComputeTaskSnapshot(
        task_id=command.task_id,
        operation=command.operation,
        operation_version=command.operation_version,
        resource_queue=command.resource_queue,
        status=ComputeTaskStatus.QUEUED,
        created_at=datetime.now(UTC),
    )


def _terminal_error(reply: ComputeTaskReply) -> ComputeRemoteError:
    if reply.error is not None:
        return ComputeRemoteError(reply.error)
    return ComputeRemoteError(
        ComputeTaskError(code=reply.status.value, message=f"Compute task ended with status={reply.status.value}", retryable=False)
    )


class KafkaWorkerClient(WorkerClient):
    """Async request/reply compute client with Redis/FileStore result locators."""

    def __init__(
        self,
        producer: KafkaEventProducer,
        redis: RedisStreamStore,
        file_store: FileStore | None = None,
        *,
        requester_id: str | None = None,
        reply_consumer: KafkaEventConsumer | None = None,
        reply_wait_timeout_seconds: float = DEFAULT_REPLY_WAIT_TIMEOUT_SECONDS,
        progress_poll_interval_seconds: float = DEFAULT_PROGRESS_POLL_INTERVAL_SECONDS,
    ) -> None:
        if reply_wait_timeout_seconds <= 0:
            raise ValueError("reply_wait_timeout_seconds must be positive")
        if progress_poll_interval_seconds <= 0:
            raise ValueError("progress_poll_interval_seconds must be positive")
        self._kafka_producer = producer
        self._redis_store = redis
        self._file_store = file_store
        self._requester_id = requester_id or uuid4().hex
        self._reply_topic = Topics.COMPUTE_RESULTS
        self._reply_consumer = reply_consumer
        self._reply_task: asyncio.Task[None] | None = None
        self._reply_wait_timeout = reply_wait_timeout_seconds
        self._progress_poll_interval = progress_poll_interval_seconds
        self._waiters: dict[UUID, asyncio.Queue[ComputeTaskReply]] = {}

    async def __aenter__(self) -> KafkaWorkerClient:
        await self.ready()
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        await self.close()

    async def close(self) -> None:
        task = self._reply_task
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self._reply_task = None
        if self._reply_consumer is not None:
            await self._reply_consumer.stop()

    async def ready(self) -> None:
        await self._redis_store.ping()
        if self._reply_task is not None:
            return
        if self._reply_consumer is None:
            self._reply_consumer = KafkaEventConsumer(
                [self._reply_topic],
                bootstrap_servers=self._kafka_producer.bootstrap_servers,
                group_id=f"compute-replies-{self._requester_id}",
                client_id=f"{self._kafka_producer.client_id}-replies",
                auto_offset_reset="latest",
            )
        await self._reply_consumer.start()
        self._reply_task = asyncio.create_task(self._reply_consumer.run(self._handle_reply), name=f"compute-replies-{self._requester_id}")
        try:
            await self._reply_consumer.wait_for_assignment()
        except BaseException:
            await self.close()
            raise

    async def submit(self, command: ComputeCommand[InputT]) -> ComputeTaskSnapshot:
        request = command.to_request().model_copy(
            update={"reply_to": ComputeReplyAddress(topic=self._reply_topic, requester_id=self._requester_id)}
        )
        event = new_event(
            "compute.task.requested",
            "application",
            correlation_id=command.task_id,
            task_id=command.task_id,
            payload=request.model_dump(mode="json"),
        )
        await self._kafka_producer.publish(_task_topic(command.resource_queue), str(command.task_id), event)
        snapshot = _queued_snapshot(command)
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
        waiter = self._register(command.task_id)
        try:
            await self.submit(replace(command, wait_for_subscriber=False))
            reply = await self._wait_for_terminal_reply(waiter, command.task_id, on_progress)
            if reply.status != ComputeTaskStatus.SUCCEEDED or reply.result is None:
                raise _terminal_error(reply)
            return result_type.model_validate(await self._load_result(reply.result))
        except asyncio.CancelledError:
            await self._request_cancel(command.task_id)
            raise
        finally:
            self._waiters.pop(command.task_id, None)

    async def stream(self, command: ComputeCommand[InputT]) -> AsyncIterator[ComputeEvent]:
        waiter = self._register(command.task_id)
        try:
            await self.submit(replace(command, wait_for_subscriber=True))
            reply = await self._wait_for_reply(waiter, command.task_id)
            if reply.stream_key is None:
                if reply.status.is_terminal:
                    raise _terminal_error(reply)
                raise ComputeStreamDisconnectedError("Compute reply did not contain a Redis stream locator")
            self._waiters.pop(command.task_id, None)
            cursor = "0-0"
            while True:
                events = await self._redis_store.read(reply.stream_key, cursor)
                for raw in events:
                    cursor = raw.id
                    event = parse_compute_event(cast(JsonObject, raw.data))
                    yield event
                    if isinstance(event, ComputeCompletedEvent | ComputeFailedEvent | ComputeCancelledEvent):
                        return
        except asyncio.CancelledError:
            await self._request_cancel(command.task_id)
            raise
        finally:
            self._waiters.pop(command.task_id, None)

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

    def _register(self, task_id: UUID) -> asyncio.Queue[ComputeTaskReply]:
        if task_id in self._waiters:
            raise RuntimeError(f"Compute task already has a reply waiter: {task_id}")
        waiter: asyncio.Queue[ComputeTaskReply] = asyncio.Queue()
        self._waiters[task_id] = waiter
        return waiter

    async def _wait_for_reply(self, waiter: asyncio.Queue[ComputeTaskReply], task_id: UUID) -> ComputeTaskReply:
        try:
            return await asyncio.wait_for(waiter.get(), timeout=self._reply_wait_timeout)
        except TimeoutError as error:
            raise ComputeReplyTimeoutError(
                f"Compute task {task_id} did not receive a Kafka reply within {self._reply_wait_timeout:g} seconds"
            ) from error

    async def _wait_for_terminal_reply(
        self,
        waiter: asyncio.Queue[ComputeTaskReply],
        task_id: UUID,
        on_progress: Callable[[float, str | None], None] | None,
    ) -> ComputeTaskReply:
        last_progress: tuple[float, str | None] | None = None
        while True:
            if on_progress is None:
                reply = await waiter.get()
            else:
                try:
                    reply = await asyncio.wait_for(waiter.get(), timeout=self._progress_poll_interval)
                except TimeoutError:
                    snapshot = await self.status(task_id)
                    current_progress = (
                        (snapshot.progress, snapshot.message) if snapshot.progress is not None else None
                    )
                    if current_progress is not None and current_progress != last_progress:
                        on_progress(*current_progress)
                        last_progress = current_progress
                    continue
            if reply.status.is_terminal:
                return reply

    async def _handle_reply(self, envelope: EventEnvelope) -> None:
        if envelope.event_type != "compute.task.reply":
            return
        try:
            reply = ComputeTaskReply.model_validate(envelope.payload)
        except ValueError:
            logger.exception("Ignoring invalid compute reply event_id=%s", envelope.event_id)
            return
        if reply.requester_id != self._requester_id:
            return
        waiter = self._waiters.get(reply.task_id)
        if waiter is not None:
            waiter.put_nowait(reply)
        elif reply.result is not None:
            try:
                await self._delete_result(reply.result)
            except Exception:
                logger.exception("Failed to discard unclaimed compute result task_id=%s", reply.task_id)

    async def _load_result(self, locator: ComputeResultLocator) -> JsonObject:
        if locator.storage == "redis":
            result = await self._redis_store.get_state(locator.key)
            if result is None:
                raise ComputeStreamDisconnectedError(f"Compute Redis result is missing: {locator.key}")
            await self._redis_store.delete(locator.key)
            return cast(JsonObject, result)
        if self._file_store is None:
            raise ComputeStreamDisconnectedError("Compute result requires FileStore but none is configured")
        try:
            path = await asyncio.to_thread(self._file_store.get_file_by_key, locator.key)
            serialized = await asyncio.to_thread(path.read_text, encoding="utf-8")
            parsed = json.loads(serialized)
            if not isinstance(parsed, dict):
                raise ValueError("Compute result must be a JSON object")
            return cast(JsonObject, parsed)
        finally:
            await asyncio.to_thread(self._file_store.delete_file, locator.key)

    async def _delete_result(self, locator: ComputeResultLocator) -> None:
        if locator.storage == "redis":
            await self._redis_store.delete(locator.key)
        elif self._file_store is not None:
            await asyncio.to_thread(self._file_store.delete_file, locator.key)


class SyncKafkaWorkerClient(SyncWorkerClient):
    """Blocking request/reply compute client with a thread-owned reply consumer."""

    def __init__(
        self,
        producer: SyncKafkaEventProducer,
        redis: SyncRedisStreamStore,
        file_store: FileStore | None = None,
        *,
        requester_id: str | None = None,
        reply_consumer: SyncKafkaEventConsumer | None = None,
        reply_wait_timeout_seconds: float = DEFAULT_REPLY_WAIT_TIMEOUT_SECONDS,
        progress_poll_interval_seconds: float = DEFAULT_PROGRESS_POLL_INTERVAL_SECONDS,
    ) -> None:
        if reply_wait_timeout_seconds <= 0:
            raise ValueError("reply_wait_timeout_seconds must be positive")
        if progress_poll_interval_seconds <= 0:
            raise ValueError("progress_poll_interval_seconds must be positive")
        self._kafka_producer = producer
        self._redis_store = redis
        self._file_store = file_store
        self._requester_id = requester_id or uuid4().hex
        self._reply_topic = Topics.COMPUTE_RESULTS
        self._reply_consumer = reply_consumer
        self._reply_wait_timeout = reply_wait_timeout_seconds
        self._progress_poll_interval = progress_poll_interval_seconds
        self._waiters: dict[UUID, Queue[ComputeTaskReply]] = {}
        self._waiters_lock = Lock()
        self._ready = False

    def close(self) -> None:
        if self._reply_consumer is not None:
            self._reply_consumer.stop()
        self._ready = False

    def ready(self) -> None:
        self._redis_store.ping()
        if self._ready:
            return
        if self._reply_consumer is None:
            self._reply_consumer = SyncKafkaEventConsumer(
                [self._reply_topic],
                bootstrap_servers=self._kafka_producer.bootstrap_servers,
                group_id=f"compute-replies-{self._requester_id}",
                client_id=f"{self._kafka_producer.client_id}-replies",
                auto_offset_reset="latest",
            )
        self._reply_consumer.start(self._handle_reply)
        self._ready = True

    def submit(self, command: ComputeCommand[InputT]) -> ComputeTaskSnapshot:
        request = command.to_request().model_copy(
            update={"reply_to": ComputeReplyAddress(topic=self._reply_topic, requester_id=self._requester_id)}
        )
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
        waiter = self._register(command.task_id)
        try:
            self.submit(replace(command, wait_for_subscriber=False))
            reply = self._wait_for_terminal_reply(waiter, command.task_id, on_progress)
            if reply.status != ComputeTaskStatus.SUCCEEDED or reply.result is None:
                raise _terminal_error(reply)
            return result_type.model_validate(self._load_result(reply.result))
        finally:
            with self._waiters_lock:
                self._waiters.pop(command.task_id, None)

    def stream(self, command: ComputeCommand[InputT]) -> Iterator[ComputeEvent]:
        waiter = self._register(command.task_id)
        try:
            self.submit(replace(command, wait_for_subscriber=True))
            reply = self._wait_for_reply(waiter, command.task_id)
            if reply.stream_key is None:
                if reply.status.is_terminal:
                    raise _terminal_error(reply)
                raise ComputeStreamDisconnectedError("Compute reply did not contain a Redis stream locator")
            with self._waiters_lock:
                self._waiters.pop(command.task_id, None)
            cursor = "0-0"
            while True:
                events = self._redis_store.read(reply.stream_key, cursor)
                for raw in events:
                    cursor = raw.id
                    event = parse_compute_event(cast(JsonObject, raw.data))
                    yield event
                    if isinstance(event, ComputeCompletedEvent | ComputeFailedEvent | ComputeCancelledEvent):
                        return
        finally:
            with self._waiters_lock:
                self._waiters.pop(command.task_id, None)

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

    def _register(self, task_id: UUID) -> Queue[ComputeTaskReply]:
        with self._waiters_lock:
            if task_id in self._waiters:
                raise RuntimeError(f"Compute task already has a reply waiter: {task_id}")
            waiter: Queue[ComputeTaskReply] = Queue()
            self._waiters[task_id] = waiter
            return waiter

    def _wait_for_reply(self, waiter: Queue[ComputeTaskReply], task_id: UUID) -> ComputeTaskReply:
        try:
            return waiter.get(timeout=self._reply_wait_timeout)
        except Empty as error:
            raise ComputeReplyTimeoutError(
                f"Compute task {task_id} did not receive a Kafka reply within {self._reply_wait_timeout:g} seconds"
            ) from error

    def _wait_for_terminal_reply(
        self,
        waiter: Queue[ComputeTaskReply],
        task_id: UUID,
        on_progress: Callable[[float, str | None], None] | None,
    ) -> ComputeTaskReply:
        if on_progress is None:
            while True:
                reply = waiter.get()
                if reply.status.is_terminal:
                    return reply

        last_progress: tuple[float, str | None] | None = None
        while True:
            try:
                reply = waiter.get(timeout=self._progress_poll_interval)
            except Empty:
                snapshot = self.status(task_id)
                current_progress = (
                    (snapshot.progress, snapshot.message) if snapshot.progress is not None else None
                )
                if current_progress is not None and current_progress != last_progress:
                    on_progress(*current_progress)
                    last_progress = current_progress
                continue
            if reply.status.is_terminal:
                return reply

    def _handle_reply(self, envelope: EventEnvelope) -> None:
        if envelope.event_type != "compute.task.reply":
            return
        try:
            reply = ComputeTaskReply.model_validate(envelope.payload)
        except ValueError:
            logger.exception("Ignoring invalid compute reply event_id=%s", envelope.event_id)
            return
        if reply.requester_id != self._requester_id:
            return
        with self._waiters_lock:
            waiter = self._waiters.get(reply.task_id)
        if waiter is not None:
            waiter.put(reply)
        elif reply.result is not None:
            try:
                self._delete_result(reply.result)
            except Exception:
                logger.exception("Failed to discard unclaimed compute result task_id=%s", reply.task_id)

    def _load_result(self, locator: ComputeResultLocator) -> JsonObject:
        if locator.storage == "redis":
            result = self._redis_store.get_state(locator.key)
            if result is None:
                raise ComputeStreamDisconnectedError(f"Compute Redis result is missing: {locator.key}")
            self._redis_store.delete(locator.key)
            return cast(JsonObject, result)
        if self._file_store is None:
            raise ComputeStreamDisconnectedError("Compute result requires FileStore but none is configured")
        try:
            parsed = json.loads(self._file_store.get_file_by_key(locator.key).read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("Compute result must be a JSON object")
            return cast(JsonObject, parsed)
        finally:
            self._file_store.delete_file(locator.key)

    def _delete_result(self, locator: ComputeResultLocator) -> None:
        if locator.storage == "redis":
            self._redis_store.delete(locator.key)
        elif self._file_store is not None:
            self._file_store.delete_file(locator.key)
