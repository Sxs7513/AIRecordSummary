from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from l1_foundation.messaging import EventEnvelope, KafkaEventProducer, Topics, new_event
from l1_foundation.streaming import RedisStreamStore
from l1_foundation.worker.contracts import (
    ComputeCancelRequest,
    ComputeDeltaEvent,
    ComputeHeartbeatEvent,
    ComputeQueuedEvent,
    ComputeTaskRequest,
    ExecutionScope,
)
from l3_app.compute_worker.runtime import ComputeWorkerRuntime


def compute_state_key(task_id: object) -> str:
    return f"compute:{task_id}:state"


def compute_stream_key(task_id: object) -> str:
    return f"compute:{task_id}:events"


def execution_scope_cancel_id(scope: ExecutionScope) -> str:
    return f"execution:{scope.kind}:{scope.id}"


class KafkaComputeTaskHandler:
    """Bridge durable Kafka commands into the resource-aware compute runtime."""

    def __init__(
        self,
        runtime: ComputeWorkerRuntime,
        kafka_producer: KafkaEventProducer,
        redis_event_store: RedisStreamStore,
    ) -> None:
        self._runtime = runtime
        self._kafka_producer = kafka_producer
        self._redis_event_store = redis_event_store

    async def handle(self, envelope: EventEnvelope) -> None:
        if envelope.event_type != "compute.task.requested":
            return
        request = ComputeTaskRequest.model_validate(envelope.payload).model_copy(update={"wait_for_subscriber": True})
        await self._runtime.submit(request)
        if await self._is_cancel_requested(request):
            self._runtime.cancel(request.task_id)
        await self._publish_state(request, "queued")
        cancel_monitor = asyncio.create_task(self._monitor_cancel(request), name=f"compute-cancel-{request.task_id}")
        try:
            async for event in self._runtime.events(request.task_id):
                event_data = event.model_dump(mode="json")
                await self._redis_event_store.append(compute_stream_key(request.task_id), event.type, event_data)
                if event.type not in {"delta", "heartbeat"}:
                    snapshot = self._runtime.status(request.task_id).model_dump(mode="json")
                    await self._redis_event_store.set_state(compute_state_key(request.task_id), snapshot)
                if not isinstance(event, ComputeDeltaEvent | ComputeHeartbeatEvent):
                    await self._kafka_producer.publish(
                        Topics.COMPUTE_RESULTS,
                        str(request.task_id),
                        new_event(
                            f"compute.task.{event.type}",
                            "compute-worker",
                            correlation_id=envelope.correlation_id,
                            causation_id=envelope.event_id,
                            task_id=request.task_id,
                            payload=event_data,
                        ),
                    )
                if event.type in {"completed", "failed", "cancelled"}:
                    await self._redis_event_store.finish(compute_state_key(request.task_id), compute_stream_key(request.task_id))
                    return
        finally:
            cancel_monitor.cancel()
            await asyncio.gather(cancel_monitor, return_exceptions=True)

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
        await self._redis_event_store.set_state(compute_state_key(request.task_id), snapshot)
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
