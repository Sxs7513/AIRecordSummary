from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

from l1_foundation.messaging import EventEnvelope, KafkaEventProducer, Topics, new_event
from l1_foundation.observability.contracts import ModelInvocationRecord, RagExecutionSpanRecord

logger = logging.getLogger("observability")


class EventPublisher(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def publish(self, topic: str, key: str, event: EventEnvelope) -> None: ...


class _DisabledPublisher:
    async def start(self) -> None:
        return

    async def stop(self) -> None:
        return

    async def publish(self, topic: str, key: str, event: EventEnvelope) -> None:
        return


@dataclass(frozen=True, slots=True)
class _QueuedEvent:
    topic: str
    key: str
    event: EventEnvelope


class ObservabilityClient:
    """Publish RAG and model telemetry to Kafka without HTTP ingestion coupling."""

    def __init__(
        self,
        *,
        bootstrap_servers: str | None = None,
        client_id: str = "production-api-observability",
        request_timeout_ms: int = 30_000,
        enabled: bool = True,
        producer: EventPublisher | None = None,
    ) -> None:
        if producer is None and bootstrap_servers is None:
            if enabled:
                raise ValueError("bootstrap_servers is required when observability is enabled")
            bootstrap_servers = "127.0.0.1:9092"
        self._enabled = enabled
        self._producer = producer or (
            KafkaEventProducer(bootstrap_servers or "127.0.0.1:9092", client_id, request_timeout_ms) if enabled else _DisabledPublisher()
        )
        self._queue: asyncio.Queue[_QueuedEvent] = asyncio.Queue()
        self._sender_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._enabled and self._sender_task is None:
            await self._producer.start()
            self._sender_task = asyncio.create_task(self._send_loop(), name="observability-kafka-producer")

    async def close(self) -> None:
        task = self._sender_task
        if task is not None:
            await self._queue.join()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self._sender_task = None
            await self._producer.stop()

    def publish_span(self, record: RagExecutionSpanRecord) -> None:
        self._enqueue(
            Topics.RAG_EXECUTION_EVENTS,
            str(record.generation_run_id),
            new_event(
                "rag.span.recorded",
                "production-api",
                correlation_id=record.generation_run_id,
                workspace_id=record.workspace_id,
                generation_id=record.generation_run_id,
                trace_id=record.id,
                payload=record.model_dump(mode="json"),
            ),
        )

    def publish_model_invocation(self, record: ModelInvocationRecord) -> None:
        self._enqueue(
            Topics.MODEL_INVOCATION_EVENTS,
            str(record.generation_run_id),
            new_event(
                "model.invocation.recorded",
                "production-api",
                correlation_id=record.generation_run_id,
                workspace_id=record.workspace_id,
                generation_id=record.generation_run_id,
                trace_id=record.span_id,
                payload=record.model_dump(mode="json"),
            ),
        )

    def _enqueue(self, topic: str, key: str, event: EventEnvelope) -> None:
        if self._enabled:
            self._queue.put_nowait(_QueuedEvent(topic, key, event))

    async def _send_loop(self) -> None:
        while True:
            queued = await self._queue.get()
            try:
                await self._producer.publish(queued.topic, queued.key, queued.event)
            except Exception:
                logger.exception("observability Kafka delivery failed event_id=%s topic=%s", queued.event.event_id, queued.topic)
                await asyncio.sleep(0.5)
                self._queue.put_nowait(queued)
            finally:
                self._queue.task_done()
