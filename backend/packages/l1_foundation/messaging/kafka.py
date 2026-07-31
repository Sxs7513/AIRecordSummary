from __future__ import annotations

import asyncio

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from concurrent.futures import Future
from threading import Event, Thread
from typing import cast

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.structs import ConsumerRecord

from l1_foundation.messaging.contracts import EventEnvelope

EventHandler = Callable[[EventEnvelope], Awaitable[None]]


def _encode_event(event: EventEnvelope) -> bytes:
    return event.model_dump_json().encode("utf-8")


def _decode_event(value: bytes) -> EventEnvelope:
    return EventEnvelope.model_validate_json(value)


def _encode_key(value: str) -> bytes:
    return value.encode("utf-8")


def _decode_key(value: bytes) -> str:
    return value.decode("utf-8")


class KafkaEventProducer:
    """Idempotent, all-ack Kafka producer for typed event envelopes."""

    def __init__(self, bootstrap_servers: str, client_id: str, request_timeout_ms: int = 30_000) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap_servers,
            client_id=client_id,
            acks="all",
            enable_idempotence=True,
            request_timeout_ms=request_timeout_ms,
            value_serializer=_encode_event,
            key_serializer=_encode_key,
        )
        self._started = False

    async def start(self) -> None:
        if not self._started:
            try:
                await self._producer.start()
            except Exception:
                await self._producer.stop()
                raise
            else:
                self._started = True

    async def stop(self) -> None:
        if self._started:
            await self._producer.stop()
            self._started = False

    async def publish(self, topic: str, key: str, event: EventEnvelope) -> None:
        if not self._started:
            raise RuntimeError("Kafka producer is not started")
        await self._producer.send_and_wait(topic, key=key, value=event)


class SyncKafkaEventProducer:
    """Blocking facade over the async Kafka producer for thread-owned stage code."""

    def __init__(self, bootstrap_servers: str, client_id: str, request_timeout_ms: int = 30_000) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._client_id = client_id
        self._request_timeout_ms = request_timeout_ms
        self._loop: asyncio.AbstractEventLoop | None = None
        self._producer: KafkaEventProducer | None = None
        self._thread: Thread | None = None
        self._ready = Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(target=self._run_loop, name=f"{self._client_id}-kafka", daemon=True)
        self._thread.start()
        self._ready.wait()
        try:
            self._submit(self._start()).result()
        except BaseException:
            self._shutdown_loop()
            raise

    def stop(self) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            self._submit(self._stop()).result()
        finally:
            self._shutdown_loop()

    def _shutdown_loop(self) -> None:
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        self._loop = None
        self._producer = None
        self._ready.clear()

    def publish(self, topic: str, key: str, event: EventEnvelope) -> None:
        self._submit(self._publish(topic, key, event)).result()

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        loop.run_forever()
        loop.close()

    async def _start(self) -> None:
        if self._producer is None:
            # aiokafka binds itself to get_running_loop() during construction.
            # Build it inside the coroutine executed by this thread's active loop.
            self._producer = KafkaEventProducer(self._bootstrap_servers, self._client_id, self._request_timeout_ms)
        producer = self._producer
        await producer.start()

    async def _stop(self) -> None:
        if self._producer is not None:
            await self._producer.stop()

    async def _publish(self, topic: str, key: str, event: EventEnvelope) -> None:
        producer = self._producer
        if producer is None:
            raise RuntimeError("Kafka producer is not started")
        await producer.publish(topic, key, event)

    def _submit(self, coroutine: Coroutine[object, object, None]) -> Future[None]:
        loop = self._loop
        if loop is None:
            raise RuntimeError("Kafka producer is not started")
        return asyncio.run_coroutine_threadsafe(coroutine, loop)


class KafkaTopicAdmin:
    """Create the fixed topic set explicitly; broker auto-create stays disabled."""

    def __init__(self, bootstrap_servers: str, client_id: str) -> None:
        self._admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap_servers, client_id=client_id)

    async def ensure_topics(self, topics: Sequence[str], compacted_topics: Sequence[str], partitions: int = 1) -> None:
        await self._admin.start()
        try:
            compacted = set(compacted_topics)
            existing = await self._admin.list_topics()
            definitions = [
                NewTopic(
                    name=topic,
                    num_partitions=partitions,
                    replication_factor=1,
                    topic_configs={"cleanup.policy": "compact" if topic in compacted else "delete"},
                )
                for topic in topics
                if topic not in existing
            ]
            if definitions:
                await self._admin.create_topics(definitions)
        finally:
            await self._admin.close()


class KafkaEventConsumer:
    """Manual-commit consumer that acknowledges only successful handlers."""

    def __init__(
        self,
        topics: Sequence[str],
        *,
        bootstrap_servers: str,
        group_id: str,
        client_id: str,
        max_poll_interval_ms: int = 900_000,
    ) -> None:
        self._consumer = AIOKafkaConsumer(
            *topics,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            client_id=client_id,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            max_poll_interval_ms=max_poll_interval_ms,
            value_deserializer=_decode_event,
            key_deserializer=_decode_key,
        )
        self._started = False

    async def start(self) -> None:
        if not self._started:
            await self._consumer.start()
            self._started = True

    async def stop(self) -> None:
        if self._started:
            await self._consumer.stop()
            self._started = False

    async def run(
        self,
        handler: EventHandler,
        *,
        producer: KafkaEventProducer | None = None,
        retry_topic: str | None = None,
        dlq_topic: str | None = None,
        max_attempts: int = 3,
    ) -> None:
        if not self._started:
            raise RuntimeError("Kafka consumer is not started")
        async for record in self._consumer:
            typed_record = cast(ConsumerRecord[str, EventEnvelope], record)
            event = typed_record.value
            if event is not None:
                try:
                    await handler(event)
                except Exception as error:
                    if producer is None or retry_topic is None or dlq_topic is None:
                        raise
                    attempt = event.attempt + 1
                    target = retry_topic if attempt < max_attempts else dlq_topic
                    failed = event.model_copy(update={"attempt": attempt, "last_error": (str(error) or type(error).__name__)[:2000]})
                    await producer.publish(target, self._event_key(event), failed)
            await self._consumer.commit()

    @staticmethod
    def _event_key(event: EventEnvelope) -> str:
        identity = event.task_id or event.processing_id or event.generation_id or event.correlation_id
        return str(identity)


def event_to_log_fields(event: EventEnvelope) -> dict[str, str | int | None]:
    """Return stable structured fields without serializing message payloads."""
    return {
        "event_id": str(event.event_id),
        "event_type": event.event_type,
        "schema_version": event.schema_version,
        "correlation_id": str(event.correlation_id),
        "task_id": str(event.task_id) if event.task_id else None,
        "processing_id": str(event.processing_id) if event.processing_id else None,
        "generation_id": str(event.generation_id) if event.generation_id else None,
    }
