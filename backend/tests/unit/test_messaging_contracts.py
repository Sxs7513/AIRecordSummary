from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from pydantic import ValidationError

from l1_foundation.messaging import EventEnvelope, new_event
from l1_foundation.messaging import kafka as kafka_module


def test_new_event_uses_event_id_as_default_correlation_id() -> None:
    event = new_event("processing.requested", "production-api", processing_id=uuid4(), payload={"pipeline": "recording_processing"})

    assert event.correlation_id == event.event_id
    assert EventEnvelope.model_validate_json(event.model_dump_json()) == event


def test_event_requires_non_empty_type_and_producer() -> None:
    with pytest.raises(ValidationError):
        EventEnvelope(event_type="", producer="", correlation_id=uuid4())


def test_sync_kafka_producer_is_constructed_inside_its_running_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    lifecycle: list[str] = []

    class FakeKafkaEventProducer:
        def __init__(self, _bootstrap_servers: str, _client_id: str, _request_timeout_ms: int) -> None:
            asyncio.get_running_loop()
            lifecycle.append("constructed")

        async def start(self) -> None:
            lifecycle.append("started")

        async def stop(self) -> None:
            lifecycle.append("stopped")

        async def publish(self, _topic: str, _key: str, _event: EventEnvelope) -> None:
            lifecycle.append("published")

    monkeypatch.setattr(kafka_module, "KafkaEventProducer", FakeKafkaEventProducer)
    producer = kafka_module.SyncKafkaEventProducer("localhost:9092", "test")

    producer.start()
    producer.publish("test.events", "key", new_event("test.recorded", "test"))
    producer.stop()

    assert lifecycle == ["constructed", "started", "published", "stopped"]
