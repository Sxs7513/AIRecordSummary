from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

from l1_foundation.messaging import OutboxRepository, Topics, new_event


def test_enqueue_persists_the_complete_envelope_and_kafka_destination() -> None:
    connection = MagicMock()
    repository = OutboxRepository()
    aggregate_id = uuid4()
    event = new_event(
        "processing.requested",
        "production-api",
        correlation_id=aggregate_id,
        processing_id=aggregate_id,
        payload={"processing_id": str(aggregate_id)},
    )

    repository.enqueue(
        connection,
        channel="processing-command",
        topic=Topics.PROCESSING_COMMANDS,
        partition_key=str(aggregate_id),
        aggregate_type="processing",
        aggregate_id=aggregate_id,
        event=event,
    )

    statement, parameters = connection.execute.call_args.args
    assert "insert into integration_outbox" in str(statement)
    assert parameters["event_id"] == event.event_id
    assert parameters["topic"] == Topics.PROCESSING_COMMANDS
    assert parameters["partition_key"] == str(aggregate_id)
    assert str(event.event_id) in parameters["payload"]


def test_outbox_schema_contains_pending_ordering() -> None:
    from scripts.initialize_database import BASE_SCHEMA_PATH

    schema = BASE_SCHEMA_PATH.read_text(encoding="utf-8")
    assert "create table if not exists integration_outbox" in schema
    assert "'generation-state'" in schema
    assert "integration_outbox_pending_idx" in schema
