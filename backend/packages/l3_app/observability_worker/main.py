from __future__ import annotations

import logging

from repository import ObservabilityRepository

from l1_foundation.infrastructure.db.session import create_database_engine
from l1_foundation.messaging import KafkaEventConsumer, KafkaEventProducer, Topics
from l1_foundation.settings import get_settings
from l3_app.observability_worker.worker import ObservabilityEventProjector


async def run() -> None:
    settings = get_settings()
    engine = create_database_engine(settings)
    repository = ObservabilityRepository(engine)
    projector = ObservabilityEventProjector(repository)
    consumer = KafkaEventConsumer(
        [Topics.RAG_EXECUTION_EVENTS, Topics.MODEL_INVOCATION_EVENTS, Topics.OBSERVABILITY_RETRY],
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id="observability-postgres-projector-v1",
        client_id=f"{settings.kafka_client_id}-observability-worker",
        max_poll_interval_ms=settings.kafka_consumer_max_poll_interval_ms,
    )
    producer = KafkaEventProducer(
        settings.kafka_bootstrap_servers,
        f"{settings.kafka_client_id}-observability-retry",
        settings.kafka_request_timeout_ms,
    )

    await producer.start()
    await consumer.start()
    try:
        await consumer.run(
            projector.handle,
            producer=producer,
            retry_topic=Topics.OBSERVABILITY_RETRY,
            dlq_topic=Topics.OBSERVABILITY_DLQ,
        )
    finally:
        await consumer.stop()
        await producer.stop()
        engine.dispose()


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
