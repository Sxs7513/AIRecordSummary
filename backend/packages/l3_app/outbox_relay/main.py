from __future__ import annotations

import asyncio
import logging
import socket
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from l1_foundation.infrastructure.db.session import create_database_engine
from l1_foundation.messaging import KafkaEventProducer, OutboxRepository
from l1_foundation.settings import get_settings
from l1_foundation.streaming import SyncRedisStreamStore
from l2_core.generation.redis_runtime import GenerationRedisRuntime
from l3_app.outbox_relay.generation_state import GenerationStateProjector

logger = logging.getLogger("outbox_relay")


async def run() -> None:
    settings = get_settings()
    engine = create_database_engine(settings)
    repository = OutboxRepository(engine)
    redis_store = SyncRedisStreamStore.from_url(
        settings.redis_url,
        maxlen=settings.redis_stream_maxlen,
        terminal_ttl_seconds=settings.redis_terminal_ttl_seconds,
    )
    generation_state_projector = GenerationStateProjector(GenerationRedisRuntime(redis_store))
    producer = KafkaEventProducer(
        settings.kafka_bootstrap_servers,
        f"{settings.kafka_client_id}-outbox-relay",
        settings.kafka_request_timeout_ms,
    )
    relay_id = f"{socket.gethostname()}-{uuid4()}"
    last_metrics_at = 0.0
    last_cleanup_at = 0.0
    await producer.start()
    logger.info("outbox relay started relay_id=%s", relay_id)
    try:
        while True:
            messages = await asyncio.to_thread(
                repository.claim,
                relay_id,
                batch_size=settings.outbox_relay_batch_size,
                lease_seconds=settings.outbox_relay_lease_seconds,
            )
            for message in messages:
                try:
                    if message.channel == "generation-state":
                        await asyncio.to_thread(generation_state_projector.handle, message.event)
                    else:
                        await producer.publish(message.topic, message.partition_key, message.event)
                except Exception as error:
                    await asyncio.to_thread(
                        repository.mark_failed,
                        message.event_id,
                        relay_id,
                        str(error) or type(error).__name__,
                        max_attempts=settings.outbox_relay_max_attempts,
                    )
                    logger.warning(
                        "outbox publish failed event_id=%s channel=%s attempt=%d",
                        message.event_id,
                        message.channel,
                        message.attempt_count + 1,
                        exc_info=True,
                    )
                else:
                    await asyncio.to_thread(repository.mark_published, message.event_id, relay_id)

            now = asyncio.get_running_loop().time()
            if now - last_metrics_at >= settings.outbox_relay_metrics_seconds:
                metrics = await asyncio.to_thread(repository.metrics)
                logger.info("outbox metrics channels=%s", metrics)
                last_metrics_at = now
            if now - last_cleanup_at >= 86_400:
                cutoff = datetime.now(UTC) - timedelta(days=settings.outbox_retention_days)
                deleted = await asyncio.to_thread(repository.delete_published_before, cutoff)
                logger.info("outbox retention cleanup deleted=%d cutoff=%s", deleted, cutoff.isoformat())
                last_cleanup_at = now
            if not messages:
                await asyncio.sleep(settings.outbox_relay_poll_seconds)
    finally:
        await producer.stop()
        redis_store.close()
        engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
