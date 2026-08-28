from __future__ import annotations

import asyncio
import logging

from l1_foundation.infrastructure.db.session import create_database_engine
from l1_foundation.infrastructure.storage.local import LocalStorage
from l1_foundation.messaging import KafkaEventConsumer, KafkaEventProducer, SyncKafkaEventProducer, Topics
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l1_foundation.settings import get_settings
from l1_foundation.streaming import RedisStreamStore, SyncRedisStreamStore
from l1_foundation.worker import KafkaWorkerClient, SyncKafkaWorkerClient
from l2_core.audio_processing.definition import build_recording_processing
from l2_core.audio_processing.hooks import RecordingProcessingHooks
from l2_core.audio_processing.registry import build_recording_stage_registry, build_recording_summary_stage
from l3_app.processing_worker.worker import ProcessingCancelHandler, ProcessingCommandHandler


async def run() -> None:
    settings = get_settings()
    engine = create_database_engine(settings)
    storage = LocalStorage(settings.resolved_local_storage_root)
    storage.initialize()
    artifact_store = ArtifactStore(storage)
    redis = SyncRedisStreamStore.from_url(
        settings.redis_url,
        maxlen=settings.redis_stream_maxlen,
        terminal_ttl_seconds=settings.redis_terminal_ttl_seconds,
    )
    async_redis = RedisStreamStore.from_url(
        settings.redis_url,
        maxlen=settings.redis_stream_maxlen,
        terminal_ttl_seconds=settings.redis_terminal_ttl_seconds,
    )
    producer = KafkaEventProducer(
        settings.kafka_bootstrap_servers,
        f"{settings.kafka_client_id}-processing-worker",
        settings.kafka_request_timeout_ms,
    )
    sync_producer = SyncKafkaEventProducer(
        settings.kafka_bootstrap_servers,
        f"{settings.kafka_client_id}-processing-sync-compute",
        settings.kafka_request_timeout_ms,
    )
    await producer.start()
    sync_producer.start()
    async_compute = KafkaWorkerClient(producer, async_redis, poll_interval_seconds=settings.compute_worker_poll_interval_seconds)
    sync_compute = SyncKafkaWorkerClient(sync_producer, redis, poll_interval_seconds=settings.compute_worker_poll_interval_seconds)
    summary_stage = build_recording_summary_stage(settings, artifact_store, sync_compute)
    registry = build_recording_stage_registry(
        settings,
        artifact_store,
        sync_compute,
        async_compute,
        summary_stage=summary_stage,
        engine=engine,
    )
    definition = build_recording_processing(settings.asr_provider)
    handler = ProcessingCommandHandler(definition, registry, artifact_store, redis, producer, RecordingProcessingHooks(engine))
    consumer = KafkaEventConsumer(
        [Topics.PROCESSING_COMMANDS, Topics.PROCESSING_RETRY],
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id="processing-worker-v1",
        client_id=f"{settings.kafka_client_id}-processing-worker",
        max_poll_interval_ms=settings.processing_consumer_max_poll_interval_ms,
    )
    cancel_consumer = KafkaEventConsumer(
        [Topics.PROCESSING_CANCEL, Topics.PROCESSING_CANCEL_RETRY],
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id="processing-cancel-projector-v1",
        client_id=f"{settings.kafka_client_id}-processing-cancel",
        max_poll_interval_ms=settings.kafka_consumer_max_poll_interval_ms,
    )
    await consumer.start()
    await cancel_consumer.start()
    command_task = asyncio.create_task(
        consumer.run(
            handler.handle,
            producer=producer,
            retry_topic=Topics.PROCESSING_RETRY,
            dlq_topic=Topics.PROCESSING_DLQ,
        ),
        name="processing-command-consumer",
    )
    cancel_handler = ProcessingCancelHandler(redis)
    cancel_task = asyncio.create_task(
        cancel_consumer.run(
            cancel_handler.handle,
            producer=producer,
            retry_topic=Topics.PROCESSING_CANCEL_RETRY,
            dlq_topic=Topics.PROCESSING_CANCEL_DLQ,
        ),
        name="processing-cancel-consumer",
    )
    try:
        await asyncio.gather(command_task, cancel_task)
    finally:
        command_task.cancel()
        cancel_task.cancel()
        await asyncio.gather(command_task, cancel_task, return_exceptions=True)
        await consumer.stop()
        await cancel_consumer.stop()
        await producer.stop()
        sync_producer.stop()
        await async_redis.close()
        redis.close()
        engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
