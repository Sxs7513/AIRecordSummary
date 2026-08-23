from __future__ import annotations

import asyncio
import logging

from l1_foundation.infrastructure.db.session import create_database_engine
from l1_foundation.messaging import KafkaEventConsumer, KafkaEventProducer, SyncKafkaEventProducer, Topics
from l1_foundation.observability import ObservabilityClient
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l1_foundation.settings import get_settings
from l1_foundation.streaming import RedisStreamStore, SyncRedisStreamStore
from l1_foundation.worker import KafkaWorkerClient, SyncKafkaWorkerClient
from l2_core.audio_processing.registry import build_recording_summary_stage, build_summary_embedding_indexer
from l2_core.audio_processing.stages.summary.regeneration import RecordingSummaryRegenerationService
from l2_core.conversations.history_store import ConversationHistoryStore
from l2_core.conversations.service import ConversationService
from l2_core.generation.redis_runtime import GenerationRedisRuntime
from l2_core.generation.service import GenerationService
from l2_core.generation.store import GenerationEventStore
from l2_core.rag.service import RagService
from l3_app.generation_worker.worker import GenerationCancelHandler, GenerationCommandHandler, GenerationResultProjector


async def run() -> None:
    settings = get_settings()
    engine = create_database_engine(settings)
    sync_redis_store = SyncRedisStreamStore.from_url(
        settings.redis_url,
        maxlen=settings.redis_stream_maxlen,
        terminal_ttl_seconds=settings.redis_terminal_ttl_seconds,
    )
    conversation_history_store = ConversationHistoryStore.from_url(settings.redis_url)
    async_redis_store = RedisStreamStore.from_url(
        settings.redis_url,
        maxlen=settings.redis_stream_maxlen,
        terminal_ttl_seconds=settings.redis_terminal_ttl_seconds,
    )
    generation_service = GenerationService(engine, GenerationRedisRuntime(sync_redis_store))
    compute_producer = KafkaEventProducer(
        settings.kafka_bootstrap_servers,
        f"{settings.kafka_client_id}-generation-compute",
        settings.kafka_request_timeout_ms,
    )
    sync_compute_producer = SyncKafkaEventProducer(
        settings.kafka_bootstrap_servers,
        f"{settings.kafka_client_id}-generation-sync-compute",
        settings.kafka_request_timeout_ms,
    )
    await compute_producer.start()
    sync_compute_producer.start()
    worker_client = KafkaWorkerClient(
        compute_producer,
        async_redis_store,
        poll_interval_seconds=settings.compute_worker_poll_interval_seconds,
    )
    sync_worker_client = SyncKafkaWorkerClient(
        sync_compute_producer,
        sync_redis_store,
        poll_interval_seconds=settings.compute_worker_poll_interval_seconds,
    )
    observability = ObservabilityClient(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        client_id=f"{settings.kafka_client_id}-generation-observability",
        request_timeout_ms=settings.kafka_request_timeout_ms,
        enabled=settings.observability_enabled,
    )
    consumer = KafkaEventConsumer(
        [Topics.GENERATION_COMMANDS, Topics.GENERATION_RETRY],
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id="generation-worker-v1",
        client_id=f"{settings.kafka_client_id}-generation-worker",
        max_poll_interval_ms=settings.kafka_consumer_max_poll_interval_ms,
    )
    projector_consumer = KafkaEventConsumer(
        [Topics.GENERATION_EVENTS, Topics.GENERATION_PROJECTION_RETRY],
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id="generation-postgres-projector-v1",
        client_id=f"{settings.kafka_client_id}-generation-projector",
        max_poll_interval_ms=settings.kafka_consumer_max_poll_interval_ms,
    )
    cancel_consumer = KafkaEventConsumer(
        [Topics.GENERATION_CANCEL, Topics.GENERATION_CANCEL_RETRY],
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id="generation-cancel-projector-v1",
        client_id=f"{settings.kafka_client_id}-generation-cancel",
        max_poll_interval_ms=settings.kafka_consumer_max_poll_interval_ms,
    )

    await worker_client.ready()
    await observability.start()
    rag_service = RagService(engine, settings, worker_client, sync_worker_client, observability, sync_redis_store)
    summary_stage = build_recording_summary_stage(settings, ArtifactStore(settings.resolved_local_storage_root), sync_worker_client)
    summary_service = RecordingSummaryRegenerationService(
        engine,
        generation_service,
        summary_stage,
        summary_embedding_indexer=build_summary_embedding_indexer(settings, sync_worker_client),
    )
    conversation_service = ConversationService(engine, generation_service, conversation_history_store)
    handler = GenerationCommandHandler(
        rag_service,
        generation_service,
        conversation_service,
        compute_producer,
        summary_service,
    )
    await consumer.start()
    await projector_consumer.start()
    await cancel_consumer.start()
    command_task = asyncio.create_task(
        consumer.run(
            handler.handle,
            producer=compute_producer,
            retry_topic=Topics.GENERATION_RETRY,
            dlq_topic=Topics.GENERATION_DLQ,
        ),
        name="generation-command-consumer",
    )
    projector = GenerationResultProjector(GenerationEventStore(engine))
    projector_task = asyncio.create_task(
        projector_consumer.run(
            projector.handle,
            producer=compute_producer,
            retry_topic=Topics.GENERATION_PROJECTION_RETRY,
            dlq_topic=Topics.GENERATION_PROJECTION_DLQ,
        ),
        name="generation-result-projector",
    )
    cancel_handler = GenerationCancelHandler(generation_service, compute_producer, conversation_service)
    cancel_task = asyncio.create_task(
        cancel_consumer.run(
            cancel_handler.handle,
            producer=compute_producer,
            retry_topic=Topics.GENERATION_CANCEL_RETRY,
            dlq_topic=Topics.GENERATION_CANCEL_DLQ,
        ),
        name="generation-cancel-consumer",
    )
    try:
        await asyncio.gather(command_task, projector_task, cancel_task)
    finally:
        command_task.cancel()
        projector_task.cancel()
        cancel_task.cancel()
        await asyncio.gather(command_task, projector_task, cancel_task, return_exceptions=True)
        await consumer.stop()
        await projector_consumer.stop()
        await cancel_consumer.stop()
        await observability.close()
        await worker_client.close()
        sync_worker_client.close()
        await compute_producer.stop()
        sync_compute_producer.stop()
        await async_redis_store.close()
        sync_redis_store.close()
        conversation_history_store.close()
        engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
