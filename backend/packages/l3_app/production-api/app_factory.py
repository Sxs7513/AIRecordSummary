from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from production_router import router as production_router

from l1_foundation.infrastructure.db.session import create_database_engine
from l1_foundation.infrastructure.storage.local import LocalStorage
from l1_foundation.messaging import KafkaEventProducer, OutboxRepository, SyncKafkaEventProducer
from l1_foundation.observability import ObservabilityClient
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l1_foundation.settings import Settings, get_settings
from l1_foundation.streaming import RedisStreamStore, SyncRedisStreamStore
from l1_foundation.worker import KafkaWorkerClient, SyncKafkaWorkerClient, SyncWorkerClient, WorkerClient
from l2_core.application.processing_queue import ProcessingCommandPublisher
from l2_core.audio_processing.definition import build_recording_processing
from l2_core.audio_processing.registry import build_recording_summary_stage
from l2_core.audio_processing.stages.summary.regeneration import RecordingSummaryRegenerationService
from l2_core.conversations.history_store import ConversationHistoryStore
from l2_core.generation.redis_runtime import GenerationRedisRuntime
from l2_core.generation.service import GenerationService
from l2_core.rag.queue import GenerationCommandPublisher
from l2_core.rag.service import RagService

logger = logging.getLogger(__name__)


def _configure_rag_logger() -> None:
    """Expose diagnostic RAG milestones without enabling noisy global request logs."""
    rag_logger = logging.getLogger("rag")
    rag_logger.setLevel(logging.INFO)
    rag_logger.propagate = False
    if rag_logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    rag_logger.addHandler(handler)


def _configure_audio_processing_logger() -> None:
    """Expose audio pipeline milestones without enabling noisy global request logs."""
    audio_logger = logging.getLogger("audio_processing")
    audio_logger.setLevel(logging.INFO)
    audio_logger.propagate = False
    if audio_logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    audio_logger.addHandler(handler)


def _configure_llm_logger() -> None:
    """Expose LLM lifecycle and online-request logs at a single INFO channel."""
    llm_logger = logging.getLogger("llm")
    llm_logger.setLevel(logging.INFO)
    llm_logger.propagate = False
    if llm_logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    llm_logger.addHandler(handler)


def _configure_worker_logger() -> None:
    """Expose internal compute-worker lifecycle logs at INFO."""
    worker_logger = logging.getLogger("worker")
    worker_logger.setLevel(logging.INFO)
    worker_logger.propagate = False
    if worker_logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    worker_logger.addHandler(handler)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Initialize HTTP infrastructure and its queue workers as one application service."""
    storage = LocalStorage(app.state.settings.resolved_local_storage_root)
    storage.initialize()
    app.state.storage = storage
    app.state.database_engine = create_database_engine(app.state.settings)
    app.state.async_redis_store = RedisStreamStore.from_url(
        app.state.settings.redis_url,
        maxlen=app.state.settings.redis_stream_maxlen,
        terminal_ttl_seconds=app.state.settings.redis_terminal_ttl_seconds,
    )
    app.state.sync_redis_store = SyncRedisStreamStore.from_url(
        app.state.settings.redis_url,
        maxlen=app.state.settings.redis_stream_maxlen,
        terminal_ttl_seconds=app.state.settings.redis_terminal_ttl_seconds,
    )
    app.state.conversation_history_store = ConversationHistoryStore.from_url(app.state.settings.redis_url)
    app.state.generation_redis_runtime = GenerationRedisRuntime(app.state.sync_redis_store)
    app.state.generation_service = GenerationService(app.state.database_engine, app.state.generation_redis_runtime)
    app.state.generation_command_producer = app.state.injected_kafka_producer or KafkaEventProducer(
        app.state.settings.kafka_bootstrap_servers,
        f"{app.state.settings.kafka_client_id}-production-api-generation",
        app.state.settings.kafka_request_timeout_ms,
    )
    try:
        await app.state.generation_command_producer.start()
    except Exception:
        # PostgreSQL-backed command endpoints remain available; direct Kafka-only
        # endpoints will fail until this process is restarted after Kafka recovers.
        logger.warning("direct Kafka producer unavailable; outbox-backed commands remain available", exc_info=True)
    app.state.outbox_repository = OutboxRepository(app.state.database_engine)
    app.state.generation_command_publisher = GenerationCommandPublisher(app.state.generation_command_producer, app.state.outbox_repository)
    app.state.processing_command_publisher = ProcessingCommandPublisher(app.state.generation_command_producer, app.state.outbox_repository)
    artifact_store = ArtifactStore(app.state.settings.resolved_local_storage_root)
    injected_worker_client = app.state.injected_worker_client
    app.state.worker_client = injected_worker_client or KafkaWorkerClient(
        app.state.generation_command_producer,
        app.state.async_redis_store,
        poll_interval_seconds=app.state.settings.compute_worker_poll_interval_seconds,
    )
    await app.state.worker_client.ready()
    injected_sync_worker_client = app.state.injected_sync_worker_client
    sync_compute_producer: SyncKafkaEventProducer | None = None
    if injected_sync_worker_client is None:
        sync_compute_producer = SyncKafkaEventProducer(
            app.state.settings.kafka_bootstrap_servers,
            f"{app.state.settings.kafka_client_id}-production-api-sync-compute",
            app.state.settings.kafka_request_timeout_ms,
        )
        try:
            sync_compute_producer.start()
        except Exception:
            logger.warning("sync compute producer unavailable; direct compute requests are degraded", exc_info=True)
        app.state.sync_worker_client = SyncKafkaWorkerClient(
            sync_compute_producer,
            app.state.sync_redis_store,
            poll_interval_seconds=app.state.settings.compute_worker_poll_interval_seconds,
        )
    else:
        app.state.sync_worker_client = injected_sync_worker_client
    if injected_sync_worker_client is None:
        app.state.sync_worker_client.ready()
    summary_stage = build_recording_summary_stage(
        app.state.settings,
        artifact_store,
        app.state.sync_worker_client,
        app.state.generation_service,
    )
    app.state.observability_client = ObservabilityClient(
        bootstrap_servers=app.state.settings.kafka_bootstrap_servers,
        client_id=f"{app.state.settings.kafka_client_id}-observability",
        request_timeout_ms=app.state.settings.kafka_request_timeout_ms,
        enabled=app.state.settings.observability_enabled,
    )
    try:
        await app.state.observability_client.start()
    except Exception:
        logger.warning("observability Kafka producer unavailable; telemetry is degraded", exc_info=True)
    app.state.rag_service = RagService(
        app.state.database_engine,
        app.state.settings,
        app.state.worker_client,
        app.state.sync_worker_client,
        app.state.observability_client,
        app.state.sync_redis_store,
    )
    app.state.recording_summary_regeneration_service = RecordingSummaryRegenerationService(
        app.state.database_engine, app.state.generation_service, summary_stage, app.state.generation_command_publisher
    )
    try:
        yield
    finally:
        await app.state.recording_summary_regeneration_service.shutdown()
        await app.state.observability_client.close()
        await app.state.generation_command_producer.stop()
        if injected_worker_client is None:
            await app.state.worker_client.close()
        if injected_sync_worker_client is None:
            app.state.sync_worker_client.close()
            if sync_compute_producer is not None:
                sync_compute_producer.stop()
        await app.state.async_redis_store.close()
        app.state.sync_redis_store.close()
        app.state.conversation_history_store.close()
        app.state.database_engine.dispose()


def create_app(
    settings: Settings | None = None,
    *,
    router: APIRouter | None = None,
    worker_client: WorkerClient | None = None,
    sync_worker_client: SyncWorkerClient | None = None,
    kafka_producer: KafkaEventProducer | None = None,
) -> FastAPI:
    """Build the HTTP application with no import-time side effects."""
    _configure_rag_logger()
    _configure_audio_processing_logger()
    _configure_llm_logger()
    _configure_worker_logger()
    configured_settings = settings or get_settings()
    app = FastAPI(title=configured_settings.app_name, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Accept", "Authorization", "Content-Type", "Last-Event-ID"],
    )
    app.state.settings = configured_settings
    app.state.injected_worker_client = worker_client
    app.state.injected_sync_worker_client = sync_worker_client
    app.state.injected_kafka_producer = kafka_producer
    app.state.recording_processing_definition = build_recording_processing(configured_settings.asr_provider)
    app.include_router(router or production_router, prefix=configured_settings.api_prefix)
    return app
