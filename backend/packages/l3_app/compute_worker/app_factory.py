from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from l1_foundation.infrastructure.storage.local import LocalStorage
from l1_foundation.messaging import KafkaEventConsumer, KafkaEventProducer, Topics
from l1_foundation.settings import Settings, get_settings
from l1_foundation.streaming import RedisStreamStore
from l3_app.compute_worker.executor import ComputeExecutionPool
from l3_app.compute_worker.kafka_runtime import KafkaComputeCancelHandler, KafkaComputeTaskHandler
from l3_app.compute_worker.registry import ComputeOperationRegistry
from l3_app.compute_worker.registry_factory import build_compute_operation_registry
from l3_app.compute_worker.routes import router
from l3_app.compute_worker.runtime import ComputeWorkerRuntime

WORKER_ROUTE_PREFIX = "/internal/v1/compute"


def _configure_worker_loggers() -> None:
    """Expose compute lifecycle and model-operation logs from the Worker process."""
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for logger_name in ("worker", "audio_processing", "rag", "llm"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if logger.handlers:
            for handler in logger.handlers:
                handler.setLevel(logging.INFO)
            continue
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def build_compute_worker_runtime(settings: Settings, registry: ComputeOperationRegistry | None = None) -> ComputeWorkerRuntime:
    storage = LocalStorage(settings.resolved_local_storage_root)
    storage.initialize()
    return ComputeWorkerRuntime(
        registry if registry is not None else build_compute_operation_registry(settings, storage),
        ComputeExecutionPool(),
        file_store=storage,
        completed_ttl_seconds=settings.compute_worker_completed_ttl_seconds,
        max_tasks=settings.compute_worker_max_tasks,
        heartbeat_seconds=settings.compute_worker_heartbeat_seconds,
        cancel_grace_seconds=settings.compute_worker_cancel_grace_seconds,
        internal_token=settings.compute_worker_internal_token,
    )


def create_worker_app(settings: Settings | None = None, registry: ComputeOperationRegistry | None = None) -> FastAPI:
    _configure_worker_loggers()
    configured_settings = settings or get_settings()
    runtime = build_compute_worker_runtime(configured_settings, registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        storage = LocalStorage(configured_settings.resolved_local_storage_root)
        storage.initialize()
        app.state.storage = storage
        app.state.compute_worker_runtime = runtime
        await runtime.start()
        redis_store = RedisStreamStore.from_url(
            configured_settings.redis_url,
            maxlen=configured_settings.redis_stream_maxlen,
            terminal_ttl_seconds=configured_settings.redis_terminal_ttl_seconds,
        )
        producer = KafkaEventProducer(
            configured_settings.kafka_bootstrap_servers,
            f"{configured_settings.kafka_client_id}-compute-worker",
            configured_settings.kafka_request_timeout_ms,
        )
        consumer = KafkaEventConsumer(
            [Topics.COMPUTE_TASKS_IO, Topics.COMPUTE_TASKS_CPU, Topics.COMPUTE_TASKS_GPU_HIGH, Topics.COMPUTE_TASKS_GPU_NORMAL, Topics.COMPUTE_RETRY],
            bootstrap_servers=configured_settings.kafka_bootstrap_servers,
            group_id="compute-worker-v1",
            client_id=f"{configured_settings.kafka_client_id}-compute-worker",
            max_poll_interval_ms=configured_settings.kafka_consumer_max_poll_interval_ms,
        )
        cancel_consumer = KafkaEventConsumer(
            [Topics.COMPUTE_CANCEL, Topics.COMPUTE_CANCEL_RETRY],
            bootstrap_servers=configured_settings.kafka_bootstrap_servers,
            group_id="compute-cancel-projector-v1",
            client_id=f"{configured_settings.kafka_client_id}-compute-cancel",
            max_poll_interval_ms=configured_settings.kafka_consumer_max_poll_interval_ms,
        )
        await producer.start()
        await consumer.start()
        await cancel_consumer.start()
        kafka_handler = KafkaComputeTaskHandler(runtime, producer, redis_store)
        consumer_task = asyncio.create_task(
            consumer.run(
                kafka_handler.handle,
                producer=producer,
                retry_topic=Topics.COMPUTE_RETRY,
                dlq_topic=Topics.COMPUTE_DLQ,
            ),
            name="compute-kafka-consumer",
        )
        cancel_handler = KafkaComputeCancelHandler(runtime, redis_store)
        cancel_task = asyncio.create_task(
            cancel_consumer.run(
                cancel_handler.handle,
                producer=producer,
                retry_topic=Topics.COMPUTE_CANCEL_RETRY,
                dlq_topic=Topics.COMPUTE_CANCEL_DLQ,
            ),
            name="compute-cancel-consumer",
        )
        try:
            yield
        finally:
            consumer_task.cancel()
            cancel_task.cancel()
            await asyncio.gather(consumer_task, cancel_task, return_exceptions=True)
            await consumer.stop()
            await cancel_consumer.stop()
            await producer.stop()
            await redis_store.close()
            await runtime.stop()

    app = FastAPI(title="AI Record Summary Compute Worker", lifespan=lifespan)
    app.include_router(router, prefix=WORKER_ROUTE_PREFIX)
    return app
