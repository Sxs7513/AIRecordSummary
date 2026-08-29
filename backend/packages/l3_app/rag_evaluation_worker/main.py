from __future__ import annotations

import logging

from l1_foundation.infrastructure.db.session import create_database_engine
from l1_foundation.infrastructure.storage.local import LocalStorage
from l1_foundation.messaging import SyncKafkaEventProducer
from l1_foundation.settings import get_settings
from l1_foundation.streaming import SyncRedisStreamStore
from l1_foundation.worker import SyncKafkaWorkerClient
from l2_core.rag_evaluation.worker import RagEvaluationWorker


def _configure_loggers() -> None:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in ("evaluation", "rag"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if logger.handlers:
            continue
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def run() -> None:
    _configure_loggers()
    settings = get_settings()
    engine = create_database_engine(settings)
    storage = LocalStorage(settings.resolved_local_storage_root)
    storage.initialize()
    redis = SyncRedisStreamStore.from_url(
        settings.redis_url,
        maxlen=settings.redis_stream_maxlen,
        terminal_ttl_seconds=settings.redis_terminal_ttl_seconds,
    )
    producer = SyncKafkaEventProducer(
        settings.kafka_bootstrap_servers,
        f"{settings.kafka_client_id}-rag-evaluation-sync-compute",
        settings.kafka_request_timeout_ms,
    )
    producer.start()
    compute = SyncKafkaWorkerClient(
        producer,
        redis,
        storage,
        reply_wait_timeout_seconds=settings.compute_reply_wait_timeout_seconds,
    )
    compute.ready()
    worker = RagEvaluationWorker(engine, settings, compute)
    try:
        worker.run_forever()
    finally:
        compute.close()
        producer.stop()
        redis.close()
        engine.dispose()


if __name__ == "__main__":
    run()
