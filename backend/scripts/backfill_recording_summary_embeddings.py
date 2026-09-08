from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import Engine, text

from l1_foundation.infrastructure.db.session import create_database_engine
from l1_foundation.infrastructure.storage.local import LocalStorage
from l1_foundation.messaging import SyncKafkaEventProducer
from l1_foundation.settings import Settings, get_settings
from l1_foundation.streaming import SyncRedisStreamStore
from l1_foundation.worker import SyncKafkaWorkerClient
from l2_core.audio_processing.stages.build_search_chunks.token_counter import EmbeddingTokenCounter
from l2_core.audio_processing.stages.summary.retrieval_text import build_summary_retrieval_text
from l2_core.audio_processing.worker_tasks import EmbeddingEncodeTaskResult, embedding_encode_command

_PROVIDER = "sentence_transformers"


@dataclass(frozen=True, slots=True)
class RecordingProfileDocument:
    recording_id: UUID
    title: str
    retrieval_text: str
    content_hash: str


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill recording-profile embeddings without rerunning the summary LLM.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--recording-id", type=UUID)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _ensure_schema(engine: Engine) -> None:
    with engine.connect() as connection:
        tables = connection.execute(
            text("select to_regclass('public.recording_retrieval_documents'), to_regclass('public.recording_retrieval_document_embeddings')")
        ).one()
    if any(value is None for value in tables):
        raise RuntimeError("Embedding tables are missing; run scripts/db/init-python-backend.sh first")


def _load_documents(
    engine: Engine,
    settings: Settings,
    *,
    max_tokens: int,
    force: bool,
    limit: int | None,
    recording_id: UUID | None,
) -> tuple[list[RecordingProfileDocument], int]:
    counter = EmbeddingTokenCounter(settings.embedding_model, settings.resolved_embedding_model_cache_dir)
    clauses = ["btrim(summaries.summary_text) <> ''"]
    values: dict[str, object] = {
        "provider": _PROVIDER,
        "model_key": settings.embedding_profile,
        "model_name": settings.embedding_model,
        "dimensions": settings.embedding_dimensions,
    }
    if recording_id is not None:
        clauses.append("summaries.recording_id = :recording_id")
        values["recording_id"] = recording_id
    limit_sql = " limit :limit" if limit is not None else ""
    if limit is not None:
        values["limit"] = limit
    with engine.connect() as connection:
        rows = (
            connection.execute(
                text(
                    f"""
                    select summaries.recording_id, recordings.title, summaries.summary_text,
                           existing.content_hash as existing_hash
                    from recording_summaries summaries
                    join recordings on recordings.id = summaries.recording_id
                    left join (
                        select documents.recording_id, vectors.content_hash
                        from recording_retrieval_documents documents
                        join recording_retrieval_document_embeddings vectors on vectors.document_id = documents.id
                        join embedding_models models on models.id = vectors.embedding_model_id
                        where models.provider = :provider
                          and models.model_name = :model_name
                          and models.dimensions = :dimensions
                          and vectors.model_key = :model_key
                          and documents.document_index = 0
                    ) existing on existing.recording_id = summaries.recording_id
                    where {" and ".join(clauses)}
                    order by summaries.updated_at, summaries.recording_id
                    {limit_sql}
                    """
                ),
                values,
            )
            .mappings()
            .all()
        )
    documents: list[RecordingProfileDocument] = []
    skipped = 0
    for row in rows:
        retrieval_text = build_summary_retrieval_text(
            str(row["title"]),
            str(row["summary_text"]),
            count_tokens=counter,
            max_tokens=max_tokens,
        )
        content_hash = hashlib.sha256(retrieval_text.encode("utf-8")).hexdigest()
        if not force and row["existing_hash"] == content_hash:
            skipped += 1
            continue
        documents.append(
            RecordingProfileDocument(
                recording_id=UUID(str(row["recording_id"])),
                title=str(row["title"]),
                retrieval_text=retrieval_text,
                content_hash=content_hash,
            )
        )
    return documents, skipped


def _persist_batch(engine: Engine, result: EmbeddingEncodeTaskResult, documents: list[RecordingProfileDocument]) -> None:
    if len(result.vectors) != len(documents):
        raise ValueError(f"Embedding result count mismatch: expected {len(documents)}, got {len(result.vectors)}")
    if any(len(vector) != result.dimensions for vector in result.vectors):
        raise ValueError("Embedding vector dimensions do not match result metadata")
    with engine.begin() as connection:
        model_id = UUID(
            str(
                connection.execute(
                    text(
                        """
                        insert into embedding_models (provider, model_name, dimensions, distance_metric, is_active)
                        values (:provider, :model_name, :dimensions, :distance_metric, true)
                        on conflict (provider, model_name, dimensions) do update set
                            distance_metric = excluded.distance_metric,
                            is_active = true
                        returning id
                        """
                    ),
                    {
                        "provider": result.provider,
                        "model_name": result.model_name,
                        "dimensions": result.dimensions,
                        "distance_metric": result.distance_metric,
                    },
                ).scalar_one()
            )
        )
        for document, vector in zip(documents, result.vectors, strict=True):
            document_id = UUID(
                str(
                    connection.execute(
                        text(
                            """
                    insert into recording_retrieval_documents (
                        recording_id, document_index, document_type, retrieval_text, content_hash
                    ) values (
                        :recording_id, 0, 'profile', :retrieval_text, :content_hash
                    )
                    on conflict (recording_id, document_index) do update set
                        document_type = excluded.document_type,
                        retrieval_text = excluded.retrieval_text,
                        content_hash = excluded.content_hash,
                        updated_at = now()
                    returning id
                    """
                        ),
                        {
                            "recording_id": document.recording_id,
                            "retrieval_text": document.retrieval_text,
                            "content_hash": document.content_hash,
                        },
                    ).scalar_one()
                )
            )
            connection.execute(
                text("delete from recording_retrieval_document_embeddings where document_id = :document_id and content_hash <> :content_hash"),
                {"document_id": document_id, "content_hash": document.content_hash},
            )
            connection.execute(
                text(
                    """
                    insert into recording_retrieval_document_embeddings (
                        document_id, embedding_model_id, model_key, dimensions, content_hash, embedding
                    ) values (
                        :document_id, :embedding_model_id, :model_key, :dimensions, :content_hash,
                        cast(:embedding as halfvec)
                    )
                    on conflict (document_id, embedding_model_id) do update set
                        model_key = excluded.model_key, dimensions = excluded.dimensions,
                        content_hash = excluded.content_hash, embedding = excluded.embedding, updated_at = now()
                    """
                ),
                {
                    "document_id": document_id,
                    "embedding_model_id": model_id,
                    "model_key": result.embedding_profile,
                    "dimensions": result.dimensions,
                    "content_hash": document.content_hash,
                    "embedding": _vector_literal(vector),
                },
            )


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.8g}" for value in vector) + "]"


def _batches(values: list[RecordingProfileDocument], size: int) -> list[list[RecordingProfileDocument]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def run(args: argparse.Namespace, settings: Settings) -> int:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    engine = create_database_engine(settings)
    producer: SyncKafkaEventProducer | None = None
    redis: SyncRedisStreamStore | None = None
    worker: SyncKafkaWorkerClient | None = None
    try:
        _ensure_schema(engine)
        documents, skipped = _load_documents(
            engine,
            settings,
            max_tokens=args.max_tokens,
            force=args.force,
            limit=args.limit,
            recording_id=args.recording_id,
        )
        print(f"recording profile embeddings: pending={len(documents)} skipped={skipped} dry_run={args.dry_run}")
        if args.dry_run or not documents:
            for document in documents[:5]:
                print(f"- {document.recording_id} {document.title}: {len(document.retrieval_text)} chars")
            return 0

        redis = SyncRedisStreamStore.from_url(
            settings.redis_url,
            maxlen=settings.redis_stream_maxlen,
            terminal_ttl_seconds=settings.redis_terminal_ttl_seconds,
        )
        active_producer = SyncKafkaEventProducer(
            settings.kafka_bootstrap_servers,
            f"{settings.kafka_client_id}-summary-embedding-backfill",
            settings.kafka_request_timeout_ms,
        )
        active_producer.start()
        producer = active_producer
        storage = LocalStorage(settings.resolved_local_storage_root)
        storage.initialize()
        worker = SyncKafkaWorkerClient(active_producer, redis, storage, reply_wait_timeout_seconds=settings.compute_reply_wait_timeout_seconds)
        worker.ready()
        completed = 0
        batches = _batches(documents, args.batch_size)
        for batch_index, batch in enumerate(batches, start=1):
            result = worker.execute(
                embedding_encode_command([document.retrieval_text for document in batch], settings.embedding_profile),
                result_type=EmbeddingEncodeTaskResult,
            )
            _persist_batch(engine, result, batch)
            completed += len(batch)
            print(f"recording profile embeddings: batch={batch_index}/{len(batches)} completed={completed}/{len(documents)}")
        return 0
    finally:
        if worker is not None:
            worker.close()
        if producer is not None:
            producer.stop()
        if redis is not None:
            redis.close()
        engine.dispose()


def main() -> None:
    raise SystemExit(run(_arguments(), get_settings()))


if __name__ == "__main__":
    main()
