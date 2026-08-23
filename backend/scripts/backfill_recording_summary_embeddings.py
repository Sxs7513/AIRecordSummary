from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import Engine, text

from l1_foundation.infrastructure.db.session import create_database_engine
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
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                do $$
                begin
                    if to_regclass('public.recording_retrieval_documents') is null
                       and to_regclass('public.recording_summary_embeddings') is not null then
                        alter table recording_summary_embeddings rename to recording_retrieval_documents;
                    end if;
                    if to_regclass('public.recording_summary_embeddings_recording_id_idx') is not null
                       and to_regclass('public.recording_retrieval_documents_recording_id_idx') is null then
                        alter index recording_summary_embeddings_recording_id_idx
                            rename to recording_retrieval_documents_recording_id_idx;
                    end if;
                    if to_regclass('public.recording_summary_embeddings_hnsw_idx') is not null
                       and to_regclass('public.recording_retrieval_documents_hnsw_idx') is null then
                        alter index recording_summary_embeddings_hnsw_idx
                            rename to recording_retrieval_documents_hnsw_idx;
                    end if;
                end
                $$
                """
            )
        )
        connection.execute(
            text(
                """
                create table if not exists recording_retrieval_documents (
                    id uuid primary key default gen_random_uuid(),
                    recording_id uuid not null references recordings(id) on delete cascade,
                    embedding_model_id uuid not null references embedding_models(id) on delete restrict,
                    document_index integer not null default 0 check (document_index >= 0),
                    document_type text not null default 'profile' check (document_type in ('profile', 'overview', 'outline')),
                    retrieval_text text not null,
                    content_hash text not null,
                    embedding halfvec(2560) not null,
                    created_at timestamptz not null default now(),
                    updated_at timestamptz not null default now(),
                    unique (recording_id, embedding_model_id, document_index)
                )
                """
            )
        )
        connection.execute(
            text(
                "alter table recording_retrieval_documents "
                "drop constraint if exists recording_summary_embeddings_recording_id_fkey"
            )
        )
        connection.execute(
            text(
                "alter table recording_retrieval_documents "
                "drop constraint if exists recording_retrieval_documents_recording_id_fkey"
            )
        )
        connection.execute(
            text(
                "alter table recording_retrieval_documents "
                "add constraint recording_retrieval_documents_recording_id_fkey "
                "foreign key (recording_id) references recordings(id) on delete cascade"
            )
        )
        connection.execute(
            text(
                "alter table recording_retrieval_documents "
                "drop constraint if exists recording_summary_embeddings_document_type_check"
            )
        )
        connection.execute(
            text(
                "alter table recording_retrieval_documents "
                "drop constraint if exists recording_retrieval_documents_document_type_check"
            )
        )
        connection.execute(text("update recording_retrieval_documents set document_type = 'profile' where document_type <> 'profile'"))
        connection.execute(
            text(
                "alter table recording_retrieval_documents "
                "add constraint recording_retrieval_documents_document_type_check "
                "check (document_type in ('profile', 'overview', 'outline'))"
            )
        )
        connection.execute(
            text(
                """
                create index if not exists recording_retrieval_documents_recording_id_idx
                on recording_retrieval_documents (recording_id)
                """
            )
        )
        connection.execute(
            text(
                """
                create index if not exists recording_retrieval_documents_hnsw_idx
                on recording_retrieval_documents using hnsw (embedding halfvec_cosine_ops)
                """
            )
        )


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
                        select embeddings.recording_id, embeddings.content_hash
                        from recording_retrieval_documents embeddings
                        join embedding_models models on models.id = embeddings.embedding_model_id
                        where models.provider = :provider
                          and models.model_name = :model_name
                          and models.dimensions = :dimensions
                          and embeddings.document_index = 0
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
    if result.dimensions != 2560:
        raise ValueError(f"recording_retrieval_documents expects 2560 dimensions, got {result.dimensions}")
    with engine.begin() as connection:
        model_id = UUID(
            str(
                connection.execute(
                    text(
                        """
                        insert into embedding_models (provider, model_name, dimensions, distance_metric, is_active)
                        values (:provider, :model_name, :dimensions, 'cosine', true)
                        on conflict (provider, model_name, dimensions) do update set
                            distance_metric = excluded.distance_metric,
                            is_active = true
                        returning id
                        """
                    ),
                    {"provider": result.provider, "model_name": result.model_name, "dimensions": result.dimensions},
                ).scalar_one()
            )
        )
        for document, vector in zip(documents, result.vectors, strict=True):
            connection.execute(
                text(
                    """
                    insert into recording_retrieval_documents (
                        recording_id, embedding_model_id, document_index, document_type,
                        retrieval_text, content_hash, embedding
                    ) values (
                        :recording_id, :embedding_model_id, 0, 'profile',
                        :retrieval_text, :content_hash, cast(:embedding as halfvec)
                    )
                    on conflict (recording_id, embedding_model_id, document_index) do update set
                        document_type = excluded.document_type,
                        retrieval_text = excluded.retrieval_text,
                        content_hash = excluded.content_hash,
                        embedding = excluded.embedding,
                        updated_at = now()
                    """
                ),
                {
                    "recording_id": document.recording_id,
                    "embedding_model_id": model_id,
                    "retrieval_text": document.retrieval_text,
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
        worker = SyncKafkaWorkerClient(active_producer, redis, poll_interval_seconds=settings.compute_worker_poll_interval_seconds)
        worker.ready()
        completed = 0
        batches = _batches(documents, args.batch_size)
        for batch_index, batch in enumerate(batches, start=1):
            result = worker.execute(
                embedding_encode_command([document.retrieval_text for document in batch]),
                result_type=EmbeddingEncodeTaskResult,
            )
            _persist_batch(engine, result, batch)
            completed += len(batch)
            print(f"recording profile embeddings: batch={batch_index}/{len(batches)} completed={completed}/{len(documents)}")
        return 0
    finally:
        if producer is not None:
            producer.stop()
        if redis is not None:
            redis.close()
        engine.dispose()


def main() -> None:
    raise SystemExit(run(_arguments(), get_settings()))


if __name__ == "__main__":
    main()
