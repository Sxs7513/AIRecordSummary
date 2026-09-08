from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID

from sqlalchemy import Engine, text

from l1_foundation.embedding_profiles import EmbeddingProfile, get_embedding_profile
from l1_foundation.infrastructure.db.session import create_database_engine
from l1_foundation.infrastructure.storage.local import LocalStorage
from l1_foundation.messaging import SyncKafkaEventProducer
from l1_foundation.settings import Settings, get_settings
from l1_foundation.streaming import SyncRedisStreamStore
from l1_foundation.worker import SyncKafkaWorkerClient
from l2_core.audio_processing.worker_tasks import EmbeddingEncodeTaskResult, embedding_encode_command
from l2_core.rag.search_document import build_retrieval_text


@dataclass(frozen=True, slots=True)
class PendingChunk:
    chunk_id: UUID
    recording_id: UUID
    chunk_index: int
    retrieval_text: str
    content_hash: str


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill SearchChunk vectors for one embedding profile.")
    parser.add_argument("--model-profile", default="qwen3-0.6b")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workspace-id", type=UUID)
    parser.add_argument("--recording-id", type=UUID)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true", help="Accepted for clarity; hash-based skipping is always enabled.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_chunks(
    engine: Engine,
    profile: EmbeddingProfile,
    *,
    workspace_id: UUID | None,
    recording_id: UUID | None,
    limit: int | None,
    force: bool,
) -> tuple[list[PendingChunk], int]:
    clauses = ["recordings.status = 'completed'"]
    values: dict[str, object] = {
        "provider": profile.provider,
        "model_name": profile.model_name,
        "dimensions": profile.dimensions,
        "model_key": profile.key,
    }
    if workspace_id is not None:
        clauses.append("recordings.workspace_id = :workspace_id")
        values["workspace_id"] = workspace_id
    if recording_id is not None:
        clauses.append("chunks.recording_id = :recording_id")
        values["recording_id"] = recording_id
    limit_sql = " limit :limit" if limit is not None else ""
    if limit is not None:
        values["limit"] = limit
    with engine.connect() as connection:
        rows = (
            connection.execute(
                text(
                    f"""
                    select chunks.id, chunks.recording_id, chunks.chunk_index, chunks.text, chunks.metadata,
                           existing.content_hash as existing_hash
                    from recording_search_chunks chunks
                    join recordings on recordings.id = chunks.recording_id
                    left join (
                        select vectors.chunk_id, vectors.content_hash
                        from recording_search_chunk_embeddings vectors
                        join embedding_models models on models.id = vectors.embedding_model_id
                        where vectors.model_key = :model_key
                          and models.provider = :provider
                          and models.model_name = :model_name
                          and models.dimensions = :dimensions
                    ) existing on existing.chunk_id = chunks.id
                    where {" and ".join(clauses)}
                    order by chunks.recording_id, chunks.chunk_index
                    {limit_sql}
                    """
                ),
                values,
            )
            .mappings()
            .all()
        )
    pending: list[PendingChunk] = []
    skipped = 0
    for row in rows:
        metadata = cast(dict[str, Any], row["metadata"] or {})
        retrieval_text = build_retrieval_text(
            str(row["text"]),
            cast(str | None, metadata.get("topic")),
            [str(value) for value in cast(list[object], metadata.get("terms") or [])],
            cast(str | None, metadata.get("search_context")),
        )
        content_hash = hashlib.sha256(retrieval_text.encode("utf-8")).hexdigest()
        if not force and row["existing_hash"] == content_hash:
            skipped += 1
            continue
        pending.append(
            PendingChunk(
                chunk_id=UUID(str(row["id"])),
                recording_id=UUID(str(row["recording_id"])),
                chunk_index=int(row["chunk_index"]),
                retrieval_text=retrieval_text,
                content_hash=content_hash,
            )
        )
    return pending, skipped


def _persist_batch(
    engine: Engine,
    profile: EmbeddingProfile,
    result: EmbeddingEncodeTaskResult,
    chunks: list[PendingChunk],
) -> None:
    if (
        result.embedding_profile != profile.key
        or result.provider != profile.provider
        or result.model_name != profile.model_name
        or result.dimensions != profile.dimensions
        or result.distance_metric != profile.distance_metric
    ):
        raise ValueError("Embedding Worker result does not match requested profile")
    if len(result.vectors) != len(chunks) or any(len(vector) != profile.dimensions for vector in result.vectors):
        raise ValueError("Embedding Worker returned an invalid vector matrix")
    with engine.begin() as connection:
        model_id = UUID(
            str(
                connection.execute(
                    text(
                        """
                        insert into embedding_models (provider, model_name, dimensions, distance_metric, is_active)
                        values (:provider, :model_name, :dimensions, :distance_metric, true)
                        on conflict (provider, model_name, dimensions) do update set
                            distance_metric = excluded.distance_metric, is_active = true
                        returning id
                        """
                    ),
                    {
                        "provider": profile.provider,
                        "model_name": profile.model_name,
                        "dimensions": profile.dimensions,
                        "distance_metric": profile.distance_metric,
                    },
                ).scalar_one()
            )
        )
        for chunk, vector in zip(chunks, result.vectors, strict=True):
            connection.execute(
                text(
                    """
                    insert into recording_search_chunk_embeddings (
                        chunk_id, embedding_model_id, model_key, dimensions, content_hash, embedding
                    ) values (
                        :chunk_id, :embedding_model_id, :model_key, :dimensions, :content_hash,
                        cast(:embedding as halfvec)
                    )
                    on conflict (chunk_id, embedding_model_id) do update set
                        model_key = excluded.model_key, dimensions = excluded.dimensions,
                        content_hash = excluded.content_hash, embedding = excluded.embedding, updated_at = now()
                    """
                ),
                {
                    "chunk_id": chunk.chunk_id,
                    "embedding_model_id": model_id,
                    "model_key": profile.key,
                    "dimensions": profile.dimensions,
                    "content_hash": chunk.content_hash,
                    "embedding": _vector_literal(vector),
                },
            )


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.8g}" for value in vector) + "]"


def run(args: argparse.Namespace, settings: Settings) -> int:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    profile = get_embedding_profile(args.model_profile)
    engine = create_database_engine(settings)
    producer: SyncKafkaEventProducer | None = None
    redis: SyncRedisStreamStore | None = None
    worker: SyncKafkaWorkerClient | None = None
    try:
        chunks, skipped = _load_chunks(
            engine,
            profile,
            workspace_id=args.workspace_id,
            recording_id=args.recording_id,
            limit=args.limit,
            force=args.force,
        )
        print(f"search embeddings: profile={profile.key} pending={len(chunks)} skipped={skipped} dry_run={args.dry_run}")
        if args.dry_run or not chunks:
            return 0
        redis = SyncRedisStreamStore.from_url(
            settings.redis_url,
            maxlen=settings.redis_stream_maxlen,
            terminal_ttl_seconds=settings.redis_terminal_ttl_seconds,
        )
        active_producer = SyncKafkaEventProducer(
            settings.kafka_bootstrap_servers,
            f"{settings.kafka_client_id}-search-embedding-backfill",
            settings.kafka_request_timeout_ms,
        )
        active_producer.start()
        producer = active_producer
        storage = LocalStorage(settings.resolved_local_storage_root)
        storage.initialize()
        worker = SyncKafkaWorkerClient(
            active_producer,
            redis,
            storage,
            reply_wait_timeout_seconds=settings.compute_reply_wait_timeout_seconds,
        )
        worker.ready()
        total_batches = (len(chunks) + args.batch_size - 1) // args.batch_size
        for batch_number, offset in enumerate(range(0, len(chunks), args.batch_size), start=1):
            batch = chunks[offset : offset + args.batch_size]
            result = worker.execute(
                embedding_encode_command([chunk.retrieval_text for chunk in batch], profile.key),
                result_type=EmbeddingEncodeTaskResult,
            )
            _persist_batch(engine, profile, result, batch)
            print(f"search embeddings: batch={batch_number}/{total_batches} completed={offset + len(batch)}/{len(chunks)}")
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
