from __future__ import annotations

import gc
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from time import perf_counter
from typing import Any, Protocol, cast
from uuid import UUID

from sqlalchemy import Engine, text

from l1_foundation.infrastructure.huggingface import resolve_local_snapshot
from l1_foundation.settings import Settings
from l2_core.rag.contracts import Evidence, EvidenceChunk, EvidenceFacts, EvidenceRecording, ResolvedFilters
from l2_core.rag.normalization import normalize_search_text
from l2_core.rag.observability import log_event

MAX_SCOPE_RECORDINGS = 50
MAX_SCOPE_UTTERANCES = 1_000
MAX_SCOPE_CHARS = 30_000
logger = logging.getLogger("rag")


@dataclass
class RetrievalCandidate:
    row: dict[str, object]
    vector_rank: int | None = None
    vector_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    fused_score: float = 0.0


class EmbeddingModel(Protocol):
    def encode(self, texts: Sequence[str], **kwargs: object) -> object: ...


class SentenceTransformersModule(Protocol):
    def SentenceTransformer(self, model_name_or_path: str, **kwargs: object) -> EmbeddingModel: ...


class RagRetriever:
    def __init__(self, engine: Engine, settings: Settings) -> None:
        self._engine = engine
        self._settings = settings
        self._model: EmbeddingModel | None = None

    def release(self) -> None:
        had_model = self._model is not None
        self._model = None
        gc.collect()
        try:
            torch = cast(Any, import_module("torch"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, RuntimeError, AttributeError) as error:
            logger.warning("rag embedding model released, but device cache cleanup failed: %s", error)
        else:
            if not had_model:
                return
            logger.info("rag embedding model and device cache released")

    @property
    def hybrid_search_enabled(self) -> bool:
        return self._settings.rag_hybrid_search_enabled

    def resolve_recording_scope(self, filters: ResolvedFilters, limit: int | None, rank: int | None) -> list[UUID]:
        """Resolve all recording-level filters once, before either chunk branch runs."""

        values: dict[str, object] = {}
        clauses = ["recordings.status = 'completed'"]
        self._append_recording_filters(clauses, values, filters)
        pagination = ""
        if limit is not None or rank is not None:
            values["limit"] = 1 if rank else max(1, min(10, limit or 1))
            values["offset"] = max(0, min(9, rank - 1)) if rank else 0
            pagination = " limit :limit offset :offset"
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(
                    f"select recordings.id from recordings where {' and '.join(clauses)} "
                    f"order by recordings.created_at desc{pagination}"
                ),
                values,
            ).scalars()
            return [cast(UUID, row) for row in rows]

    def resolve_ranked_recording_ids(self, filters: ResolvedFilters, limit: int | None, rank: int | None) -> list[UUID]:
        """Compatibility alias for callers migrating to the unified scope resolver."""

        return self.resolve_recording_scope(filters, limit, rank)

    def generate_query_embedding(self, topic: str) -> list[float]:
        try:
            return self._embed(topic)
        finally:
            self.release()

    def retrieve_vector_candidates(
        self,
        embedding: list[float],
        filters: ResolvedFilters,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        vector = "[" + ",".join(f"{item:.8g}" for item in embedding) + "]"
        values: dict[str, object] = {
            "embedding": vector,
            "embedding_provider": "sentence_transformers",
            "embedding_model": self._settings.embedding_model,
            "embedding_dimensions": self._settings.embedding_dimensions,
            "limit": limit or self._settings.rag_vector_candidate_limit,
        }
        clauses = ["recordings.status = 'completed'"]
        clauses.extend(
            (
                "embedding_models.provider = :embedding_provider",
                "embedding_models.model_name = :embedding_model",
                "embedding_models.dimensions = :embedding_dimensions",
            )
        )
        self._append_chunk_filters(clauses, values, filters)
        with self._engine.connect() as connection:
            return [
                dict(row)
                for row in (
                connection.execute(
                    text(
                        f"""
                    select chunks.id as chunk_id, chunks.recording_id, chunks.text, chunks.start_ms, chunks.end_ms,
                           chunks.speaker_labels, chunks.is_target_person, chunks.source_utterance_segment_ids,
                           recordings.title, recordings.file_name,
                           recordings.location, recordings.duration_seconds,
                           1 - (chunks.embedding <=> cast(:embedding as halfvec)) as score
                    from recording_search_chunks chunks
                    join recordings on recordings.id = chunks.recording_id
                    join embedding_models on embedding_models.id = chunks.embedding_model_id
                    where {" and ".join(clauses)}
                    order by chunks.embedding <=> cast(:embedding as halfvec)
                    limit :limit
                    """
                    ),
                    values,
                )
                .mappings()
                .all()
                )
            ]

    def retrieve_lexical_candidates(
        self,
        topic: str,
        filters: ResolvedFilters,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        query = normalize_search_text(topic)
        if not query:
            return []
        values: dict[str, object] = {
            "query": query,
            "limit": limit or self._settings.rag_lexical_candidate_limit,
        }
        clauses = ["recordings.status = 'completed'", "word_similarity(:query, chunks.normalized_text) > 0"]
        self._append_chunk_filters(clauses, values, filters)
        with self._engine.connect() as connection:
            return [
                dict(row)
                for row in (
                    connection.execute(
                        text(
                            f"""
                            select chunks.id as chunk_id, chunks.recording_id, chunks.text,
                                   chunks.start_ms, chunks.end_ms, chunks.speaker_labels,
                                   chunks.is_target_person, chunks.source_utterance_segment_ids,
                                   recordings.title, recordings.file_name,
                                   recordings.location, recordings.duration_seconds,
                                   word_similarity(:query, chunks.normalized_text) as score
                            from recording_search_chunks chunks
                            join recordings on recordings.id = chunks.recording_id
                            where {" and ".join(clauses)}
                            order by :query <<-> chunks.normalized_text
                            limit :limit
                            """
                        ),
                        values,
                    )
                    .mappings()
                    .all()
                )
            ]

    def fuse_candidates(
        self,
        vector_rows: list[dict[str, object]],
        lexical_rows: list[dict[str, object]],
        limit: int,
    ) -> list[dict[str, object]]:
        candidates: dict[UUID, RetrievalCandidate] = {}
        for rank, row in enumerate(vector_rows, start=1):
            chunk_id = cast(UUID, row["chunk_id"])
            candidates[chunk_id] = RetrievalCandidate(
                row=row,
                vector_rank=rank,
                vector_score=float(cast(float, row["score"])),
            )
        for rank, row in enumerate(lexical_rows, start=1):
            chunk_id = cast(UUID, row["chunk_id"])
            candidate = candidates.setdefault(chunk_id, RetrievalCandidate(row=row))
            candidate.lexical_rank = rank
            candidate.lexical_score = float(cast(float, row["score"]))
        for candidate in candidates.values():
            if candidate.vector_rank is not None:
                candidate.fused_score += self._settings.rag_vector_weight / (self._settings.rag_rrf_k + candidate.vector_rank)
            if candidate.lexical_rank is not None:
                candidate.fused_score += self._settings.rag_lexical_weight / (self._settings.rag_rrf_k + candidate.lexical_rank)
        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                -item.fused_score,
                min(rank for rank in (item.vector_rank, item.lexical_rank) if rank is not None),
                str(item.row["chunk_id"]),
            ),
        )
        result: list[dict[str, object]] = []
        for candidate in ordered[: min(limit, self._settings.rag_fused_candidate_limit)]:
            row = dict(candidate.row)
            row["score"] = candidate.fused_score
            row["match_type"] = (
                "hybrid"
                if candidate.vector_rank is not None and candidate.lexical_rank is not None
                else "vector"
                if candidate.vector_rank is not None
                else "lexical"
            )
            result.append(row)
        return result

    def expand_candidates(self, rows: list[dict[str, object]]) -> list[Evidence]:
        with self._engine.connect() as connection:
            expanded = self._expand_chunk_contexts(connection, rows)
        merged = self._merge_overlapping_contexts(expanded)
        return [self._chunk_evidence(index, row) for index, row in enumerate(merged, start=1)]

    @staticmethod
    def _merge_overlapping_contexts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        """Merge expanded candidates that cover the same recording time range."""

        merged: list[dict[str, object]] = []
        for row in rows:
            overlapping = next(
                (
                    item
                    for item in merged
                    if item["recording_id"] == row["recording_id"]
                    and int(cast(int, item["start_ms"])) <= int(cast(int, row["end_ms"]))
                    and int(cast(int, row["start_ms"])) <= int(cast(int, item["end_ms"]))
                ),
                None,
            )
            if overlapping is None:
                merged.append(dict(row))
                continue
            overlapping["start_ms"] = min(
                int(cast(int, overlapping["start_ms"])),
                int(cast(int, row["start_ms"])),
            )
            overlapping["end_ms"] = max(
                int(cast(int, overlapping["end_ms"])),
                int(cast(int, row["end_ms"])),
            )
            overlapping["text"] = "\n".join(
                dict.fromkeys(
                    [
                        *str(overlapping["text"]).splitlines(),
                        *str(row["text"]).splitlines(),
                    ]
                )
            )
            overlapping["speaker_labels"] = list(
                dict.fromkeys(
                    [
                        *cast(list[str], overlapping.get("speaker_labels") or []),
                        *cast(list[str], row.get("speaker_labels") or []),
                    ]
                )
            )
            overlapping["matched_speaker_profile_ids"] = list(
                dict.fromkeys(
                    [
                        *cast(list[UUID], overlapping.get("matched_speaker_profile_ids") or []),
                        *cast(list[UUID], row.get("matched_speaker_profile_ids") or []),
                    ]
                )
            )
            overlapping["is_target_person"] = bool(overlapping["is_target_person"]) or bool(row["is_target_person"])
            if overlapping.get("match_type") != row.get("match_type"):
                overlapping["match_type"] = "hybrid"
        return merged

    def retrieve_chunks(
        self,
        topic: str,
        filters: ResolvedFilters,
        limit: int,
        run_id: str = "standalone",
    ) -> list[Evidence]:
        if not self._settings.rag_hybrid_search_enabled:
            embedding = self.generate_query_embedding(topic)
            rows = self.retrieve_vector_candidates(embedding, filters, limit)
            for row in rows:
                row["match_type"] = "vector"
            return self.expand_candidates(rows)

        started = perf_counter()
        vector_rows: list[dict[str, object]] = []
        lexical_rows: list[dict[str, object]] = []
        vector_error: Exception | None = None
        lexical_error: Exception | None = None
        try:
            embedding = self.generate_query_embedding(topic)
            vector_rows = self.retrieve_vector_candidates(embedding, filters)
        except Exception as error:
            vector_error = error
            log_event(
                "retrieval_branch_failed",
                run_id,
                level=logging.WARNING,
                exc_info=True,
                branch="vector",
                error_type=type(error).__name__,
            )
        try:
            lexical_rows = self.retrieve_lexical_candidates(topic, filters)
        except Exception as error:
            lexical_error = error
            log_event(
                "retrieval_branch_failed",
                run_id,
                level=logging.WARNING,
                exc_info=True,
                branch="lexical",
                error_type=type(error).__name__,
            )
        if vector_error is not None and lexical_error is not None:
            raise RuntimeError("Both RAG hybrid retrieval branches failed") from vector_error
        fused = self.fuse_candidates(vector_rows, lexical_rows, limit)
        evidence = self.expand_candidates(fused)
        overlap = len({row["chunk_id"] for row in vector_rows} & {row["chunk_id"] for row in lexical_rows})
        log_event(
            "hybrid_retrieval_completed",
            run_id,
            query_chars=len(topic),
            scope_recording_count=len(filters.recording_ids),
            vector_candidates=len(vector_rows),
            lexical_candidates=len(lexical_rows),
            overlap=overlap,
            fused_candidates=len(fused),
            evidence_count=len(evidence),
            vector_degraded=vector_error is not None,
            lexical_degraded=lexical_error is not None,
            elapsed_ms=round((perf_counter() - started) * 1_000, 2),
        )
        return evidence

    def _expand_chunk_contexts(self, connection: Any, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        window = self._settings.rag_chunk_context_window_utterances
        chunk_ids = [str(row["chunk_id"]) for row in rows if row.get("source_utterance_segment_ids")]
        if not chunk_ids:
            return rows
        utterances = (
            connection.execute(
                text(
                    """
                    with candidate_bounds as (
                        select chunks.id as chunk_id, chunks.recording_id,
                               min(source_utterances.utterance_index) as first_index,
                               max(source_utterances.utterance_index) as last_index
                        from recording_search_chunks chunks
                        join utterance_segments source_utterances
                          on source_utterances.recording_id = chunks.recording_id
                         and source_utterances.id = any(chunks.source_utterance_segment_ids)
                        where chunks.id = any(cast(:chunk_ids as uuid[]))
                        group by chunks.id, chunks.recording_id
                    )
                    select bounds.chunk_id, utterances.utterance_index,
                           coalesce(profiles.display_name, mappings.display_name, utterances.speaker_label) as speaker_label,
                           utterances.text, utterances.start_ms, utterances.end_ms,
                           (mappings.speaker_profile_id is not null or utterances.is_target_person) as is_target_person,
                           mappings.speaker_profile_id
                    from candidate_bounds bounds
                    join utterance_segments utterances
                      on utterances.recording_id = bounds.recording_id
                     and utterances.utterance_index
                         between greatest(0, bounds.first_index - :window) and bounds.last_index + :window
                    left join recording_speaker_mappings mappings
                      on mappings.recording_id = utterances.recording_id
                     and mappings.speaker_cluster_id = utterances.speaker_cluster_id
                    left join speaker_profiles profiles on profiles.id = mappings.speaker_profile_id
                    order by bounds.chunk_id, utterances.utterance_index
                    """
                ),
                {"chunk_ids": chunk_ids, "window": window},
            )
            .mappings()
            .all()
        )
        utterances_by_chunk: dict[UUID, list[dict[str, object]]] = {}
        for utterance in utterances:
            utterance_data = dict(utterance)
            utterances_by_chunk.setdefault(cast(UUID, utterance_data["chunk_id"]), []).append(utterance_data)
        for row in rows:
            chunk_utterances = utterances_by_chunk.get(cast(UUID, row["chunk_id"]), [])
            if not chunk_utterances:
                continue
            row["text"] = "\n".join(f"{item['speaker_label'] or 'Unknown Speaker'}: {item['text']}" for item in chunk_utterances)
            row["start_ms"] = chunk_utterances[0]["start_ms"]
            row["end_ms"] = chunk_utterances[-1]["end_ms"]
            row["speaker_labels"] = list(
                dict.fromkeys(str(item["speaker_label"]) for item in chunk_utterances if item["speaker_label"])
            )
            row["is_target_person"] = any(bool(item["is_target_person"]) for item in chunk_utterances)
            row["matched_speaker_profile_ids"] = list(
                dict.fromkeys(item["speaker_profile_id"] for item in chunk_utterances if item["speaker_profile_id"] is not None)
            )
        return rows

    def retrieve_scope(self, filters: ResolvedFilters, limit: int | None, rank: int | None) -> list[Evidence]:
        values: dict[str, object] = {}
        clauses = ["recordings.status = 'completed'"]
        self._append_recording_filters(clauses, values, filters)
        values["limit"] = 1 if rank else max(1, min(MAX_SCOPE_RECORDINGS, limit or MAX_SCOPE_RECORDINGS))
        values["offset"] = max(0, min(9, rank - 1)) if rank else 0
        with self._engine.connect() as connection:
            recordings = (
                connection.execute(
                    text(
                        f"""select recordings.id, recordings.title, recordings.file_name, recordings.location, recordings.duration_seconds
                    from recordings where {" and ".join(clauses)} order by recordings.created_at desc limit :limit offset :offset"""
                    ),
                    values,
                )
                .mappings()
                .all()
            )
            if not recordings:
                return []
            recording_ids = [cast(UUID, recording["id"]) for recording in recordings]
            utterance_rows = (
                connection.execute(
                    text(
                        """
                        with ranked_utterances as (
                            select utterances.recording_id, utterances.utterance_index,
                                   coalesce(profiles.display_name, mappings.display_name, utterances.speaker_label) as speaker_label,
                                   utterances.text, utterances.start_ms, utterances.end_ms,
                                   (mappings.speaker_profile_id is not null or utterances.is_target_person) as is_target_person,
                                   mappings.speaker_profile_id,
                                   row_number() over (
                                       partition by utterances.recording_id order by utterances.utterance_index
                                   ) as bounded_row
                            from utterance_segments utterances
                            left join recording_speaker_mappings mappings
                              on mappings.recording_id = utterances.recording_id
                             and mappings.speaker_cluster_id = utterances.speaker_cluster_id
                            left join speaker_profiles profiles on profiles.id = mappings.speaker_profile_id
                            where utterances.recording_id = any(cast(:recording_ids as uuid[]))
                        ),
                        statistics as (
                            select recording_id, count(*) as utterance_count,
                                   coalesce(
                                       array_agg(distinct speaker_label order by speaker_label)
                                           filter (where speaker_label is not null and btrim(speaker_label) <> ''),
                                       array[]::text[]
                                   ) as speaker_labels
                            from ranked_utterances
                            group by recording_id
                        )
                        select utterances.recording_id, utterances.utterance_index, utterances.speaker_label,
                               utterances.text, utterances.start_ms, utterances.end_ms,
                               utterances.is_target_person, utterances.speaker_profile_id,
                               statistics.utterance_count, statistics.speaker_labels
                        from ranked_utterances utterances
                        join statistics on statistics.recording_id = utterances.recording_id
                        where utterances.bounded_row <= :utterance_limit
                        order by utterances.recording_id, utterances.utterance_index
                        """
                    ),
                    {
                        "recording_ids": [str(recording_id) for recording_id in recording_ids],
                        "utterance_limit": MAX_SCOPE_UTTERANCES,
                    },
                )
                .mappings()
                .all()
            )
        utterances_by_recording: dict[UUID, list[object]] = {recording_id: [] for recording_id in recording_ids}
        for row in utterance_rows:
            utterances_by_recording[cast(UUID, row["recording_id"])].append(row)
        return [
            self._scope_evidence(index, recording, utterances_by_recording[cast(UUID, recording["id"])])
            for index, recording in enumerate(recordings, start=1)
        ]

    @staticmethod
    def _scope_evidence(index: int, recording: object, utterance_rows: list[object]) -> Evidence:
        recording_data = cast(dict[str, object], recording)
        utterances = [cast(dict[str, object], row) for row in utterance_rows]
        first_row = utterances[0] if utterances else None
        speaker_labels = [str(label) for label in cast(list[object], first_row["speaker_labels"])] if first_row is not None else []
        utterance_count = int(cast(int, first_row["utterance_count"])) if first_row is not None else 0
        untruncated_body = "\n".join(f"{item['speaker_label'] or 'Unknown Speaker'}: {item['text']}" for item in utterances)
        body = untruncated_body[:MAX_SCOPE_CHARS]
        start_ms = int(cast(int, utterances[0]["start_ms"])) if utterances else 0
        end_ms = int(cast(int, utterances[-1]["end_ms"])) if utterances else 0
        recording_id = cast(UUID, recording_data["id"])
        return Evidence(
            index=index,
            recording=EvidenceRecording(
                id=recording_id,
                title=str(recording_data["title"]),
                file_name=str(recording_data["file_name"]),
                location=cast(str | None, recording_data["location"]),
                duration_seconds=cast(int | None, recording_data["duration_seconds"]),
            ),
            chunk=EvidenceChunk(
                id=recording_id,
                text=body or "该录音暂无连续发言文本。",
                start_ms=start_ms,
                end_ms=end_ms,
                speaker_labels=speaker_labels,
                is_target_person=any(bool(item["is_target_person"]) for item in utterances),
                matched_speaker_profiles=list(
                    dict.fromkeys(
                        cast(UUID, item["speaker_profile_id"]) for item in utterances if item["speaker_profile_id"] is not None
                    )
                ),
            ),
            score=1.0,
            match_type="scope",
            facts=EvidenceFacts(
                scope_verified=True,
                speaker_count=len(speaker_labels),
                utterance_count=utterance_count,
                transcript_truncated=utterance_count > len(utterances) or len(untruncated_body) > MAX_SCOPE_CHARS,
            ),
            url=f"/recordings/{recording_id}?t={start_ms}",
        )

    def _embed(self, query: str) -> list[float]:
        encoded = self._load_model().encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        values = getattr(encoded, "tolist", lambda: None)()
        if not isinstance(values, list) or not values or not isinstance(values[0], list):
            raise RuntimeError("Embedding model returned an invalid query vector")
        return [float(cast(float | int | str, item)) for item in cast(list[object], values[0])]

    def _load_model(self) -> EmbeddingModel:
        if self._model is not None:
            return self._model
        model_path = resolve_local_snapshot(self._settings.embedding_model, self._settings.resolved_embedding_model_cache_dir)
        module = cast(SentenceTransformersModule, import_module("sentence_transformers"))
        self._model = module.SentenceTransformer(str(model_path), local_files_only=True, trust_remote_code=True)
        return self._model

    @staticmethod
    def _chunk_evidence(index: int, row: object) -> Evidence:
        data = cast(dict[str, object], row)
        return Evidence(
            index=index,
            recording=EvidenceRecording(
                id=cast(UUID, data["recording_id"]),
                title=str(data["title"]),
                file_name=str(data["file_name"]),
                location=cast(str | None, data["location"]),
                duration_seconds=cast(int | None, data["duration_seconds"]),
            ),
            chunk=EvidenceChunk(
                id=cast(UUID, data["chunk_id"]),
                text=str(data["text"]),
                start_ms=int(cast(int, data["start_ms"])),
                end_ms=int(cast(int, data["end_ms"])),
                speaker_labels=cast(list[str], data["speaker_labels"] or []),
                is_target_person=bool(data["is_target_person"]),
                matched_speaker_profiles=cast(list[UUID], data.get("matched_speaker_profile_ids") or []),
            ),
            score=float(cast(float, data["score"])),
            match_type=cast(Any, data.get("match_type", "vector")),
            url=f"/recordings/{data['recording_id']}?t={data['start_ms']}",
        )

    @staticmethod
    def _append_chunk_filters(clauses: list[str], values: dict[str, object], filters: ResolvedFilters) -> None:
        RagRetriever._append_recording_filters(clauses, values, filters, chunk_alias="chunks")
        if filters.person_names:
            values["person_patterns"] = [f"%{item}%" for item in filters.person_names]
            clauses.append(
                """exists (
                    select 1
                    from recording_speaker_mappings chunk_speakers
                    left join speaker_profiles chunk_profiles on chunk_profiles.id = chunk_speakers.speaker_profile_id
                    where chunk_speakers.recording_id = chunks.recording_id
                      and chunk_speakers.speaker_cluster_id = any(chunks.speaker_cluster_ids)
                      and (
                          chunk_speakers.display_name ilike any(cast(:person_patterns as text[]))
                          or chunk_profiles.display_name ilike any(cast(:person_patterns as text[]))
                      )
                )"""
            )
        if filters.speaker_profile_ids:
            values["speaker_profile_ids"] = [str(item) for item in filters.speaker_profile_ids]
            clauses.append(
                """exists (
                    select 1 from recording_speaker_mappings chunk_speakers
                    where chunk_speakers.recording_id = chunks.recording_id
                      and chunk_speakers.speaker_cluster_id = any(chunks.speaker_cluster_ids)
                      and chunk_speakers.speaker_profile_id = any(cast(:speaker_profile_ids as uuid[]))
                )"""
            )
        if filters.target_person_only:
            clauses.append(
                """exists (
                    select 1 from recording_speaker_mappings chunk_speakers
                    where chunk_speakers.recording_id = chunks.recording_id
                      and chunk_speakers.speaker_cluster_id = any(chunks.speaker_cluster_ids)
                      and chunk_speakers.speaker_profile_id is not null
                )"""
            )

    @staticmethod
    def _append_recording_filters(clauses: list[str], values: dict[str, object], filters: ResolvedFilters, chunk_alias: str | None = None) -> None:
        if filters.match_none:
            clauses.append("false")
            return
        if filters.recording_scope_resolved and not filters.recording_ids:
            clauses.append("false")
            return
        if filters.recording_ids:
            values["recording_ids"] = [str(item) for item in filters.recording_ids]
            field = f"{chunk_alias}.recording_id" if chunk_alias else "recordings.id"
            clauses.append(f"{field} = any(cast(:recording_ids as uuid[]))")
        if filters.recording_scope_resolved:
            return
        if filters.locations:
            values["locations"] = [f"%{item}%" for item in filters.locations]
            clauses.append("recordings.location ilike any(cast(:locations as text[]))")
        if filters.person_names:
            values["person_patterns"] = [f"%{item}%" for item in filters.person_names]
            clauses.append(
                """exists (
                    select 1
                    from recording_speaker_mappings mappings
                    left join speaker_profiles profiles on profiles.id = mappings.speaker_profile_id
                    where mappings.recording_id = recordings.id
                      and (
                          mappings.display_name ilike any(cast(:person_patterns as text[]))
                          or profiles.display_name ilike any(cast(:person_patterns as text[]))
                      )
                )"""
            )
        if filters.speaker_profile_ids and chunk_alias is None:
            values["speaker_profile_ids"] = [str(item) for item in filters.speaker_profile_ids]
            clauses.append(
                "exists (select 1 from recording_speaker_mappings mappings where mappings.recording_id = recordings.id "
                "and mappings.speaker_profile_id = any(cast(:speaker_profile_ids as uuid[])))"
            )
        if filters.target_person_only and chunk_alias is None:
            clauses.append(
                "exists (select 1 from recording_speaker_mappings mappings "
                "where mappings.recording_id = recordings.id and mappings.speaker_profile_id is not null)"
            )
        if filters.created_from:
            values["created_from"] = filters.created_from
            clauses.append("recordings.created_at >= :created_from")
        if filters.created_to:
            values["created_to"] = filters.created_to
            clauses.append("recordings.created_at < :created_to")
