from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, TypedDict, cast
from uuid import UUID

from sqlalchemy import Engine, text

from l1_foundation.settings import Settings
from l1_foundation.worker import SyncWorkerClient
from l2_core.audio_processing.worker_tasks import EmbeddingEncodeTaskResult, embedding_encode_command
from l2_core.rag.adjudication.contracts import AdjudicationAgentState
from l2_core.rag.checkpoint import render_evidence_text
from l2_core.rag.contracts import (
    Evidence,
    EvidenceChunk,
    EvidenceFacts,
    EvidenceRecording,
    JsonObject,
    JsonValue,
    RecordingMetadataRow,
    ResolvedFilters,
    RetrievalCandidateRow,
    ScopeRecordingRow,
    ScopeUtteranceRow,
)
from l2_core.rag.evidence_overlays import apply_evidence_overlays, render_correction_notices
from l2_core.rag.normalization import normalize_search_text
from l2_core.rag.observability import log_event
from l2_core.rag.worker_tasks import RerankCandidateInput, RerankResult, rerank_command

MAX_SCOPE_RECORDINGS = 50
MAX_SCOPE_UTTERANCES = 1_000
MAX_SCOPE_CHARS = 30_000
logger = logging.getLogger("rag")


@dataclass
class RankedCandidate:
    row: RetrievalCandidateRow
    vector_rank: int | None = None
    vector_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    fused_score: float = 0.0


class ExpandedUtteranceRow(TypedDict):
    chunk_id: UUID
    utterance_index: int
    speaker_label: str | None
    text: str
    start_ms: int
    end_ms: int
    is_target_person: bool
    speaker_profile_id: UUID | None


def _json_objects(value: JsonValue | None) -> list[JsonObject]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _retrieval_candidate_row(value: Mapping[Any, object]) -> RetrievalCandidateRow:
    """Mark the fixed projection returned by the candidate SQL queries as trusted."""

    return cast(RetrievalCandidateRow, dict(value))


def _expanded_utterance_row(value: Mapping[Any, object]) -> ExpandedUtteranceRow:
    """Convert the fixed context-expansion SQL projection at the database boundary."""

    return cast(ExpandedUtteranceRow, dict(value))


def _recording_metadata_row(value: Mapping[Any, object]) -> RecordingMetadataRow:
    """Convert the fixed metadata SQL projection at the database boundary."""

    return cast(RecordingMetadataRow, dict(value))


def _scope_recording_row(value: Mapping[Any, object]) -> ScopeRecordingRow:
    return cast(ScopeRecordingRow, dict(value))


def _scope_utterance_row(value: Mapping[Any, object]) -> ScopeUtteranceRow:
    return cast(ScopeUtteranceRow, dict(value))


class RagRetriever:
    def __init__(self, engine: Engine, settings: Settings, worker_client: SyncWorkerClient) -> None:
        self._engine = engine
        self._settings = settings
        self._worker_client = worker_client

    def release(self) -> None:
        return

    def hydrate_checkpoint_state(self, state: JsonObject) -> JsonObject:
        """Restore omitted candidate/evidence text from the authoritative recording tables."""

        hydrated: JsonObject = {**state}
        candidates = [dict(item) for item in _json_objects(state.get("retrieval_candidates"))]
        chunk_ids = [str(item["chunk_id"]) for item in candidates if item.get("chunk_id") is not None]
        candidate_text: dict[str, str] = {}
        if chunk_ids:
            with self._engine.connect() as connection:
                self._set_statement_timeout(connection)
                rows = connection.execute(
                    text("select id, text from recording_search_chunks where id = any(cast(:ids as uuid[]))"),
                    {"ids": chunk_ids},
                ).mappings()
                candidate_text = {str(row["id"]): str(row["text"]) for row in rows}
        for candidate in candidates:
            candidate["text"] = candidate_text.get(str(candidate.get("chunk_id")), "")
        hydrated["retrieval_candidates"] = cast(JsonValue, candidates)

        text_cache: dict[tuple[str, int, int], str] = {}
        for field in ("evidence", "answer_evidence"):
            hydrated[field] = cast(
                JsonValue,
                self._hydrate_checkpoint_evidence(_json_objects(state.get(field)), text_cache),
            )
        raw_strategy = state.get("strategy_result")
        if isinstance(raw_strategy, dict):
            strategy = raw_strategy
            hydrated_strategy: JsonObject = dict(strategy)
            evidence = self._hydrate_checkpoint_evidence(
                _json_objects(strategy.get("evidence")),
                text_cache,
            )
            hydrated_strategy["evidence"] = cast(JsonValue, evidence)
            validated_evidence = [Evidence.model_validate(item) for item in evidence]
            hydrated_strategy["answer_context"] = render_evidence_text(validated_evidence)
            if strategy.get("corrected_answer_context") is not None:
                raw_adjudication = state.get("adjudication_agent_state")
                if raw_adjudication is None:
                    raise ValueError("Corrected answer checkpoint is missing adjudication state")
                adjudication = AdjudicationAgentState.model_validate(raw_adjudication)
                corrected_evidence = apply_evidence_overlays(validated_evidence, adjudication.overlays)
                hydrated_strategy["corrected_answer_context"] = (
                    f"{render_evidence_text(corrected_evidence)}\n\n{render_correction_notices(adjudication.overlays)}"
                )
            hydrated["strategy_result"] = hydrated_strategy
        return hydrated

    def _hydrate_checkpoint_evidence(
        self,
        values: list[JsonObject],
        cache: dict[tuple[str, int, int], str],
    ) -> list[JsonObject]:
        result: list[JsonObject] = []
        for value in values:
            item = dict(value)
            recording = cast(JsonObject, item["recording"])
            chunk = dict(cast(JsonObject, item["chunk"]))
            key = (str(recording["id"]), cast(int, chunk["start_ms"]), cast(int, chunk["end_ms"]))
            if key not in cache:
                with self._engine.connect() as connection:
                    self._set_statement_timeout(connection)
                    utterances = (
                        connection.execute(
                            text(
                                """
                                select coalesce(profiles.display_name, mappings.display_name, utterances.speaker_label) as speaker_label,
                                       utterances.text
                                from utterance_segments utterances
                                left join recording_speaker_mappings mappings
                                  on mappings.recording_id = utterances.recording_id
                                 and mappings.speaker_cluster_id = utterances.speaker_cluster_id
                                left join speaker_profiles profiles on profiles.id = mappings.speaker_profile_id
                                where utterances.recording_id = :recording_id
                                  and utterances.end_ms >= :start_ms and utterances.start_ms <= :end_ms
                                order by utterances.utterance_index
                                """
                            ),
                            {"recording_id": key[0], "start_ms": key[1], "end_ms": key[2]},
                        )
                        .mappings()
                        .all()
                    )
                    cache[key] = "\n".join(f"{row['speaker_label'] or 'Unknown Speaker'}: {row['text']}" for row in utterances)
            chunk["text"] = cache[key] or "该录音暂无连续发言文本。"
            item["chunk"] = chunk
            result.append(item)
        return result

    @property
    def hybrid_search_enabled(self) -> bool:
        return self._settings.rag_hybrid_search_enabled

    @property
    def rerank_enabled(self) -> bool:
        return self._settings.rag_rerank_enabled

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
            self._set_statement_timeout(connection)
            rows = connection.execute(
                text(f"select recordings.id from recordings where {' and '.join(clauses)} order by recordings.created_at desc{pagination}"),
                values,
            ).scalars()
            return [cast(UUID, row) for row in rows]

    def resolve_ranked_recording_ids(self, filters: ResolvedFilters, limit: int | None, rank: int | None) -> list[UUID]:
        """Compatibility alias for callers migrating to the unified scope resolver."""

        return self.resolve_recording_scope(filters, limit, rank)

    def generate_query_embedding(self, topic: str) -> list[float]:
        return self.generate_query_embeddings([topic])[0]

    def generate_query_embeddings(self, topics: Sequence[str]) -> list[list[float]]:
        if not topics:
            return []
        result = self._worker_client.execute(embedding_encode_command(topics), result_type=EmbeddingEncodeTaskResult)
        if len(result.vectors) != len(topics):
            raise RuntimeError("Embedding Worker returned an invalid query vector count")
        return result.vectors

    def retrieve_vector_candidates(
        self,
        embedding: list[float],
        filters: ResolvedFilters,
        limit: int | None = None,
    ) -> list[RetrievalCandidateRow]:
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
            self._set_statement_timeout(connection)
            return [
                _retrieval_candidate_row(row)
                for row in (
                    connection.execute(
                        text(
                            f"""
                    select chunks.id as chunk_id, chunks.recording_id, chunks.text, chunks.start_ms, chunks.end_ms,
                           chunks.speaker_labels, chunks.is_target_person, chunks.source_utterance_segment_ids,
                           chunks.metadata,
                           recordings.title, recordings.file_name,
                           recordings.location, recordings.duration_seconds, recordings.created_at,
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
    ) -> list[RetrievalCandidateRow]:
        query = normalize_search_text(topic)
        if not query:
            return []
        values: dict[str, object] = {
            "query": query,
            "limit": limit or self._settings.rag_lexical_candidate_limit,
        }
        clauses = [
            "recordings.status = 'completed'",
            "(position(:query in chunks.normalized_text) > 0 or word_similarity(:query, chunks.normalized_text) > 0)",
        ]
        self._append_chunk_filters(clauses, values, filters)
        with self._engine.connect() as connection:
            self._set_statement_timeout(connection)
            return [
                _retrieval_candidate_row(row)
                for row in (
                    connection.execute(
                        text(
                            f"""
                            select chunks.id as chunk_id, chunks.recording_id, chunks.text,
                                   chunks.start_ms, chunks.end_ms, chunks.speaker_labels,
                                   chunks.is_target_person, chunks.source_utterance_segment_ids,
                                   chunks.metadata,
                                   recordings.title, recordings.file_name,
                                   recordings.location, recordings.duration_seconds, recordings.created_at,
                                   (position(:query in chunks.normalized_text) > 0) as exact_match,
                                   case
                                     when position(:query in chunks.normalized_text) > 0 then 1.0
                                     else word_similarity(:query, chunks.normalized_text)
                                   end as score
                            from recording_search_chunks chunks
                            join recordings on recordings.id = chunks.recording_id
                            where {" and ".join(clauses)}
                            order by (position(:query in chunks.normalized_text) > 0) desc,
                                     :query <<-> chunks.normalized_text
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
        vector_rows: list[RetrievalCandidateRow],
        lexical_rows: list[RetrievalCandidateRow],
        limit: int,
    ) -> list[RetrievalCandidateRow]:
        return self.fuse_candidate_lists([vector_rows], [lexical_rows], limit)

    def fuse_candidate_lists(
        self,
        vector_lists: list[list[RetrievalCandidateRow]],
        lexical_lists: list[list[RetrievalCandidateRow]],
        limit: int,
    ) -> list[RetrievalCandidateRow]:
        """Fuse each retrieval query's ranked list with RRF before applying Top-N."""

        candidates: dict[UUID, RankedCandidate] = {}
        for rows in vector_lists:
            for rank, row in enumerate(rows, start=1):
                chunk_id = row["chunk_id"]
                candidate = candidates.setdefault(chunk_id, RankedCandidate(row=row))
                candidate.vector_rank = min(candidate.vector_rank, rank) if candidate.vector_rank is not None else rank
                score = row["score"]
                candidate.vector_score = max(candidate.vector_score, score) if candidate.vector_score is not None else score
                candidate.fused_score += self._settings.rag_vector_weight / (self._settings.rag_rrf_k + rank)
        for rows in lexical_lists:
            for rank, row in enumerate(rows, start=1):
                chunk_id = row["chunk_id"]
                candidate = candidates.setdefault(chunk_id, RankedCandidate(row=row))
                candidate.lexical_rank = min(candidate.lexical_rank, rank) if candidate.lexical_rank is not None else rank
                score = row["score"]
                candidate.lexical_score = max(candidate.lexical_score, score) if candidate.lexical_score is not None else score
                candidate.fused_score += self._settings.rag_lexical_weight / (self._settings.rag_rrf_k + rank)
        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                -item.fused_score,
                min(rank for rank in (item.vector_rank, item.lexical_rank) if rank is not None),
                str(item.row["chunk_id"]),
            ),
        )
        result: list[RetrievalCandidateRow] = []
        for candidate in ordered[: min(limit, self._settings.rag_fused_candidate_limit)]:
            row = candidate.row.copy()
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

    def expand_candidates(self, rows: list[RetrievalCandidateRow]) -> list[Evidence]:
        with self._engine.connect() as connection:
            self._set_statement_timeout(connection)
            expanded = self._expand_chunk_contexts(connection, rows)
        merged = self._merge_overlapping_contexts(expanded)
        return [self._chunk_evidence(index, row) for index, row in enumerate(merged, start=1)]

    def rerank_evidence(self, query: str, evidence: list[Evidence]) -> tuple[list[Evidence], RerankResult | None]:
        if not self._settings.rag_rerank_enabled or not evidence:
            return evidence, None
        candidates = evidence[: self._settings.rag_rerank_candidate_limit]
        result = self._worker_client.execute(
            rerank_command(
                query,
                [RerankCandidateInput(candidate_id=str(item.chunk.id), text=item.chunk.retrieval_text()) for item in candidates],
                self._settings.rag_rerank_max_total_tokens,
            ),
            result_type=RerankResult,
        )
        if not result.scores:
            return [item.model_copy(update={"index": index}) for index, item in enumerate(evidence[: self._settings.rag_rerank_output_limit], start=1)], result
        evidence_by_id = {str(item.chunk.id): item for item in candidates}
        score_by_id = {item.candidate_id: item.score for item in result.scores}
        ordered = [evidence_by_id[item.candidate_id] for item in sorted(result.scores, key=lambda score: (-score.score, score.candidate_id))]
        scored_ids = set(score_by_id)
        ordered.extend(item for item in candidates if str(item.chunk.id) not in scored_ids)
        ordered.extend(evidence[len(candidates) :])
        return [
            item.model_copy(
                update={
                    "index": index,
                    "score": score_by_id.get(str(item.chunk.id), item.score),
                }
            )
            for index, item in enumerate(ordered[: self._settings.rag_rerank_output_limit], start=1)
        ], result

    @staticmethod
    def _merge_overlapping_contexts(rows: list[RetrievalCandidateRow]) -> list[RetrievalCandidateRow]:
        """Merge expanded candidates that cover the same recording time range."""

        merged: list[RetrievalCandidateRow] = []
        for row in rows:
            overlapping = next(
                (
                    item
                    for item in merged
                    if item["recording_id"] == row["recording_id"]
                    and item["start_ms"] <= row["end_ms"]
                    and row["start_ms"] <= item["end_ms"]
                ),
                None,
            )
            if overlapping is None:
                merged.append(row.copy())
                continue
            overlapping["start_ms"] = min(
                overlapping["start_ms"],
                row["start_ms"],
            )
            overlapping["end_ms"] = max(
                overlapping["end_ms"],
                row["end_ms"],
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
                        *overlapping["speaker_labels"],
                        *row["speaker_labels"],
                    ]
                )
            )
            overlapping["matched_speaker_profile_ids"] = list(
                dict.fromkeys(
                    [
                        *overlapping.get("matched_speaker_profile_ids", []),
                        *row.get("matched_speaker_profile_ids", []),
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
            return self.expand_candidates(self.retrieve_candidates(topic, filters, limit))

        started = perf_counter()
        vector_rows: list[RetrievalCandidateRow] = []
        lexical_rows: list[RetrievalCandidateRow] = []
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

    def retrieve_candidates(self, topic: str, filters: ResolvedFilters, limit: int) -> list[RetrievalCandidateRow]:
        embedding = self.generate_query_embedding(topic)
        rows = self.retrieve_vector_candidates(embedding, filters, limit)
        for row in rows:
            row["match_type"] = "vector"
        return rows

    def _expand_chunk_contexts(self, connection: Any, rows: list[RetrievalCandidateRow]) -> list[RetrievalCandidateRow]:
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
        utterances_by_chunk: dict[UUID, list[ExpandedUtteranceRow]] = {}
        for utterance in utterances:
            utterance_data = _expanded_utterance_row(utterance)
            utterances_by_chunk.setdefault(utterance_data["chunk_id"], []).append(utterance_data)
        for row in rows:
            chunk_utterances = utterances_by_chunk.get(row["chunk_id"], [])
            if not chunk_utterances:
                continue
            row["text"] = "\n".join(f"{item['speaker_label'] or 'Unknown Speaker'}: {item['text']}" for item in chunk_utterances)
            row["start_ms"] = chunk_utterances[0]["start_ms"]
            row["end_ms"] = chunk_utterances[-1]["end_ms"]
            row["speaker_labels"] = list(dict.fromkeys(str(item["speaker_label"]) for item in chunk_utterances if item["speaker_label"]))
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
            self._set_statement_timeout(connection)
            raw_recordings = (
                connection.execute(
                    text(
                        f"""select recordings.id, recordings.title, recordings.file_name, recordings.location,
                                  recordings.duration_seconds, recordings.created_at
                    from recordings where {" and ".join(clauses)} order by recordings.created_at desc limit :limit offset :offset"""
                    ),
                    values,
                )
                .mappings()
                .all()
            )
            if not raw_recordings:
                return []
            recordings = [_scope_recording_row(row) for row in raw_recordings]
            recording_ids = [recording["id"] for recording in recordings]
            raw_utterance_rows = (
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
        utterances_by_recording: dict[UUID, list[ScopeUtteranceRow]] = {recording_id: [] for recording_id in recording_ids}
        for raw_row in raw_utterance_rows:
            row = _scope_utterance_row(raw_row)
            utterances_by_recording[row["recording_id"]].append(row)
        return [
            self._scope_evidence(index, recording, utterances_by_recording[recording["id"]])
            for index, recording in enumerate(recordings, start=1)
        ]

    def retrieve_metadata(
        self,
        filters: ResolvedFilters,
        limit: int | None,
        rank: int | None,
    ) -> list[RecordingMetadataRow]:
        """Load only the trusted recording metadata exposed to metadata_lookup."""

        values: dict[str, object] = {}
        clauses = ["recordings.status = 'completed'"]
        self._append_recording_filters(clauses, values, filters)
        values["limit"] = 1 if rank else max(1, min(MAX_SCOPE_RECORDINGS, limit or MAX_SCOPE_RECORDINGS))
        values["offset"] = max(0, min(9, rank - 1)) if rank else 0
        with self._engine.connect() as connection:
            self._set_statement_timeout(connection)
            rows = (
                connection.execute(
                    text(
                        f"""
                    select recordings.id, recordings.file_name, recordings.location,
                           recordings.duration_seconds, recordings.created_at,
                           coalesce(speakers.stats, '[]'::jsonb) as speakers
                    from recordings
                    left join lateral (
                        select jsonb_agg(
                                   jsonb_build_object(
                                       'name', speaker_name,
                                       'speaking_duration_seconds', speaking_duration_seconds
                                   ) order by speaking_duration_seconds desc, speaker_name
                               ) filter (where speaker_name is not null and btrim(speaker_name) <> '') as stats
                        from (
                            select coalesce(profiles.display_name, mappings.display_name, utterances.speaker_label) as speaker_name,
                                   round(sum(greatest(0, utterances.end_ms - utterances.start_ms)) / 1000.0, 3)
                                       as speaking_duration_seconds
                            from utterance_segments utterances
                            left join recording_speaker_mappings mappings
                              on mappings.recording_id = utterances.recording_id
                             and mappings.speaker_cluster_id = utterances.speaker_cluster_id
                            left join speaker_profiles profiles on profiles.id = mappings.speaker_profile_id
                            where utterances.recording_id = recordings.id
                            group by coalesce(profiles.display_name, mappings.display_name, utterances.speaker_label)
                        ) recording_speakers
                    ) speakers on true
                    where {" and ".join(clauses)}
                    order by recordings.created_at desc
                    limit :limit offset :offset
                    """
                    ),
                    values,
                )
                .mappings()
                .all()
            )
            return [_recording_metadata_row(row) for row in rows]

    def _set_statement_timeout(self, connection: Any) -> None:
        if not hasattr(connection, "dialect"):
            return
        connection.execute(
            text("select set_config('statement_timeout', :timeout, true)"),
            {"timeout": f"{getattr(self._settings, 'rag_sql_statement_timeout_ms', 15_000)}ms"},
        )

    @staticmethod
    def _scope_evidence(index: int, recording: ScopeRecordingRow, utterances: list[ScopeUtteranceRow]) -> Evidence:
        first_row = utterances[0] if utterances else None
        speaker_labels = first_row["speaker_labels"] if first_row is not None else []
        utterance_count = first_row["utterance_count"] if first_row is not None else 0
        untruncated_body = "\n".join(f"{item['speaker_label'] or 'Unknown Speaker'}: {item['text']}" for item in utterances)
        body = untruncated_body[:MAX_SCOPE_CHARS]
        start_ms = utterances[0]["start_ms"] if utterances else 0
        end_ms = utterances[-1]["end_ms"] if utterances else 0
        recording_id = recording["id"]
        return Evidence(
            index=index,
            recording=EvidenceRecording(
                id=recording_id,
                title=recording["title"],
                file_name=recording["file_name"],
                location=recording["location"],
                duration_seconds=recording["duration_seconds"],
                created_at=recording["created_at"],
            ),
            chunk=EvidenceChunk(
                id=recording_id,
                text=body or "该录音暂无连续发言文本。",
                start_ms=start_ms,
                end_ms=end_ms,
                speaker_labels=speaker_labels,
                is_target_person=any(bool(item["is_target_person"]) for item in utterances),
                matched_speaker_profiles=list(
                    dict.fromkeys(item["speaker_profile_id"] for item in utterances if item["speaker_profile_id"] is not None)
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
            url=f"/recordings/{recording_id}?t={start_ms}&end={end_ms}",
        )

    @staticmethod
    def _chunk_evidence(index: int, data: RetrievalCandidateRow) -> Evidence:
        raw_metadata = data.get("metadata")
        metadata: dict[str, object] = (
            {key: value for key, value in cast(Mapping[object, object], raw_metadata).items() if isinstance(key, str)}
            if isinstance(raw_metadata, Mapping)
            else {}
        )
        raw_terms = metadata.get("terms")
        terms = [term for term in cast(list[object], raw_terms) if isinstance(term, str)] if isinstance(raw_terms, list) else []
        topic = metadata.get("topic")
        search_context = metadata.get("search_context")
        return Evidence(
            index=index,
            recording=EvidenceRecording(
                id=data["recording_id"],
                title=data["title"],
                file_name=data["file_name"],
                location=data["location"],
                duration_seconds=data["duration_seconds"],
                created_at=data["created_at"],
            ),
            chunk=EvidenceChunk(
                id=data["chunk_id"],
                text=data["text"],
                start_ms=data["start_ms"],
                end_ms=data["end_ms"],
                speaker_labels=data["speaker_labels"],
                is_target_person=data["is_target_person"],
                matched_speaker_profiles=data.get("matched_speaker_profile_ids", []),
                topic=topic if isinstance(topic, str) and topic else None,
                terms=terms,
                search_context=search_context if isinstance(search_context, str) and search_context else None,
            ),
            score=data["score"],
            match_type=data.get("match_type", "vector"),
            url=f"/recordings/{data['recording_id']}?t={data['start_ms']}&end={data['end_ms']}",
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
        if filters.file_names:
            values["file_names"] = filters.file_names
            clauses.append("recordings.file_name = any(cast(:file_names as text[]))")
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
