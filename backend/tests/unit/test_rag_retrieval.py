from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine

from l1_foundation.settings import Settings
from l1_foundation.worker import SyncWorkerClient
from l2_core.rag.contracts import ResolvedFilters
from l2_core.rag.normalization import normalize_search_text
from l2_core.rag.retrieval import RagRetriever


class FakeMappingsResult:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def mappings(self) -> FakeMappingsResult:
        return self

    def all(self) -> list[dict[str, object]]:
        return cast(list[dict[str, object]], self._rows)

    def scalars(self) -> Iterator[object]:
        return iter(self._rows)


class FakeConnection:
    def __init__(self, results: list[list[object]]) -> None:
        self._results = iter(results)
        self.executions: list[tuple[str, dict[str, object]]] = []

    def execute(self, statement: object, parameters: dict[str, object]) -> FakeMappingsResult:
        self.executions.append((str(statement), parameters))
        return FakeMappingsResult(next(self._results))


class FakeConnectionContext(AbstractContextManager[FakeConnection]):
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection

    def __enter__(self) -> FakeConnection:
        return self._connection

    def __exit__(self, *args: object) -> None:
        return None


class FakeEngine:
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection

    def connect(self) -> FakeConnectionContext:
        return FakeConnectionContext(self._connection)


class FakeSettings:
    rag_chunk_context_window_utterances = 2
    rag_hybrid_search_enabled = True
    rag_vector_candidate_limit = 30
    rag_lexical_candidate_limit = 30
    rag_fused_candidate_limit = 20
    rag_rrf_k = 60
    rag_original_vector_weight = 0.7
    rag_expanded_vector_weight = 0.2
    rag_lexical_weight = 0.1
    embedding_model = "Qwen/Qwen3-Embedding-4B"
    embedding_dimensions = 2560
    rag_recording_profile_search_enabled = True
    rag_recording_profile_candidate_limit = 3
    rag_recording_profile_min_score = 0.3
    rag_recording_profile_scoped_chunk_limit = 2


def _retriever(connection: FakeConnection) -> RagRetriever:
    return RagRetriever(
        cast(Engine, cast(Any, FakeEngine(connection))),
        cast(Settings, cast(Any, FakeSettings())),
        cast(SyncWorkerClient, object()),
    )


def _recording(recording_id: UUID, title: str) -> dict[str, object]:
    return {
        "id": recording_id,
        "title": title,
        "file_name": f"{title}.mp3",
        "location": None,
        "duration_seconds": 60,
        "created_at": datetime(2026, 8, 8, 9, 30, tzinfo=UTC),
    }


def test_recording_profile_retrieval_uses_recording_level_embeddings() -> None:
    recording_id = uuid4()
    created_from = datetime(2026, 8, 1, tzinfo=UTC)
    connection = FakeConnection([[{"recording_id": recording_id, "score": 0.61}]])

    rows = _retriever(connection).retrieve_recording_profile_candidates(
        [0.1, 0.2],
        ResolvedFilters(created_from=created_from),
    )

    assert rows == [{"recording_id": recording_id, "score": 0.61}]
    sql, parameters = connection.executions[0]
    assert "from recording_retrieval_documents profile_documents" in sql
    assert "recordings.created_at >= :created_from" in sql
    assert parameters["created_from"] == created_from
    assert parameters["min_score"] == 0.3


def test_recording_profile_scoped_retrieval_materializes_filtered_chunks_and_marks_locator_lane() -> None:
    recording_id = uuid4()
    chunk_id = uuid4()
    row = {
        **_recording(recording_id, "项目汇报"),
        "chunk_id": chunk_id,
        "recording_id": recording_id,
        "text": "团队介绍了项目进展。",
        "start_ms": 0,
        "end_ms": 1_000,
        "speaker_labels": ["Speaker A"],
        "is_target_person": False,
        "source_utterance_segment_ids": [],
        "metadata": {},
        "score": 0.42,
    }
    connection = FakeConnection([[row]])

    rows = _retriever(connection).retrieve_recording_profile_scoped_chunk_candidates(
        [0.1, 0.2],
        [{"recording_id": recording_id, "score": 0.71}],
    )

    assert rows[0]["chunk_id"] == chunk_id
    assert rows[0]["retrieved_via_recording_profile"] is True
    assert rows[0]["recording_profile_score"] == 0.71
    sql, parameters = connection.executions[0]
    assert "with scoped_chunks as materialized" in sql
    assert parameters["recording_ids"] == [str(recording_id)]


def test_retrieve_scope_loads_all_recording_utterances_in_one_batch_query() -> None:
    first_id = uuid4()
    second_id = uuid4()
    profile_id = uuid4()
    connection = FakeConnection(
        [
            [_recording(first_id, "第一条"), _recording(second_id, "第二条")],
            [
                {
                    "recording_id": first_id,
                    "utterance_index": 0,
                    "speaker_label": "张三",
                    "text": "第一句话",
                    "start_ms": 100,
                    "end_ms": 200,
                    "is_target_person": True,
                    "speaker_profile_id": profile_id,
                    "utterance_count": 2,
                    "speaker_labels": ["张三", "李四"],
                },
                {
                    "recording_id": first_id,
                    "utterance_index": 1,
                    "speaker_label": "李四",
                    "text": "第二句话",
                    "start_ms": 200,
                    "end_ms": 300,
                    "is_target_person": False,
                    "speaker_profile_id": None,
                    "utterance_count": 2,
                    "speaker_labels": ["张三", "李四"],
                },
            ],
        ]
    )

    evidence = _retriever(connection).retrieve_scope(ResolvedFilters(recording_ids=[first_id, second_id]), limit=2, rank=None)

    assert len(connection.executions) == 2
    assert "recordings.created_at" in connection.executions[0][0]
    assert connection.executions[1][1]["recording_ids"] == [str(first_id), str(second_id)]
    assert evidence[0].chunk.text == "张三: 第一句话\n李四: 第二句话"
    assert evidence[0].facts.utterance_count == 2
    assert evidence[0].chunk.matched_speaker_profiles == [profile_id]
    assert evidence[0].recording.created_at == datetime(2026, 8, 8, 9, 30, tzinfo=UTC)
    assert evidence[1].chunk.text == "该录音暂无连续发言文本。"
    assert evidence[1].facts.utterance_count == 0


def test_retrieve_metadata_uses_resolved_scope_and_only_selects_trusted_fields() -> None:
    recording_id = uuid4()
    created_at = datetime(2026, 8, 8, 9, 30, tzinfo=UTC)
    connection = FakeConnection(
        [
            [
                {
                    **_recording(recording_id, "产品周会"),
                    "created_at": created_at,
                    "status": "completed",
                }
            ]
        ]
    )

    rows = _retriever(connection).retrieve_metadata(
        ResolvedFilters(recording_scope_resolved=True, recording_ids=[recording_id]),
        limit=None,
        rank=None,
    )

    sql, values = connection.executions[0]
    assert rows[0]["id"] == recording_id
    assert "recordings.duration_seconds" in sql
    assert "recordings.created_at" in sql
    assert "utterance_segments" in sql
    assert "recording_speaker_mappings" in sql
    assert "coalesce(speakers.stats" in sql
    assert "speaking_duration_seconds" in sql
    assert "speaking_ratio" not in sql
    assert "recordings.id = any" in sql
    assert values["recording_ids"] == [str(recording_id)]


def test_chunk_context_expansion_loads_all_candidates_in_one_batch_query() -> None:
    recording_id = uuid4()
    first_chunk_id = uuid4()
    second_chunk_id = uuid4()
    first_source_id = uuid4()
    second_source_id = uuid4()
    profile_id = uuid4()
    connection = FakeConnection(
        [
            [
                {
                    "chunk_id": first_chunk_id,
                    "utterance_index": 0,
                    "speaker_label": "张三",
                    "text": "扩展后的第一段",
                    "start_ms": 10,
                    "end_ms": 20,
                    "is_target_person": True,
                    "speaker_profile_id": profile_id,
                },
                {
                    "chunk_id": second_chunk_id,
                    "utterance_index": 5,
                    "speaker_label": "李四",
                    "text": "扩展后的第二段",
                    "start_ms": 50,
                    "end_ms": 60,
                    "is_target_person": False,
                    "speaker_profile_id": None,
                },
            ]
        ]
    )
    rows: list[dict[str, object]] = [
        {
            "chunk_id": first_chunk_id,
            "recording_id": recording_id,
            "source_utterance_segment_ids": [first_source_id],
            "text": "原始第一段",
            "start_ms": 12,
            "end_ms": 18,
            "speaker_labels": [],
            "is_target_person": False,
        },
        {
            "chunk_id": second_chunk_id,
            "recording_id": recording_id,
            "source_utterance_segment_ids": [second_source_id],
            "text": "原始第二段",
            "start_ms": 52,
            "end_ms": 58,
            "speaker_labels": [],
            "is_target_person": False,
        },
    ]

    expanded = _retriever(connection)._expand_chunk_contexts(connection, rows)  # pyright: ignore[reportPrivateUsage]

    assert len(connection.executions) == 1
    assert connection.executions[0][1] == {
        "chunk_ids": [str(first_chunk_id), str(second_chunk_id)],
        "window": 2,
    }
    assert expanded[0]["text"] == "张三: 扩展后的第一段"
    assert expanded[0]["start_ms"] == 10
    assert expanded[0]["matched_speaker_profile_ids"] == [profile_id]
    assert expanded[1]["text"] == "李四: 扩展后的第二段"


def test_normalize_search_text_applies_nfkc_case_and_whitespace_rules() -> None:
    assert normalize_search_text("  ＡＰＩ：版本１２３，\n 发布！  ") == "api 版本123 发布"


def test_recording_scope_is_resolved_with_one_recording_query() -> None:
    recording_id = uuid4()
    connection = FakeConnection([[recording_id]])
    filters = ResolvedFilters(locations=["上海"])

    result = _retriever(connection).resolve_recording_scope(filters, limit=None, rank=None)

    assert result == [recording_id]
    sql, values = connection.executions[0]
    assert "recordings.location ilike" in sql
    assert "limit :limit" not in sql
    assert values == {"locations": ["%上海%"]}


def test_recording_scope_uses_exact_file_name_filter() -> None:
    recording_id = uuid4()
    connection = FakeConnection([[recording_id]])

    result = _retriever(connection).resolve_recording_scope(
        ResolvedFilters(file_names=["test3.m4a"]), limit=None, rank=None
    )

    assert result == [recording_id]
    sql, values = connection.executions[0]
    assert "recordings.file_name = any" in sql
    assert values["file_names"] == ["test3.m4a"]


def test_resolved_scope_reuses_ids_but_preserves_chunk_level_person_filter() -> None:
    recording_id = uuid4()
    clauses = ["recordings.status = 'completed'"]
    values: dict[str, object] = {}
    filters = ResolvedFilters(
        recording_scope_resolved=True,
        recording_ids=[recording_id],
        person_names=["张三"],
        locations=["上海"],
    )

    RagRetriever._append_chunk_filters(clauses, values, filters)  # pyright: ignore[reportPrivateUsage]

    joined = "\n".join(clauses)
    assert "chunks.recording_id = any" in joined
    assert "chunk_speakers.display_name ilike" in joined
    assert "recordings.location ilike" not in joined
    assert values["recording_ids"] == [str(recording_id)]
    assert values["person_patterns"] == ["%张三%"]


def test_lexical_candidates_apply_resolved_scope_and_chunk_filters() -> None:
    recording_id = uuid4()
    profile_id = uuid4()
    connection = FakeConnection([[]])
    filters = ResolvedFilters(
        recording_scope_resolved=True,
        recording_ids=[recording_id],
        speaker_profile_ids=[profile_id],
    )

    result = _retriever(connection).retrieve_lexical_candidates("  ＡＰＩ 版本  ", filters)

    assert result == []
    sql, values = connection.executions[0]
    assert "chunks.normalized_original_text" in sql
    assert "word_similarity(:query, coalesce" in sql
    assert "as exact_match" in sql
    assert "then 1.0" in sql
    assert "order by (position(:query in coalesce" in sql
    assert ":query <<-> coalesce" in sql
    assert "chunks.recording_id = any" in sql
    assert "chunk_speakers.speaker_profile_id" in sql
    assert values["query"] == "api 版本"
    assert values["recording_ids"] == [str(recording_id)]
    assert values["speaker_profile_ids"] == [str(profile_id)]


def test_rrf_fusion_deduplicates_and_marks_match_types() -> None:
    first_id = uuid4()
    overlap_id = uuid4()
    lexical_id = uuid4()
    retriever = _retriever(FakeConnection([]))

    result = retriever.fuse_candidates(
        [{"chunk_id": first_id, "score": 0.9}, {"chunk_id": overlap_id, "score": 0.8}],
        [{"chunk_id": overlap_id, "score": 0.7}, {"chunk_id": lexical_id, "score": 0.6}],
        limit=10,
    )

    assert [row["chunk_id"] for row in result] == [overlap_id, first_id, lexical_id]
    assert [row["match_type"] for row in result] == ["hybrid", "vector", "lexical"]
    assert float(cast(float, result[0]["score"])) > float(cast(float, result[1]["score"]))


def test_rrf_fusion_rewards_a_chunk_returned_by_multiple_query_variants() -> None:
    original_only = uuid4()
    expansion_only = uuid4()
    returned_by_both = uuid4()
    retriever = _retriever(FakeConnection([]))

    result = retriever.fuse_candidate_lists(
        [
            [{"chunk_id": original_only, "score": 0.9}, {"chunk_id": returned_by_both, "score": 0.8}],
            [{"chunk_id": expansion_only, "score": 0.9}, {"chunk_id": returned_by_both, "score": 0.8}],
        ],
        [],
        limit=10,
    )

    assert result[0]["chunk_id"] == returned_by_both
    assert [row["chunk_id"] for row in result[1:]] == [original_only, expansion_only]
    assert float(cast(float, result[1]["score"])) > float(cast(float, result[2]["score"]))


def test_rrf_normalizes_total_lexical_weight_across_query_variants() -> None:
    lexical_hit = uuid4()
    retriever = _retriever(FakeConnection([]))

    one_query = retriever.fuse_candidate_lists([], [[{"chunk_id": lexical_hit, "score": 1.0}]], limit=10)
    two_queries = retriever.fuse_candidate_lists(
        [],
        [
            [{"chunk_id": lexical_hit, "score": 1.0}],
            [{"chunk_id": lexical_hit, "score": 1.0}],
        ],
        limit=10,
    )

    assert float(cast(float, one_query[0]["score"])) == pytest.approx(float(cast(float, two_queries[0]["score"])))


def test_overlapping_expanded_contexts_are_merged_without_duplicate_lines() -> None:
    recording_id = uuid4()
    first_profile = uuid4()
    second_profile = uuid4()
    rows: list[dict[str, object]] = [
        {
            "chunk_id": uuid4(),
            "recording_id": recording_id,
            "start_ms": 100,
            "end_ms": 300,
            "text": "张三: 第一段\n李四: 重叠段",
            "speaker_labels": ["张三", "李四"],
            "matched_speaker_profile_ids": [first_profile],
            "is_target_person": True,
            "match_type": "vector",
        },
        {
            "chunk_id": uuid4(),
            "recording_id": recording_id,
            "start_ms": 250,
            "end_ms": 400,
            "text": "李四: 重叠段\n王五: 第三段",
            "speaker_labels": ["李四", "王五"],
            "matched_speaker_profile_ids": [second_profile],
            "is_target_person": False,
            "match_type": "lexical",
        },
    ]

    merged = RagRetriever._merge_overlapping_contexts(rows)  # pyright: ignore[reportPrivateUsage]

    assert len(merged) == 1
    assert merged[0]["start_ms"] == 100
    assert merged[0]["end_ms"] == 400
    assert merged[0]["text"] == "张三: 第一段\n李四: 重叠段\n王五: 第三段"
    assert merged[0]["speaker_labels"] == ["张三", "李四", "王五"]
    assert merged[0]["matched_speaker_profile_ids"] == [first_profile, second_profile]
    assert merged[0]["match_type"] == "hybrid"
