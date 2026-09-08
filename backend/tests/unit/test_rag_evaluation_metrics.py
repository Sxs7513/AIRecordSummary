from uuid import uuid4

from l2_core.rag.contracts import EvidenceGrade
from l2_core.rag_evaluation.contracts import EvidenceAnchor, EvidenceMatch, RankedItem
from l2_core.rag_evaluation.evidence_matcher import match_ranked_item
from l2_core.rag_evaluation.metrics import grade_metrics, retrieval_metrics


def test_retrieval_metrics_measure_rank_and_unique_evidence_recall() -> None:
    first = uuid4()
    second = uuid4()
    metrics = retrieval_metrics(
        [
            EvidenceMatch(None, 0, "none"),
            EvidenceMatch(first, 3, "time_overlap"),
            EvidenceMatch(first, 3, "time_overlap"),
            EvidenceMatch(second, 2, "quote"),
        ],
        [3, 2],
    )

    assert metrics.hit_at_1 == 0
    assert metrics.hit_at_5 == 1
    assert metrics.recall_at_5 == 1
    assert abs(metrics.reciprocal_rank - 0.5) < 1e-9
    assert 0 < metrics.ndcg_at_10 < 1


def test_grade_metrics_expose_false_rejection_and_unsafe_answering() -> None:
    expected_inference = EvidenceGrade(verdict="qualified_answer")
    expected_abstention = EvidenceGrade(verdict="abstain")
    metrics = grade_metrics(
        [
            (
                expected_inference,
                EvidenceGrade(verdict="abstain"),
            ),
            (
                expected_abstention,
                EvidenceGrade(verdict="direct_answer"),
            ),
        ]
    )

    assert metrics.answerability_accuracy == 0
    assert metrics.answer_false_negative_rate == 1
    assert metrics.unsafe_answer_rate == 1


def test_grade_metrics_treat_direct_and_qualified_as_answerable() -> None:
    metrics = grade_metrics(
        [
            (
                EvidenceGrade(verdict="direct_answer"),
                EvidenceGrade(verdict="qualified_answer"),
            )
        ]
    )

    assert metrics.answerability_accuracy == 1
    assert metrics.answer_false_negative_rate == 0
    assert metrics.unsafe_answer_rate == 0


def test_evidence_matcher_prefers_exact_source_chunk() -> None:
    recording_id = uuid4()
    chunk_id = uuid4()
    anchor = EvidenceAnchor(
        id=uuid4(),
        recording_id=recording_id,
        source_chunk_id=chunk_id,
        quote="一年大概做到五千万",
        start_ms=1_000,
        end_ms=5_000,
        relevance=3,
        content_checksum="checksum",
    )
    item = RankedItem(
        recording_id=recording_id,
        source_chunk_id=chunk_id,
        text="扩展后的上下文",
        start_ms=0,
        end_ms=8_000,
        score=0.9,
    )

    match = match_ranked_item(item, [anchor])

    assert match.evidence_id == anchor.id
    assert match.relevance == 3
    assert match.kind == "checksum"


def test_evidence_matcher_uses_time_overlap_across_chunk_versions() -> None:
    recording_id = uuid4()
    anchor = EvidenceAnchor(
        id=uuid4(),
        recording_id=recording_id,
        source_chunk_id=uuid4(),
        quote="正确证据",
        start_ms=2_000,
        end_ms=6_000,
        relevance=2,
        content_checksum="checksum",
    )
    rebuilt_chunk = RankedItem(
        recording_id=recording_id,
        source_chunk_id=uuid4(),
        text="重新切块后的文本",
        start_ms=3_000,
        end_ms=7_000,
        score=0.7,
    )

    match = match_ranked_item(rebuilt_chunk, [anchor])

    assert match.evidence_id == anchor.id
    assert match.kind == "time_overlap"


def test_expanded_result_covers_every_overlapping_gold_evidence_for_recall() -> None:
    recording_id = uuid4()
    seed_chunk_id = uuid4()
    exact_anchor = EvidenceAnchor(
        id=uuid4(),
        recording_id=recording_id,
        source_chunk_id=seed_chunk_id,
        quote="第一条证据",
        start_ms=1_000,
        end_ms=3_000,
        relevance=3,
        content_checksum="first",
    )
    adjacent_anchor = EvidenceAnchor(
        id=uuid4(),
        recording_id=recording_id,
        source_chunk_id=uuid4(),
        quote="第二条证据",
        start_ms=3_000,
        end_ms=5_000,
        relevance=2,
        content_checksum="second",
    )
    expanded = RankedItem(
        recording_id=recording_id,
        source_chunk_id=seed_chunk_id,
        text="说话人：第一条证据\n说话人：第二条证据",
        start_ms=0,
        end_ms=6_000,
        score=0.9,
    )

    match = match_ranked_item(expanded, [exact_anchor, adjacent_anchor])
    metrics = retrieval_metrics([match], [3, 2])

    assert match.evidence_id == exact_anchor.id
    assert match.kind == "checksum"
    assert {item.evidence_id for item in match.all_matches()} == {exact_anchor.id, adjacent_anchor.id}
    assert metrics.hit_at_1 == 1
    assert metrics.recall_at_10 == 1
    assert metrics.reciprocal_rank == 1


def test_multi_match_recall_deduplicates_gold_covered_at_later_ranks() -> None:
    first = uuid4()
    second = uuid4()
    primary = EvidenceMatch(first, 3, "checksum")
    adjacent = EvidenceMatch(second, 2, "time_overlap")
    multi_match = EvidenceMatch(first, 3, "checksum", (primary, adjacent))

    metrics = retrieval_metrics([multi_match, EvidenceMatch(second, 2, "checksum")], [3, 2])
    metrics_without_duplicate = retrieval_metrics([multi_match], [3, 2])

    assert metrics.recall_at_5 == 1
    assert metrics.ndcg_at_10 == metrics_without_duplicate.ndcg_at_10


def test_multi_match_recall_keeps_new_coverage_when_primary_was_already_seen() -> None:
    first = EvidenceMatch(uuid4(), 3, "checksum")
    second = EvidenceMatch(uuid4(), 2, "time_overlap")
    third = EvidenceMatch(uuid4(), 1, "time_overlap")
    first_result = EvidenceMatch(first.evidence_id, first.relevance, first.kind, (first, second))
    second_result = EvidenceMatch(second.evidence_id, second.relevance, second.kind, (second, third))

    metrics = retrieval_metrics([first_result, second_result], [3, 2, 1])

    assert metrics.recall_at_5 == 1
