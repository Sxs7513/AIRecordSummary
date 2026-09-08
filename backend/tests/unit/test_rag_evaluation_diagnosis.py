from uuid import uuid4

from l2_core.rag_evaluation.contracts import EvidenceAnchor, EvidenceMatch
from l2_core.rag_evaluation.diagnosis import OperationMatches, build_evidence_journeys, summarize_journeys


def _anchor() -> EvidenceAnchor:
    return EvidenceAnchor(uuid4(), uuid4(), uuid4(), "gold quote", 1000, 2000, 3, "checksum")


def _match(anchor: EvidenceAnchor, kind: str = "checksum") -> EvidenceMatch:
    return EvidenceMatch(anchor.id, anchor.relevance, kind)


def test_parallel_base_miss_does_not_hide_lexical_hit() -> None:
    anchor = _anchor()
    journeys = build_evidence_journeys(
        [anchor],
        [
            OperationMatches("retrieve.vector.original", "succeeded", [EvidenceMatch(None, 0, "none")]),
            OperationMatches("retrieve.lexical.term", "succeeded", [_match(anchor)]),
            OperationMatches("retrieve.rrf", "succeeded", [_match(anchor)]),
            OperationMatches("retrieve.expand", "succeeded", [_match(anchor)]),
        ],
    )

    assert journeys[0]["final_covered"] is True
    assert journeys[0]["first_loss_node"] is None
    assert journeys[0]["stages"]["base_retrieval"]["status"] == "hit"  # type: ignore[index]


def test_gold_loss_is_located_after_last_visible_stage() -> None:
    anchor = _anchor()
    journeys = build_evidence_journeys(
        [anchor],
        [
            OperationMatches("retrieve.vector.original", "succeeded", [_match(anchor)]),
            OperationMatches("retrieve.rrf", "succeeded", [_match(anchor)]),
            OperationMatches("retrieve.expand", "succeeded", [_match(anchor, "time_overlap")]),
            OperationMatches("retrieve.rerank", "succeeded", []),
        ],
    )

    assert journeys[0]["final_covered"] is False
    assert journeys[0]["last_visible_node"] == "expand"
    assert journeys[0]["first_loss_node"] == "rerank"


def test_expand_can_restore_a_gold_without_reporting_a_loss() -> None:
    anchor = _anchor()
    journeys = build_evidence_journeys(
        [anchor],
        [
            OperationMatches("retrieve.vector", "succeeded", []),
            OperationMatches("retrieve.rrf", "succeeded", []),
            OperationMatches("retrieve.expand", "succeeded", [_match(anchor, "time_overlap")]),
        ],
    )

    assert journeys[0]["final_covered"] is True
    assert journeys[0]["first_loss_node"] is None


def test_expand_result_covers_multiple_gold_items() -> None:
    first = _anchor()
    second = _anchor()
    multi = EvidenceMatch(first.id, 3, "checksum", (_match(first), _match(second, "time_overlap")))

    journeys = build_evidence_journeys(
        [first, second],
        [OperationMatches("retrieve.expand", "succeeded", [multi])],
    )

    assert [item["final_covered"] for item in journeys] == [True, True]
    assert journeys[1]["stages"]["expand"]["match_kind"] == "time_overlap"  # type: ignore[index]


def test_failed_stage_is_unknown_instead_of_a_false_miss() -> None:
    anchor = _anchor()
    journeys = build_evidence_journeys(
        [anchor],
        [OperationMatches("retrieve.vector", "failed", [])],
    )

    assert journeys[0]["first_loss_node"] == "unknown"


def test_summary_counts_each_gold_independently() -> None:
    first = _anchor()
    second = _anchor()
    journeys = build_evidence_journeys(
        [first, second],
        [OperationMatches("retrieve.vector", "succeeded", [_match(first)])],
    )

    assert summarize_journeys(journeys) == {
        "diagnostic_version": "gold_node_loss_v1",
        "gold_count": 2,
        "covered_gold_count": 1,
        "uncovered_gold_count": 1,
        "unknown_gold_count": 0,
        "coverage_status": "partial_coverage",
        "first_loss_node_counts": {"base_retrieval": 1},
    }
