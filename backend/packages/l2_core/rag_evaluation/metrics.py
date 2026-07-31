from __future__ import annotations

import math
from collections.abc import Sequence

from l2_core.rag_evaluation.contracts import EvidenceMatch, GradeAssessmentPair, GradeMetrics, RetrievalMetrics


def retrieval_metrics(matches: Sequence[EvidenceMatch], evidence_relevances: Sequence[int]) -> RetrievalMetrics:
    """Calculate rank metrics from one ordered result list and frozen evidence judgments."""

    matches = _deduplicate_matches(matches)
    relevant_total = max(1, len(evidence_relevances))
    return RetrievalMetrics(
        hit_at_1=_hit(matches, 1),
        hit_at_5=_hit(matches, 5),
        hit_at_10=_hit(matches, 10),
        recall_at_5=_recall(matches, relevant_total, 5),
        recall_at_10=_recall(matches, relevant_total, 10),
        recall_at_20=_recall(matches, relevant_total, 20),
        reciprocal_rank=_reciprocal_rank(matches),
        ndcg_at_10=_ndcg(matches, evidence_relevances, 10),
    )


def mean_metrics(values: Sequence[RetrievalMetrics]) -> dict[str, float]:
    if not values:
        return {name: 0.0 for name in RetrievalMetrics(0, 0, 0, 0, 0, 0, 0, 0).as_dict()}
    rows = [item.as_dict() for item in values]
    return {name: sum(row[name] for row in rows) / len(rows) for name in rows[0]}


def percentile(values: Sequence[int], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return float(ordered[min(rank, len(ordered) - 1)])


def grade_metrics(pairs: Sequence[GradeAssessmentPair]) -> GradeMetrics:
    """Evaluate the minimal evidence gate verdict."""

    if not pairs:
        return GradeMetrics(0.0, 0.0, 0.0)
    expected_answers = [pair for pair in pairs if pair[0].verdict != "abstain"]
    expected_abstentions = [pair for pair in pairs if pair[0].verdict == "abstain"]
    return GradeMetrics(
        verdict_accuracy=_ratio(
            sum(expected.verdict == predicted.verdict for expected, predicted in pairs),
            len(pairs),
        ),
        answer_false_negative_rate=_ratio(
            sum(predicted.verdict == "abstain" for _, predicted in expected_answers),
            len(expected_answers),
        ),
        unsafe_answer_rate=_ratio(
            sum(predicted.verdict != "abstain" for _, predicted in expected_abstentions),
            len(expected_abstentions),
        ),
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _hit(matches: Sequence[EvidenceMatch], limit: int) -> float:
    return float(any(item.evidence_id is not None for item in matches[:limit]))


def _recall(matches: Sequence[EvidenceMatch], total: int, limit: int) -> float:
    evidence_ids = {item.evidence_id for item in matches[:limit] if item.evidence_id is not None}
    return min(1.0, len(evidence_ids) / total)


def _reciprocal_rank(matches: Sequence[EvidenceMatch]) -> float:
    for rank, item in enumerate(matches, start=1):
        if item.evidence_id is not None:
            return 1.0 / rank
    return 0.0


def _ndcg(matches: Sequence[EvidenceMatch], evidence_relevances: Sequence[int], limit: int) -> float:
    gains = [item.relevance for item in matches[:limit]]
    dcg = _dcg(gains)
    ideal = _dcg(sorted(evidence_relevances, reverse=True)[:limit])
    return dcg / ideal if ideal > 0 else 0.0


def _dcg(relevances: Sequence[int]) -> float:
    return sum((2**relevance - 1) / math.log2(rank + 1) for rank, relevance in enumerate(relevances, start=1))


def _deduplicate_matches(matches: Sequence[EvidenceMatch]) -> list[EvidenceMatch]:
    seen: set[object] = set()
    result: list[EvidenceMatch] = []
    for item in matches:
        if item.evidence_id is None or item.evidence_id not in seen:
            result.append(item)
            if item.evidence_id is not None:
                seen.add(item.evidence_id)
        else:
            result.append(EvidenceMatch(None, 0, "none"))
    return result
