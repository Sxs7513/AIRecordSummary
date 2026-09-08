from __future__ import annotations

from l2_core.rag_evaluation.answer_annotations import AnswerAnnotation, AnswerVerdict, answerability_matches
from l2_core.rag_evaluation.answer_evaluator import AnswerEvaluationResult

KEY_POINT_CORRECTNESS_WEIGHT = 65.0
CITATION_SUPPORT_WEIGHT = 25.0
ANSWERABILITY_ACCURACY_WEIGHT = 10.0
UNSUPPORTED_CLAIM_PENALTY = 15.0
CONTRADICTION_PENALTY = 30.0


def answer_case_metrics(
    annotation: AnswerAnnotation,
    actual_verdict: AnswerVerdict,
    evaluation: AnswerEvaluationResult,
) -> dict[str, float]:
    answerability_accuracy = float(answerability_matches(annotation.expected_verdict, actual_verdict))
    unsupported_claim_count = sum(item.factual_status == "unsupported" for item in evaluation.claim_results)
    contradiction_count = sum(item.factual_status == "contradicted" for item in evaluation.claim_results)
    metrics = {
        "answerability_accuracy": answerability_accuracy,
        "answer_false_negative": float(annotation.expected_verdict != "abstain" and actual_verdict == "abstain"),
        "unsafe_answer": float(annotation.expected_verdict == "abstain" and actual_verdict != "abstain"),
        "unsupported_claim_count": float(unsupported_claim_count),
        "contradiction_count": float(contradiction_count),
    }
    key_point_correctness = 0.0
    if annotation.key_points:
        covered = sum(item.covered for item in evaluation.key_point_results)
        correct = sum(item.correct for item in evaluation.key_point_results)
        metrics["key_point_coverage"] = covered / len(annotation.key_points)
        key_point_correctness = correct / len(annotation.key_points)
        metrics["key_point_correctness"] = key_point_correctness
        metrics["answer_correctness"] = float(all(item.correct for item in evaluation.key_point_results) and contradiction_count == 0)
    citation_support_rate = 0.0
    if actual_verdict != "abstain":
        citation_support_rate = (
            sum(item.citation_status == "supported" for item in evaluation.claim_results) / len(evaluation.claim_results) if evaluation.claim_results else 0.0
        )
        metrics["citation_support_rate"] = citation_support_rate
    metrics["answer_score"] = _answer_score(
        annotation,
        actual_verdict,
        answerability_accuracy=answerability_accuracy,
        key_point_correctness=key_point_correctness,
        citation_support_rate=citation_support_rate,
        unsupported_claim_count=unsupported_claim_count,
        contradiction_count=contradiction_count,
    )
    return metrics


def _answer_score(
    annotation: AnswerAnnotation,
    actual_verdict: AnswerVerdict,
    *,
    answerability_accuracy: float,
    key_point_correctness: float,
    citation_support_rate: float,
    unsupported_claim_count: int,
    contradiction_count: int,
) -> float:
    if annotation.expected_verdict == "abstain":
        return 100.0 if actual_verdict == "abstain" else 0.0
    if actual_verdict == "abstain":
        return 0.0

    base_score = (
        key_point_correctness * KEY_POINT_CORRECTNESS_WEIGHT
        + citation_support_rate * CITATION_SUPPORT_WEIGHT
        + answerability_accuracy * ANSWERABILITY_ACCURACY_WEIGHT
    )
    penalty = unsupported_claim_count * UNSUPPORTED_CLAIM_PENALTY + contradiction_count * CONTRADICTION_PENALTY
    return max(0.0, min(100.0, base_score - penalty))
