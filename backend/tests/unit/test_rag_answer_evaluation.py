from uuid import uuid4

import pytest
from pydantic import ValidationError

from l2_core.rag_evaluation.answer_annotations import AnswerAnnotation, AnswerKeyPoint
from l2_core.rag_evaluation.answer_evaluator import (
    ANSWER_JUDGE_SYSTEM_PROMPT,
    AnswerEvaluationResult,
    ClaimEvaluation,
    KeyPointEvaluation,
    parse_generated_answer_annotation,
)
from l2_core.rag_evaluation.answer_metrics import answer_case_metrics


def test_answer_judge_schema_describes_every_output_field() -> None:
    for model in (KeyPointEvaluation, ClaimEvaluation, AnswerEvaluationResult):
        assert all(field.description for field in model.model_fields.values())


def test_answer_judge_prompt_separates_gold_and_cited_evidence_roles() -> None:
    assert "gold_evidence 是评测数据集认可的标准事实依据" in ANSWER_JUDGE_SYSTEM_PROMPT
    assert "cited_evidence 只包含 generated_answer 实际引用的证据" in ANSWER_JUDGE_SYSTEM_PROMPT
    assert "所有输出字段及枚举值均按 JSON Schema 的 description 填写" in ANSWER_JUDGE_SYSTEM_PROMPT
    assert "事实状态判定优先级" in ANSWER_JUDGE_SYSTEM_PROMPT
    assert "factual_status 与 citation_status 必须独立判断" in ANSWER_JUDGE_SYSTEM_PROMPT
    assert "必须三选一" not in ANSWER_JUDGE_SYSTEM_PROMPT

    factual_description = ClaimEvaluation.model_fields["factual_status"].description
    citation_description = ClaimEvaluation.model_fields["citation_status"].description
    correct_description = KeyPointEvaluation.model_fields["correct"].description
    assert factual_description is not None
    assert citation_description is not None
    assert correct_description is not None
    assert all(status in factual_description for status in ("supported", "unsupported", "contradicted"))
    assert all(status in citation_description for status in ("supported", "missing", "misaligned"))
    assert "不得使用 gold_evidence 补救引用" in citation_description
    assert "数字、单位、人物、否定关系或状态错误" in correct_description


def test_answerable_annotation_requires_key_points_and_gold_evidence() -> None:
    evidence_id = uuid4()
    annotation = AnswerAnnotation(
        expected_verdict="direct_answer",
        reference_answer="预算是五百万元。",
        key_points=[AnswerKeyPoint(id="budget", text="预算是五百万元", evidence_ids=[evidence_id])],
    )

    annotation.validate_case_evidence({evidence_id})

    with pytest.raises(ValueError, match="outside the case"):
        annotation.validate_case_evidence({uuid4()})


def test_abstain_annotation_can_keep_retrieved_but_insufficient_context() -> None:
    annotation = AnswerAnnotation(expected_verdict="abstain", reference_answer=None, key_points=[])

    assert annotation.reference_answer is None
    annotation.validate_case_evidence(set())
    annotation.validate_case_evidence({uuid4()})

    with pytest.raises(ValidationError, match="cannot contain a reference answer"):
        AnswerAnnotation(expected_verdict="abstain", reference_answer="没有答案", key_points=[])


def test_freeze_remaps_key_point_evidence_ids() -> None:
    draft_evidence_id = uuid4()
    frozen_evidence_id = uuid4()
    annotation = AnswerAnnotation(
        expected_verdict="qualified_answer",
        reference_answer="现有录音只能确认预算。",
        key_points=[AnswerKeyPoint(id="budget", text="预算已确认", evidence_ids=[draft_evidence_id])],
    )

    frozen = annotation.remap_evidence_ids({draft_evidence_id: frozen_evidence_id})

    assert frozen.key_points[0].evidence_ids == [frozen_evidence_id]


def test_answer_metrics_cover_correctness_citations_and_refusal_failures() -> None:
    evidence_id = uuid4()
    annotation = AnswerAnnotation(
        expected_verdict="direct_answer",
        reference_answer="预算是五百万元。",
        key_points=[AnswerKeyPoint(id="budget", text="预算是五百万元", evidence_ids=[evidence_id])],
    )
    evaluation = AnswerEvaluationResult(
        answerability_correct=True,
        key_point_results=[KeyPointEvaluation(key_point_id="budget", covered=True, correct=True, reason="supported")],
        claim_results=[
            ClaimEvaluation(
                claim="预算是五百万元",
                factual_status="supported",
                citation_status="supported",
                citation_indexes=[1],
                reason="supported",
            )
        ],
    )

    metrics = answer_case_metrics(annotation, "direct_answer", evaluation)

    assert metrics["key_point_coverage"] == 1
    assert metrics["key_point_correctness"] == 1
    assert metrics["answer_correctness"] == 1
    assert metrics["citation_support_rate"] == 1
    assert metrics["answer_score"] == 100
    assert metrics["unsafe_answer"] == 0


def test_direct_and_qualified_verdicts_share_the_same_answerability_class() -> None:
    evidence_id = uuid4()
    annotation = AnswerAnnotation(
        expected_verdict="direct_answer",
        reference_answer="预算是五百万元。",
        key_points=[AnswerKeyPoint(id="budget", text="预算是五百万元", evidence_ids=[evidence_id])],
    )
    evaluation = AnswerEvaluationResult(
        answerability_correct=True,
        key_point_results=[KeyPointEvaluation(key_point_id="budget", covered=True, correct=True, reason="supported")],
        claim_results=[
            ClaimEvaluation(
                claim="预算是五百万元",
                factual_status="supported",
                citation_status="supported",
                citation_indexes=[1],
                reason="supported",
            )
        ],
    )

    metrics = answer_case_metrics(annotation, "qualified_answer", evaluation)

    assert metrics["answerability_accuracy"] == 1
    assert metrics["answer_false_negative"] == 0
    assert metrics["unsafe_answer"] == 0
    assert metrics["answer_score"] == 100


def test_answer_metrics_mark_answering_an_unanswerable_case_as_unsafe() -> None:
    annotation = AnswerAnnotation(expected_verdict="abstain", reference_answer=None, key_points=[])
    evaluation = AnswerEvaluationResult(
        answerability_correct=False,
        key_point_results=[],
        claim_results=[
            ClaimEvaluation(
                claim="预算是五百万元",
                factual_status="unsupported",
                citation_status="missing",
                citation_indexes=[],
                reason="没有事实依据",
            )
        ],
    )

    metrics = answer_case_metrics(annotation, "direct_answer", evaluation)

    assert metrics["answerability_accuracy"] == 0
    assert metrics["unsafe_answer"] == 1
    assert metrics["answer_score"] == 0


def test_answer_without_claim_citation_evaluation_loses_citation_score() -> None:
    evidence_id = uuid4()
    annotation = AnswerAnnotation(
        expected_verdict="direct_answer",
        reference_answer="预算是五百万元。",
        key_points=[AnswerKeyPoint(id="budget", text="预算是五百万元", evidence_ids=[evidence_id])],
    )
    evaluation = AnswerEvaluationResult(
        answerability_correct=True,
        key_point_results=[KeyPointEvaluation(key_point_id="budget", covered=True, correct=True, reason="correct")],
        claim_results=[],
    )

    metrics = answer_case_metrics(annotation, "direct_answer", evaluation)

    assert metrics["citation_support_rate"] == 0
    assert metrics["answer_score"] == 75


def test_answer_score_rewards_partial_correctness_and_supported_citations() -> None:
    evidence_id = uuid4()
    annotation = AnswerAnnotation(
        expected_verdict="direct_answer",
        reference_answer="预算是五百万元，工期为三个月。",
        key_points=[
            AnswerKeyPoint(id="budget", text="预算是五百万元", evidence_ids=[evidence_id]),
            AnswerKeyPoint(id="duration", text="工期为三个月", evidence_ids=[evidence_id]),
        ],
    )
    evaluation = AnswerEvaluationResult(
        answerability_correct=True,
        key_point_results=[
            KeyPointEvaluation(key_point_id="budget", covered=True, correct=True, reason="supported"),
            KeyPointEvaluation(key_point_id="duration", covered=False, correct=False, reason="missing"),
        ],
        claim_results=[
            ClaimEvaluation(
                claim="预算是五百万元",
                factual_status="supported",
                citation_status="supported",
                citation_indexes=[1],
                reason="supported",
            ),
        ],
    )

    metrics = answer_case_metrics(annotation, "direct_answer", evaluation)

    assert metrics["key_point_correctness"] == 0.5
    assert metrics["citation_support_rate"] == 1
    assert metrics["answer_score"] == 67.5


def test_answer_score_applies_claim_and_contradiction_penalties() -> None:
    evidence_id = uuid4()
    annotation = AnswerAnnotation(
        expected_verdict="direct_answer",
        reference_answer="预算是五百万元。",
        key_points=[AnswerKeyPoint(id="budget", text="预算是五百万元", evidence_ids=[evidence_id])],
    )
    evaluation = AnswerEvaluationResult(
        answerability_correct=True,
        key_point_results=[KeyPointEvaluation(key_point_id="budget", covered=True, correct=True, reason="supported")],
        claim_results=[
            ClaimEvaluation(
                claim="项目已经交付",
                factual_status="unsupported",
                citation_status="missing",
                citation_indexes=[],
                reason="没有证据支持且没有引用",
            ),
            ClaimEvaluation(
                claim="预算是五千万元",
                factual_status="contradicted",
                citation_status="supported",
                citation_indexes=[1],
                reason="与标准预算冲突",
            ),
        ],
    )

    metrics = answer_case_metrics(annotation, "direct_answer", evaluation)

    assert metrics["answer_score"] == 42.5


def test_answer_score_is_zero_when_an_answerable_case_is_abstained() -> None:
    evidence_id = uuid4()
    annotation = AnswerAnnotation(
        expected_verdict="qualified_answer",
        reference_answer="只能确认预算。",
        key_points=[AnswerKeyPoint(id="budget", text="预算已确认", evidence_ids=[evidence_id])],
    )
    evaluation = AnswerEvaluationResult(
        answerability_correct=False,
        key_point_results=[KeyPointEvaluation(key_point_id="budget", covered=False, correct=False, reason="missing")],
        claim_results=[],
    )

    metrics = answer_case_metrics(annotation, "abstain", evaluation)

    assert metrics["answer_score"] == 0


def test_key_point_not_covered_is_normalized_to_incorrect() -> None:
    evaluation = KeyPointEvaluation(key_point_id="budget", covered=False, correct=True, reason="not mentioned")

    assert evaluation.covered is False
    assert evaluation.correct is False


def test_claim_citation_status_requires_consistent_indexes() -> None:
    with pytest.raises(ValidationError, match="missing citation cannot contain"):
        ClaimEvaluation(
            claim="预算是五百万元",
            factual_status="supported",
            citation_status="missing",
            citation_indexes=[1],
            reason="invalid",
        )

    with pytest.raises(ValidationError, match="present citation must contain"):
        ClaimEvaluation(
            claim="预算是五百万元",
            factual_status="supported",
            citation_status="misaligned",
            citation_indexes=[],
            reason="invalid",
        )


def test_answer_evaluation_rejects_duplicate_claims() -> None:
    claim = ClaimEvaluation(
        claim="预算是五百万元",
        factual_status="supported",
        citation_status="supported",
        citation_indexes=[1],
        reason="supported",
    )

    with pytest.raises(ValidationError, match="duplicate claims"):
        AnswerEvaluationResult(answerability_correct=True, key_point_results=[], claim_results=[claim, claim])


def test_generated_abstention_discards_a_model_written_refusal_message() -> None:
    annotation = parse_generated_answer_annotation('{"expected_verdict":"abstain","reference_answer":"录音中没有相关信息。","key_points":[]}')

    assert annotation.expected_verdict == "abstain"
    assert annotation.reference_answer is None
