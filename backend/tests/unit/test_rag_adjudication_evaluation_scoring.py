from uuid import uuid4

from l2_core.rag.adjudication.contracts import EvidenceOverlay, ExpressionTargetSpan
from l2_core.rag_adjudication_evaluation.scoring import GoldCorrection, score_corrections


def test_score_corrections_accepts_any_normalized_gold_expression() -> None:
    gold_id = uuid4()
    scores = score_corrections(
        [
            GoldCorrection(
                id=gold_id,
                evidence_index=1,
                chunk_id="chunk-1",
                source_text="xxxRFyyy",
                start_char=3,
                end_char=5,
                accepted_expressions=("I²C", "I2C", "JTAG总线"),
            )
        ],
        [
            EvidenceOverlay(
                proposal_id="proposal-1",
                evidence_index=1,
                chunk_id="chunk-1",
                original_expression="RF",
                resolved_expression="jtag总线",
                target_spans=[ExpressionTargetSpan(start_char=3, end_char=5)],
                status="auto_resolved",
                confidence=0.99,
            )
        ],
    )

    assert scores[0].gold_id == gold_id
    assert scores[0].passed is True
    assert scores[0].matched_proposal_id == "proposal-1"


def test_score_corrections_compares_the_corrected_text_when_agent_and_gold_spans_differ() -> None:
    scores = score_corrections(
        [
            GoldCorrection(
                id="gold-1",
                evidence_index=1,
                chunk_id="chunk-1",
                source_text="吉泰总线",
                start_char=0,
                end_char=2,
                accepted_expressions=("JTAG",),
            )
        ],
        [
            EvidenceOverlay(
                proposal_id="proposal-other",
                evidence_index=1,
                chunk_id="chunk-1",
                original_expression="吉泰总线",
                resolved_expression="JTAG 总线",
                target_spans=[ExpressionTargetSpan(start_char=0, end_char=4)],
                status="auto_resolved",
                confidence=0.99,
            ),
        ],
    )

    assert scores[0].passed is True
    assert scores[0].matched_proposal_id == "proposal-other"


def test_score_corrections_rejects_a_matching_expression_at_an_unrelated_span() -> None:
    scores = score_corrections(
        [
            GoldCorrection(
                id="gold-1",
                evidence_index=1,
                chunk_id="chunk-1",
                source_text="RF总线；RF模块",
                start_char=0,
                end_char=2,
                accepted_expressions=("JTAG",),
            )
        ],
        [
            EvidenceOverlay(
                proposal_id="proposal-other",
                evidence_index=1,
                chunk_id="chunk-1",
                original_expression="RF",
                resolved_expression="JTAG",
                target_spans=[ExpressionTargetSpan(start_char=5, end_char=7)],
                status="auto_resolved",
                confidence=0.99,
            )
        ],
    )

    assert scores[0].passed is False
