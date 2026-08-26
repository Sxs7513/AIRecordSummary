from uuid import uuid4

from l2_core.rag.adjudication.contracts import EvidenceOverlay, ExpressionTargetSpan
from l2_core.rag_adjudication_evaluation.runner import build_metric_rows
from l2_core.rag_adjudication_evaluation.scoring import GoldCorrection, score_corrections


def test_score_corrections_accepts_any_normalized_gold_expression() -> None:
    gold_id = uuid4()
    result = score_corrections(
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

    score = result.corrections[0]
    assert score.gold_id == gold_id
    assert score.passed is True
    assert score.match_kind == "exact"
    assert score.matched_proposal_id == "proposal-1"


def test_score_corrections_compares_the_corrected_text_when_agent_and_gold_spans_differ() -> None:
    result = score_corrections(
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

    assert result.corrections[0].passed is True
    assert result.corrections[0].match_kind == "exact"
    assert result.corrections[0].matched_proposal_id == "proposal-other"


def test_score_corrections_rejects_a_matching_expression_at_an_unrelated_span() -> None:
    result = score_corrections(
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

    assert result.corrections[0].passed is False
    assert result.predictions[0].match_kind == "unmatched"


def test_score_corrections_accepts_a_rapidfuzz_match_above_threshold() -> None:
    result = score_corrections(
        [
            GoldCorrection(
                id="gold-1",
                evidence_index=1,
                chunk_id="chunk-1",
                source_text="请使用吉泰高速总线",
                start_char=3,
                end_char=5,
                accepted_expressions=("JTAG高速",),
            )
        ],
        [
            EvidenceOverlay(
                proposal_id="proposal-1",
                evidence_index=1,
                chunk_id="chunk-1",
                original_expression="吉泰",
                resolved_expression="JTAG高速总",
                target_spans=[ExpressionTargetSpan(start_char=3, end_char=5)],
                status="auto_resolved",
                confidence=0.9,
            )
        ],
        fuzzy_threshold=80,
    )

    assert result.corrections[0].match_kind == "fuzzy"
    assert result.corrections[0].similarity is not None
    assert result.corrections[0].similarity >= 80


def test_score_corrections_rejects_a_fuzzy_match_below_threshold() -> None:
    result = score_corrections(
        [
            GoldCorrection(
                id="gold-1",
                evidence_index=1,
                chunk_id="chunk-1",
                source_text="这里是吉泰总线",
                start_char=3,
                end_char=5,
                accepted_expressions=("JTAG",),
            )
        ],
        [
            EvidenceOverlay(
                proposal_id="proposal-1",
                evidence_index=1,
                chunk_id="chunk-1",
                original_expression="吉泰",
                resolved_expression="USB",
                target_spans=[ExpressionTargetSpan(start_char=3, end_char=5)],
                status="auto_resolved",
                confidence=0.9,
            )
        ],
        fuzzy_threshold=90,
    )

    assert result.corrections[0].match_kind == "unmatched"
    assert result.predictions[0].match_kind == "unmatched"
    assert result.missed_gold_count == 0
    assert result.incorrect_prediction_count == 1


def test_score_corrections_accepts_expression_fuzzy_when_span_boundaries_and_wording_differ() -> None:
    result = score_corrections(
        [
            GoldCorrection(
                id="gold-1",
                evidence_index=1,
                chunk_id="chunk-1",
                source_text="前文成膜上也后文",
                start_char=2,
                end_char=6,
                accepted_expressions=("长波长响应",),
            )
        ],
        [
            EvidenceOverlay(
                proposal_id="proposal-2",
                evidence_index=1,
                chunk_id="chunk-1",
                original_expression="成膜上",
                resolved_expression="长波响应上",
                target_spans=[ExpressionTargetSpan(start_char=2, end_char=5)],
                status="auto_resolved",
                confidence=0.94,
            )
        ],
    )

    score = result.corrections[0]
    assert score.passed is True
    assert score.match_kind == "fuzzy"
    assert score.match_basis == "expression"
    assert score.similarity is not None
    assert score.similarity >= 80.0


def test_score_corrections_expression_fuzzy_still_requires_an_overlapping_span() -> None:
    result = score_corrections(
        [
            GoldCorrection(
                id="gold-1",
                evidence_index=1,
                chunk_id="chunk-1",
                source_text="成膜上也；另一处成膜上",
                start_char=0,
                end_char=4,
                accepted_expressions=("长波长响应",),
            )
        ],
        [
            EvidenceOverlay(
                proposal_id="proposal-2",
                evidence_index=1,
                chunk_id="chunk-1",
                original_expression="成膜上",
                resolved_expression="长波响应上",
                target_spans=[ExpressionTargetSpan(start_char=8, end_char=11)],
                status="auto_resolved",
                confidence=0.94,
            )
        ],
    )

    assert result.corrections[0].match_kind == "unmatched"
    assert result.predictions[0].match_kind == "unmatched"


def test_score_corrections_allows_one_merged_prediction_to_cover_multiple_overlapping_gold() -> None:
    source = "前一个同。d i 加就越有可能和 d r i 个碰撞"
    result = score_corrections(
        [
            GoldCorrection(
                id="gold-channel",
                evidence_index=2,
                chunk_id="chunk-2",
                source_text=source,
                start_char=3,
                end_char=4,
                accepted_expressions=("通道",),
            ),
            GoldCorrection(
                id="gold-next-index",
                evidence_index=2,
                chunk_id="chunk-2",
                source_text=source,
                start_char=5,
                end_char=10,
                accepted_expressions=("第i+1个", "第i个", "第i"),
            ),
            GoldCorrection(
                id="gold-current-index",
                evidence_index=2,
                chunk_id="chunk-2",
                source_text=source,
                start_char=16,
                end_char=24,
                accepted_expressions=("第i+1", "第i个", "第i"),
            ),
        ],
        [
            EvidenceOverlay(
                proposal_id="proposal-3",
                evidence_index=2,
                chunk_id="chunk-2",
                original_expression=source,
                resolved_expression="第 i 加 1 个通道就越有可能和第 i 个通道碰撞",
                target_spans=[ExpressionTargetSpan(start_char=0, end_char=len(source))],
                status="auto_resolved",
                confidence=0.96,
            )
        ],
    )

    assert [item.match_kind for item in result.corrections] == ["fuzzy", "fuzzy", "fuzzy"]
    assert result.predictions[0].match_kind == "fuzzy"
    assert result.predictions[0].matched_gold_ids == (
        "gold-channel",
        "gold-next-index",
        "gold-current-index",
    )
    assert result.relaxed_prediction_count == 1


def test_score_corrections_matches_each_prediction_only_once() -> None:
    result = score_corrections(
        [
            GoldCorrection(
                id="gold-1",
                evidence_index=1,
                chunk_id="chunk-1",
                source_text="吉泰总线",
                start_char=0,
                end_char=2,
                accepted_expressions=("JTAG",),
            ),
            GoldCorrection(
                id="gold-2",
                evidence_index=1,
                chunk_id="chunk-1",
                source_text="吉泰总线",
                start_char=0,
                end_char=2,
                accepted_expressions=("JTAG",),
            ),
        ],
        [
            EvidenceOverlay(
                proposal_id="proposal-1",
                evidence_index=1,
                chunk_id="chunk-1",
                original_expression="吉泰",
                resolved_expression="JTAG",
                target_spans=[ExpressionTargetSpan(start_char=0, end_char=2)],
                status="auto_resolved",
                confidence=0.9,
            )
        ],
    )

    assert result.exact_count == 1
    assert sum(item.match_kind == "unmatched" for item in result.corrections) == 1
    assert len(result.predictions) == 1


def test_score_corrections_counts_each_overlay_span_as_a_prediction() -> None:
    result = score_corrections(
        [],
        [
            EvidenceOverlay(
                proposal_id="proposal-1",
                evidence_index=1,
                chunk_id="chunk-1",
                original_expression="RF",
                resolved_expression="JTAG",
                target_spans=[
                    ExpressionTargetSpan(start_char=0, end_char=2),
                    ExpressionTargetSpan(start_char=5, end_char=7),
                ],
                status="auto_resolved",
                confidence=0.9,
            )
        ],
        source_texts={(1, "chunk-1"): "RF总线；RF模块"},
    )

    assert len(result.predictions) == 2
    assert all(item.match_kind == "unmatched" for item in result.predictions)
    assert result.missed_gold_count == 0
    assert result.incorrect_prediction_count == 2


def test_score_corrections_only_counts_gold_without_overlapping_prediction_as_missed() -> None:
    result = score_corrections(
        [
            GoldCorrection(
                id="gold-1",
                evidence_index=1,
                chunk_id="chunk-1",
                source_text="错词",
                start_char=0,
                end_char=2,
                accepted_expressions=("正词",),
            )
        ],
        [],
    )

    assert result.missed_gold_count == 1
    assert result.incorrect_prediction_count == 0


def test_metric_rows_report_strict_and_relaxed_micro_prf() -> None:
    metrics = {row["name"]: row for row in build_metric_rows(exact=2, fuzzy=1, gold=4, predictions=5)}

    assert metrics["correction_precision_strict"]["value"] == 2 / 5
    assert metrics["correction_recall_strict"]["value"] == 2 / 4
    assert metrics["correction_f1_strict"]["value"] == 4 / 9
    assert metrics["correction_precision_relaxed"]["value"] == 3 / 5
    assert metrics["correction_recall_relaxed"]["value"] == 3 / 4
    assert metrics["correction_f1_relaxed"]["value"] == 6 / 9


def test_metric_rows_use_prediction_and_gold_coverage_counts_independently() -> None:
    metrics = {
        row["name"]: row
        for row in build_metric_rows(
            exact=0,
            fuzzy=3,
            gold=3,
            predictions=1,
            strict_matched_predictions=0,
            relaxed_matched_predictions=1,
        )
    }

    assert metrics["correction_precision_relaxed"]["value"] == 1.0
    assert metrics["correction_recall_relaxed"]["value"] == 1.0
    assert metrics["correction_f1_relaxed"]["value"] == 1.0


def test_minor_gold_contributes_half_weight_to_partial_prediction_credit() -> None:
    result = score_corrections(
        [
            GoldCorrection(
                id="important-gold",
                evidence_index=1,
                chunk_id="chunk-1",
                source_text="错甲错乙",
                start_char=0,
                end_char=2,
                accepted_expressions=("对甲",),
                importance="important",
                weight=1.0,
            ),
            GoldCorrection(
                id="minor-gold",
                evidence_index=1,
                chunk_id="chunk-1",
                source_text="错甲错乙",
                start_char=2,
                end_char=4,
                accepted_expressions=("对乙",),
                importance="minor",
                weight=0.5,
            ),
        ],
        [
            EvidenceOverlay(
                proposal_id="proposal-1",
                evidence_index=1,
                chunk_id="chunk-1",
                original_expression="错甲错乙",
                resolved_expression="对甲错乙",
                target_spans=[ExpressionTargetSpan(start_char=0, end_char=4)],
                status="auto_resolved",
                confidence=0.9,
            )
        ],
    )

    assert result.corrections[0].passed is True
    assert result.corrections[1].passed is False
    assert result.gold_weight == 1.5
    assert result.exact_weight == 1.0
    assert result.predictions[0].relaxed_credit == 2 / 3

    metrics = {
        row["name"]: row
        for row in build_metric_rows(
            exact=1,
            fuzzy=0,
            gold=2,
            predictions=1,
            strict_matched_predictions=0,
            relaxed_matched_predictions=0,
            exact_weight=result.exact_weight,
            fuzzy_weight=result.fuzzy_weight,
            gold_weight=result.gold_weight,
            strict_prediction_credit=result.strict_prediction_credit,
            relaxed_prediction_credit=result.relaxed_prediction_credit,
        )
    }
    assert metrics["correction_precision_relaxed"]["value"] == 2 / 3
    assert metrics["correction_recall_relaxed"]["value"] == 2 / 3
    assert metrics["correction_f1_relaxed"]["value"] == 2 / 3
