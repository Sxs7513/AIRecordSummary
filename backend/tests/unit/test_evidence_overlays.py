from typing import cast
from uuid import uuid4

import pytest

from l2_core.rag.adjudication.contracts import EvidenceOverlay, ExpressionTargetSpan
from l2_core.rag.checkpoint import _serialize_state_without_evidence_text  # pyright: ignore[reportPrivateUsage]
from l2_core.rag.contracts import Evidence, EvidenceChunk, EvidenceRecording, RagGraphState
from l2_core.rag.evidence_overlays import apply_evidence_overlays
from l2_core.rag.strategies.base import StrategyResult


def _evidence(text: str) -> Evidence:
    recording_id = uuid4()
    return Evidence(
        index=1,
        recording=EvidenceRecording(id=recording_id, title="接口评审", file_name="review.mp3"),
        chunk=EvidenceChunk(id=uuid4(), text=text, start_ms=1_000, end_ms=2_000),
        score=0.95,
        match_type="hybrid",
        url=f"/recordings/{recording_id}?t=1000",
    )


def _overlay(
    evidence: Evidence,
    *,
    proposal_id: str,
    original: str,
    resolved: str,
    spans: list[tuple[int, int]],
) -> EvidenceOverlay:
    return EvidenceOverlay(
        proposal_id=proposal_id,
        evidence_index=evidence.index,
        chunk_id=str(evidence.chunk.id),
        original_expression=original,
        resolved_expression=resolved,
        target_spans=[ExpressionTargetSpan(start_char=start, end_char=end) for start, end in spans],
        status="auto_resolved",
        confidence=0.98,
    )


def test_apply_evidence_overlays_replaces_all_spans_without_mutating_original() -> None:
    original_text = "RF 有规定，不能大于五秒。RF 超过五米也不行。"
    evidence = _evidence(original_text)
    first_rf = original_text.index("RF")
    second_rf = original_text.rindex("RF")
    five_seconds = original_text.index("五秒")
    overlays = [
        _overlay(
            evidence,
            proposal_id="proposal-rf",
            original="RF",
            resolved="I²C",
            spans=[(first_rf, first_rf + 2), (second_rf, second_rf + 2)],
        ),
        _overlay(
            evidence,
            proposal_id="proposal-delay",
            original="五秒",
            resolved="五微秒",
            spans=[(five_seconds, five_seconds + 2)],
        ),
    ]

    corrected = apply_evidence_overlays([evidence], overlays)

    assert evidence.chunk.text == original_text
    assert corrected[0].chunk.text == "I²C 有规定，不能大于五微秒。I²C 超过五米也不行。"


def test_apply_evidence_overlays_rejects_stale_span() -> None:
    evidence = _evidence("RF 有规定")
    overlay = _overlay(
        evidence,
        proposal_id="proposal-rf",
        original="RF",
        resolved="I²C",
        spans=[(1, 3)],
    )

    with pytest.raises(ValueError, match="no longer matches"):
        apply_evidence_overlays([evidence], [overlay])


def test_checkpoint_strips_both_answer_contexts_but_preserves_corrected_marker() -> None:
    state = cast(
        RagGraphState,
        {
            "retrieval_candidates": [],
            "evidence": [],
            "answer_evidence": [],
            "strategy_result": StrategyResult(
                status="ready",
                answer_context="原始证据正文",
                corrected_answer_context="修正后的证据正文",
            ),
        },
    )

    serialized = _serialize_state_without_evidence_text(state)

    assert serialized["strategy_result"]["answer_context"] == ""  # type: ignore[index]
    assert serialized["strategy_result"]["corrected_answer_context"] == ""  # type: ignore[index]
