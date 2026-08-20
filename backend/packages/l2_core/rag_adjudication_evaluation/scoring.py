from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from l2_core.rag.adjudication.contracts import EvidenceOverlay


@dataclass(frozen=True, slots=True)
class GoldCorrection:
    id: object
    evidence_index: int
    chunk_id: str
    source_text: str
    start_char: int
    end_char: int
    accepted_expressions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CorrectionScore:
    gold_id: object
    passed: bool
    matched_proposal_id: str | None
    actual_expression: str | None


def normalize_expression(value: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().strip().split())
    return re.sub(r"(?<=[a-z0-9])\s+(?=[\u4e00-\u9fff])|(?<=[\u4e00-\u9fff])\s+(?=[a-z0-9])", "", normalized)


def score_corrections(gold: Sequence[GoldCorrection], overlays: Sequence[EvidenceOverlay]) -> list[CorrectionScore]:
    """Score whether an overlapping agent replacement produces a Gold-accepted corrected text."""

    scores: list[CorrectionScore] = []
    for item in gold:
        expected_texts = {
            normalize_expression(_replace(item.source_text, item.start_char, item.end_char, accepted))
            for accepted in item.accepted_expressions
        }
        match = next(
            (
                (overlay, span)
                for overlay in overlays
                if overlay.evidence_index == item.evidence_index
                and overlay.chunk_id == item.chunk_id
                for span in overlay.target_spans
                if _spans_overlap(item.start_char, item.end_char, span.start_char, span.end_char)
                and item.source_text[span.start_char : span.end_char] == overlay.original_expression
                and normalize_expression(
                    _replace(item.source_text, span.start_char, span.end_char, overlay.resolved_expression)
                ) in expected_texts
            ),
            None,
        )
        scores.append(
            CorrectionScore(
                gold_id=item.id,
                passed=match is not None,
                matched_proposal_id=match[0].proposal_id if match else None,
                actual_expression=match[0].resolved_expression if match else None,
            )
        )
    return scores


def _spans_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and right_start < left_end


def _replace(source_text: str, start_char: int, end_char: int, replacement: str) -> str:
    return f"{source_text[:start_char]}{replacement}{source_text[end_char:]}"
