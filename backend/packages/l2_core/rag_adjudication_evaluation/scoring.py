from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from rapidfuzz import fuzz

from l2_core.rag.adjudication.contracts import EvidenceOverlay

MatchKind = Literal["exact", "fuzzy", "unmatched"]


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
class PredictionEdit:
    proposal_id: str
    evidence_index: int
    chunk_id: str
    source_text: str
    start_char: int
    end_char: int
    original_expression: str
    resolved_expression: str


@dataclass(frozen=True, slots=True)
class CorrectionScore:
    gold_id: object
    passed: bool
    matched_proposal_id: str | None
    actual_expression: str | None
    match_kind: MatchKind
    similarity: float | None
    matched_accepted_expression: str | None
    actual_local_text: str | None
    expected_local_text: str | None


@dataclass(frozen=True, slots=True)
class PredictionScore:
    prediction: PredictionEdit
    matched_gold_id: object | None
    match_kind: MatchKind
    similarity: float | None


@dataclass(frozen=True, slots=True)
class CorrectionScoringResult:
    corrections: tuple[CorrectionScore, ...]
    predictions: tuple[PredictionScore, ...]

    @property
    def exact_count(self) -> int:
        return sum(item.match_kind == "exact" for item in self.corrections)

    @property
    def fuzzy_count(self) -> int:
        return sum(item.match_kind == "fuzzy" for item in self.corrections)


@dataclass(frozen=True, slots=True)
class _CandidateMatch:
    gold_index: int
    prediction_index: int
    match_kind: Literal["exact", "fuzzy"]
    similarity: float
    accepted_expression: str
    actual_local_text: str
    expected_local_text: str


def normalize_expression(value: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().strip().split())
    return re.sub(r"(?<=[a-z0-9])\s+(?=[\u4e00-\u9fff])|(?<=[\u4e00-\u9fff])\s+(?=[a-z0-9])", "", normalized)


def score_corrections(
    gold: Sequence[GoldCorrection],
    overlays: Sequence[EvidenceOverlay],
    *,
    fuzzy_threshold: float = 90.0,
    source_texts: Mapping[tuple[int, str], str] | None = None,
) -> CorrectionScoringResult:
    """One-to-one score applied overlay spans against Gold corrections."""

    predictions = _prediction_edits(gold, overlays, source_texts or {})
    candidates = [
        candidate
        for gold_index, item in enumerate(gold)
        for prediction_index, prediction in enumerate(predictions)
        if (candidate := _candidate_match(gold_index, item, prediction_index, prediction)) is not None
    ]
    exact_candidates = sorted(
        (item for item in candidates if item.match_kind == "exact"),
        key=_candidate_sort_key,
    )
    fuzzy_candidates = sorted(
        (item for item in candidates if item.match_kind == "fuzzy" and item.similarity >= fuzzy_threshold),
        key=_candidate_sort_key,
    )

    assigned_gold: set[int] = set()
    assigned_predictions: set[int] = set()
    matches_by_gold: dict[int, _CandidateMatch] = {}
    matches_by_prediction: dict[int, _CandidateMatch] = {}
    for candidate in _maximum_exact_matching(exact_candidates):
        assigned_gold.add(candidate.gold_index)
        assigned_predictions.add(candidate.prediction_index)
        matches_by_gold[candidate.gold_index] = candidate
        matches_by_prediction[candidate.prediction_index] = candidate
    for candidate in fuzzy_candidates:
        if candidate.gold_index in assigned_gold or candidate.prediction_index in assigned_predictions:
            continue
        assigned_gold.add(candidate.gold_index)
        assigned_predictions.add(candidate.prediction_index)
        matches_by_gold[candidate.gold_index] = candidate
        matches_by_prediction[candidate.prediction_index] = candidate

    correction_scores = tuple(_correction_score(item, matches_by_gold.get(index), predictions) for index, item in enumerate(gold))
    prediction_scores = tuple(_prediction_score(index, prediction, matches_by_prediction.get(index), gold) for index, prediction in enumerate(predictions))
    return CorrectionScoringResult(corrections=correction_scores, predictions=prediction_scores)


def _prediction_edits(
    gold: Sequence[GoldCorrection],
    overlays: Sequence[EvidenceOverlay],
    source_texts: Mapping[tuple[int, str], str],
) -> list[PredictionEdit]:
    sources = {(item.evidence_index, item.chunk_id): item.source_text for item in gold}
    sources.update(source_texts)
    predictions: list[PredictionEdit] = []
    for overlay in overlays:
        source_text = sources.get((overlay.evidence_index, overlay.chunk_id), "")
        for span in overlay.target_spans:
            predictions.append(
                PredictionEdit(
                    proposal_id=overlay.proposal_id,
                    evidence_index=overlay.evidence_index,
                    chunk_id=overlay.chunk_id,
                    source_text=source_text,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    original_expression=overlay.original_expression,
                    resolved_expression=overlay.resolved_expression,
                )
            )
    return predictions


def _candidate_match(
    gold_index: int,
    gold: GoldCorrection,
    prediction_index: int,
    prediction: PredictionEdit,
) -> _CandidateMatch | None:
    if gold.evidence_index != prediction.evidence_index or gold.chunk_id != prediction.chunk_id:
        return None
    if not _spans_overlap(gold.start_char, gold.end_char, prediction.start_char, prediction.end_char):
        return None
    if prediction.source_text[prediction.start_char : prediction.end_char] != prediction.original_expression:
        return None

    best: _CandidateMatch | None = None
    for accepted in gold.accepted_expressions:
        actual, expected = _local_replacements(gold, prediction, accepted)
        normalized_actual = normalize_expression(actual)
        normalized_expected = normalize_expression(expected)
        similarity = float(fuzz.ratio(normalized_actual, normalized_expected))
        kind: Literal["exact", "fuzzy"] = "exact" if normalized_actual == normalized_expected else "fuzzy"
        candidate = _CandidateMatch(
            gold_index=gold_index,
            prediction_index=prediction_index,
            match_kind=kind,
            similarity=100.0 if kind == "exact" else similarity,
            accepted_expression=accepted,
            actual_local_text=actual,
            expected_local_text=expected,
        )
        if best is None or (candidate.match_kind == "exact", candidate.similarity) > (best.match_kind == "exact", best.similarity):
            best = candidate
    return best


def _local_replacements(gold: GoldCorrection, prediction: PredictionEdit, accepted: str) -> tuple[str, str]:
    window_start = min(gold.start_char, prediction.start_char)
    window_end = max(gold.end_char, prediction.end_char)
    source = gold.source_text
    actual = f"{source[window_start : prediction.start_char]}{prediction.resolved_expression}{source[prediction.end_char : window_end]}"
    expected = f"{source[window_start : gold.start_char]}{accepted}{source[gold.end_char : window_end]}"
    return actual, expected


def _correction_score(
    gold: GoldCorrection,
    match: _CandidateMatch | None,
    predictions: Sequence[PredictionEdit],
) -> CorrectionScore:
    prediction = predictions[match.prediction_index] if match else None
    return CorrectionScore(
        gold_id=gold.id,
        passed=match is not None,
        matched_proposal_id=prediction.proposal_id if prediction else None,
        actual_expression=prediction.resolved_expression if prediction else None,
        match_kind=match.match_kind if match else "unmatched",
        similarity=match.similarity if match else None,
        matched_accepted_expression=match.accepted_expression if match else None,
        actual_local_text=match.actual_local_text if match else None,
        expected_local_text=match.expected_local_text if match else None,
    )


def _prediction_score(
    prediction_index: int,
    prediction: PredictionEdit,
    match: _CandidateMatch | None,
    gold: Sequence[GoldCorrection],
) -> PredictionScore:
    return PredictionScore(
        prediction=prediction,
        matched_gold_id=gold[match.gold_index].id if match else None,
        match_kind=match.match_kind if match else "unmatched",
        similarity=match.similarity if match else None,
    )


def _candidate_sort_key(candidate: _CandidateMatch) -> tuple[float, int, int]:
    return (-candidate.similarity, candidate.gold_index, candidate.prediction_index)


def _maximum_exact_matching(candidates: Sequence[_CandidateMatch]) -> list[_CandidateMatch]:
    by_gold: dict[int, list[_CandidateMatch]] = {}
    for candidate in candidates:
        by_gold.setdefault(candidate.gold_index, []).append(candidate)
    matched_by_prediction: dict[int, _CandidateMatch] = {}

    def assign(gold_index: int, seen_predictions: set[int]) -> bool:
        for candidate in by_gold.get(gold_index, []):
            if candidate.prediction_index in seen_predictions:
                continue
            seen_predictions.add(candidate.prediction_index)
            existing = matched_by_prediction.get(candidate.prediction_index)
            if existing is None or assign(existing.gold_index, seen_predictions):
                matched_by_prediction[candidate.prediction_index] = candidate
                return True
        return False

    for gold_index in sorted(by_gold):
        assign(gold_index, set())
    return sorted(matched_by_prediction.values(), key=_candidate_sort_key)


def _spans_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and right_start < left_end
