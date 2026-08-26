from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from rapidfuzz import fuzz

from l2_core.rag.adjudication.contracts import EvidenceOverlay

MatchKind = Literal["exact", "fuzzy", "unmatched"]
MatchBasis = Literal["local", "expression"]


@dataclass(frozen=True, slots=True)
class GoldCorrection:
    id: object
    evidence_index: int
    chunk_id: str
    source_text: str
    start_char: int
    end_char: int
    accepted_expressions: tuple[str, ...]
    importance: Literal["important", "minor"] = "important"
    weight: float = 1.0


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
    match_basis: MatchBasis | None
    matched_accepted_expression: str | None
    actual_local_text: str | None
    expected_local_text: str | None
    importance: Literal["important", "minor"]
    gold_weight: float
    has_overlapping_prediction: bool


@dataclass(frozen=True, slots=True)
class PredictionScore:
    prediction: PredictionEdit
    matched_gold_ids: tuple[object, ...]
    match_kind: MatchKind
    similarity: float | None
    strict_credit: float
    relaxed_credit: float


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

    @property
    def strict_prediction_count(self) -> int:
        return sum(item.match_kind == "exact" for item in self.predictions)

    @property
    def relaxed_prediction_count(self) -> int:
        return sum(item.match_kind in {"exact", "fuzzy"} for item in self.predictions)

    @property
    def exact_weight(self) -> float:
        return sum(item.gold_weight for item in self.corrections if item.match_kind == "exact")

    @property
    def fuzzy_weight(self) -> float:
        return sum(item.gold_weight for item in self.corrections if item.match_kind == "fuzzy")

    @property
    def gold_weight(self) -> float:
        return sum(item.gold_weight for item in self.corrections)

    @property
    def strict_prediction_credit(self) -> float:
        return sum(item.strict_credit for item in self.predictions)

    @property
    def relaxed_prediction_credit(self) -> float:
        return sum(item.relaxed_credit for item in self.predictions)

    @property
    def missed_gold_count(self) -> int:
        return sum(not item.passed and not item.has_overlapping_prediction for item in self.corrections)

    @property
    def incorrect_prediction_count(self) -> int:
        return sum(item.relaxed_credit < 1.0 for item in self.predictions)


@dataclass(frozen=True, slots=True)
class _CandidateMatch:
    gold_index: int
    prediction_index: int
    match_kind: Literal["exact", "fuzzy"]
    similarity: float
    match_basis: MatchBasis
    accepted_expression: str
    actual_local_text: str
    expected_local_text: str
    occurrence_start: int
    occurrence_end: int


def normalize_expression(value: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().strip().split())
    return re.sub(r"(?<=[a-z0-9])\s+(?=[\u4e00-\u9fff])|(?<=[\u4e00-\u9fff])\s+(?=[a-z0-9])", "", normalized)


def score_corrections(
    gold: Sequence[GoldCorrection],
    overlays: Sequence[EvidenceOverlay],
    *,
    fuzzy_threshold: float = 90.0,
    expression_fuzzy_threshold: float = 80.0,
    source_texts: Mapping[tuple[int, str], str] | None = None,
) -> CorrectionScoringResult:
    """Score overlay spans against Gold corrections, allowing relaxed one-to-many coverage."""

    predictions = _prediction_edits(gold, overlays, source_texts or {})
    local_candidates = [
        candidate
        for gold_index, item in enumerate(gold)
        for prediction_index, prediction in enumerate(predictions)
        if (candidate := _local_candidate_match(
                gold_index,
                item,
                prediction_index,
                prediction,
                fuzzy_threshold=fuzzy_threshold,
            )) is not None
    ]
    exact_candidates = sorted(
        (item for item in local_candidates if item.match_kind == "exact"),
        key=_candidate_sort_key,
    )

    assigned_gold: set[int] = set()
    matches_by_gold: dict[int, _CandidateMatch] = {}
    for candidate in _maximum_exact_matching(exact_candidates):
        assigned_gold.add(candidate.gold_index)
        matches_by_gold[candidate.gold_index] = candidate

    fuzzy_candidates = [item for item in local_candidates if item.match_kind == "fuzzy"]
    fuzzy_candidates.extend(
        _expression_coverage_candidates(
            gold,
            predictions,
            expression_fuzzy_threshold=expression_fuzzy_threshold,
        )
    )
    used_occurrences: dict[int, list[tuple[int, int]]] = {}
    used_local_predictions: set[int] = set()
    for candidate in matches_by_gold.values():
        context = _prediction_coverage_context(gold, predictions[candidate.prediction_index])
        if context is None:
            continue
        normalized_actual, impact_start, impact_end = context
        for start, end, similarity in _fuzzy_occurrences(
            normalize_expression(candidate.accepted_expression),
            normalized_actual,
            100.0,
        ):
            if similarity == 100.0 and _spans_overlap(start, end, impact_start, impact_end):
                used_occurrences.setdefault(candidate.prediction_index, []).append((start, end))
                break
    for candidate in sorted(fuzzy_candidates, key=_candidate_sort_key):
        if candidate.gold_index in assigned_gold:
            continue
        if candidate.match_basis == "local":
            if candidate.prediction_index in used_local_predictions:
                continue
            used_local_predictions.add(candidate.prediction_index)
        else:
            intervals = used_occurrences.setdefault(candidate.prediction_index, [])
            if any(
                _spans_overlap(candidate.occurrence_start, candidate.occurrence_end, start, end)
                for start, end in intervals
            ):
                continue
            intervals.append((candidate.occurrence_start, candidate.occurrence_end))
        assigned_gold.add(candidate.gold_index)
        matches_by_gold[candidate.gold_index] = candidate

    correction_scores = tuple(_correction_score(item, matches_by_gold.get(index), predictions) for index, item in enumerate(gold))
    prediction_scores = tuple(
        _prediction_score(index, prediction, matches_by_gold, gold)
        for index, prediction in enumerate(predictions)
    )
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


def _local_candidate_match(
    gold_index: int,
    gold: GoldCorrection,
    prediction_index: int,
    prediction: PredictionEdit,
    *,
    fuzzy_threshold: float,
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
        local_similarity = float(fuzz.ratio(normalized_actual, normalized_expected))
        if normalized_actual == normalized_expected:
            kind: Literal["exact", "fuzzy"] = "exact"
            similarity = 100.0
            match_basis: MatchBasis = "local"
        elif local_similarity >= fuzzy_threshold:
            kind = "fuzzy"
            similarity = local_similarity
            match_basis = "local"
        else:
            continue
        candidate = _CandidateMatch(
            gold_index=gold_index,
            prediction_index=prediction_index,
            match_kind=kind,
            similarity=similarity,
            match_basis=match_basis,
            accepted_expression=accepted,
            actual_local_text=actual,
            expected_local_text=expected,
            occurrence_start=-1,
            occurrence_end=-1,
        )
        if best is None or (candidate.match_kind == "exact", candidate.similarity) > (best.match_kind == "exact", best.similarity):
            best = candidate
    return best


def _expression_coverage_candidates(
    gold: Sequence[GoldCorrection],
    predictions: Sequence[PredictionEdit],
    *,
    expression_fuzzy_threshold: float,
) -> list[_CandidateMatch]:
    candidates: list[_CandidateMatch] = []
    for prediction_index, prediction in enumerate(predictions):
        context = _prediction_coverage_context(gold, prediction)
        if context is None:
            continue
        normalized_actual, impact_start, impact_end = context
        actual = _coverage_actual_text(gold, prediction)
        overlapping = _overlapping_gold(gold, prediction)
        for gold_index, item in overlapping:
            for accepted in item.accepted_expressions:
                normalized_accepted = normalize_expression(accepted)
                for occurrence_start, occurrence_end, similarity in _fuzzy_occurrences(
                    normalized_accepted,
                    normalized_actual,
                    expression_fuzzy_threshold,
                ):
                    if not _spans_overlap(occurrence_start, occurrence_end, impact_start, impact_end):
                        continue
                    candidates.append(
                        _CandidateMatch(
                            gold_index=gold_index,
                            prediction_index=prediction_index,
                            match_kind="fuzzy",
                            similarity=similarity,
                            match_basis="expression",
                            accepted_expression=accepted,
                            actual_local_text=actual,
                            expected_local_text=accepted,
                            occurrence_start=occurrence_start,
                            occurrence_end=occurrence_end,
                        )
                    )
    return candidates


def _overlapping_gold(
    gold: Sequence[GoldCorrection], prediction: PredictionEdit
) -> list[tuple[int, GoldCorrection]]:
    return [
        (gold_index, item)
        for gold_index, item in enumerate(gold)
        if item.evidence_index == prediction.evidence_index
        and item.chunk_id == prediction.chunk_id
        and _spans_overlap(item.start_char, item.end_char, prediction.start_char, prediction.end_char)
    ]


def _coverage_actual_text(gold: Sequence[GoldCorrection], prediction: PredictionEdit) -> str:
    overlapping = _overlapping_gold(gold, prediction)
    window_start = min(prediction.start_char, *(item.start_char for _, item in overlapping))
    window_end = max(prediction.end_char, *(item.end_char for _, item in overlapping))
    prefix = prediction.source_text[window_start : prediction.start_char]
    suffix = prediction.source_text[prediction.end_char : window_end]
    return f"{prefix}{prediction.resolved_expression}{suffix}"


def _prediction_coverage_context(
    gold: Sequence[GoldCorrection], prediction: PredictionEdit
) -> tuple[str, int, int] | None:
    overlapping = _overlapping_gold(gold, prediction)
    if not overlapping:
        return None
    if prediction.source_text[prediction.start_char : prediction.end_char] != prediction.original_expression:
        return None
    window_start = min(prediction.start_char, *(item.start_char for _, item in overlapping))
    prefix = prediction.source_text[window_start : prediction.start_char]
    actual = _coverage_actual_text(gold, prediction)
    return (
        normalize_expression(actual),
        len(normalize_expression(prefix)),
        len(normalize_expression(f"{prefix}{prediction.resolved_expression}")),
    )


def _fuzzy_occurrences(needle: str, haystack: str, threshold: float) -> list[tuple[int, int, float]]:
    if not needle or not haystack:
        return []
    length_delta = max(2, len(needle) // 3)
    minimum_length = max(1, len(needle) - length_delta)
    maximum_length = min(len(haystack), len(needle) + length_delta)
    candidates: dict[tuple[int, int], float] = {}
    for start in range(len(haystack)):
        for length in range(minimum_length, maximum_length + 1):
            end = start + length
            if end > len(haystack):
                break
            similarity = float(fuzz.ratio(needle, haystack[start:end]))
            if similarity >= threshold:
                candidates[(start, end)] = similarity
    return [(start, end, score) for (start, end), score in candidates.items()]


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
        match_basis=match.match_basis if match else None,
        matched_accepted_expression=match.accepted_expression if match else None,
        actual_local_text=match.actual_local_text if match else None,
        expected_local_text=match.expected_local_text if match else None,
        importance=gold.importance,
        gold_weight=gold.weight,
        has_overlapping_prediction=any(
            gold.evidence_index == prediction.evidence_index
            and gold.chunk_id == prediction.chunk_id
            and _spans_overlap(gold.start_char, gold.end_char, prediction.start_char, prediction.end_char)
            for prediction in predictions
        ),
    )


def _prediction_score(
    prediction_index: int,
    prediction: PredictionEdit,
    matches_by_gold: Mapping[int, _CandidateMatch],
    gold: Sequence[GoldCorrection],
) -> PredictionScore:
    overlapping_gold_indexes = [
        gold_index
        for gold_index, item in enumerate(gold)
        if item.evidence_index == prediction.evidence_index
        and item.chunk_id == prediction.chunk_id
        and _spans_overlap(item.start_char, item.end_char, prediction.start_char, prediction.end_char)
    ]
    matches = [
        matches_by_gold[gold_index]
        for gold_index in overlapping_gold_indexes
        if gold_index in matches_by_gold and matches_by_gold[gold_index].prediction_index == prediction_index
    ]
    overlapping_weight = sum(gold[gold_index].weight for gold_index in overlapping_gold_indexes)
    strict_matched_weight = sum(gold[item.gold_index].weight for item in matches if item.match_kind == "exact")
    relaxed_matched_weight = sum(gold[item.gold_index].weight for item in matches)
    strict_credit = strict_matched_weight / overlapping_weight if overlapping_weight else 0.0
    relaxed_credit = relaxed_matched_weight / overlapping_weight if overlapping_weight else 0.0
    fully_matched = bool(overlapping_gold_indexes) and relaxed_credit == 1.0
    if fully_matched and strict_credit == 1.0:
        match_kind: MatchKind = "exact"
    elif fully_matched:
        match_kind = "fuzzy"
    else:
        match_kind = "unmatched"
    return PredictionScore(
        prediction=prediction,
        matched_gold_ids=tuple(gold[item.gold_index].id for item in matches),
        match_kind=match_kind,
        similarity=min((item.similarity for item in matches), default=None) if fully_matched else None,
        strict_credit=strict_credit,
        relaxed_credit=relaxed_credit,
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
