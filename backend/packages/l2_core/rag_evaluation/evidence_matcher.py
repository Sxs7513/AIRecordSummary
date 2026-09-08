from __future__ import annotations

from collections.abc import Sequence

from l2_core.rag_evaluation.contracts import EvidenceAnchor, EvidenceMatch, RankedItem


def match_ranked_item(item: RankedItem, evidence: Sequence[EvidenceAnchor]) -> EvidenceMatch:
    candidates = [anchor for anchor in evidence if anchor.recording_id == item.recording_id]
    normalized_text = " ".join(item.text.split())
    ranked_matches: list[tuple[int, int, float, EvidenceMatch]] = []
    for anchor in candidates:
        overlap = _overlap_ratio(item, anchor)
        if item.source_chunk_id is not None and anchor.source_chunk_id == item.source_chunk_id:
            priority, kind = 3, "checksum"
        elif overlap >= 0.5:
            priority, kind = 2, "time_overlap"
        elif " ".join(anchor.quote.split()) in normalized_text:
            priority, kind = 1, "quote"
        else:
            continue
        ranked_matches.append((priority, anchor.relevance, overlap, EvidenceMatch(anchor.id, anchor.relevance, kind)))
    if not ranked_matches:
        return EvidenceMatch(None, 0, "none")
    ranked_matches.sort(key=lambda value: (value[0], value[1], value[2], str(value[3].evidence_id)), reverse=True)
    covered_matches = tuple(value[3] for value in ranked_matches)
    primary = covered_matches[0]
    return EvidenceMatch(primary.evidence_id, primary.relevance, primary.kind, covered_matches)


def _overlap_ratio(item: RankedItem, anchor: EvidenceAnchor) -> float:
    duration = max(1, anchor.end_ms - anchor.start_ms)
    overlap = max(0, min(item.end_ms, anchor.end_ms) - max(item.start_ms, anchor.start_ms))
    return overlap / duration
