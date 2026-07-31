from __future__ import annotations

from collections.abc import Sequence

from l2_core.rag_evaluation.contracts import EvidenceAnchor, EvidenceMatch, RankedItem


def match_ranked_item(item: RankedItem, evidence: Sequence[EvidenceAnchor]) -> EvidenceMatch:
    candidates = [anchor for anchor in evidence if anchor.recording_id == item.recording_id]
    if item.source_chunk_id is not None:
        exact = next((anchor for anchor in candidates if anchor.source_chunk_id == item.source_chunk_id), None)
        if exact is not None:
            return EvidenceMatch(exact.id, exact.relevance, "checksum")

    overlapping = [anchor for anchor in candidates if _overlap_ratio(item, anchor) >= 0.5]
    if overlapping:
        best = max(overlapping, key=lambda anchor: (anchor.relevance, _overlap_ratio(item, anchor)))
        return EvidenceMatch(best.id, best.relevance, "time_overlap")

    normalized_text = " ".join(item.text.split())
    quoted = [anchor for anchor in candidates if " ".join(anchor.quote.split()) in normalized_text]
    if quoted:
        best = max(quoted, key=lambda anchor: anchor.relevance)
        return EvidenceMatch(best.id, best.relevance, "quote")
    return EvidenceMatch(None, 0, "none")


def _overlap_ratio(item: RankedItem, anchor: EvidenceAnchor) -> float:
    duration = max(1, anchor.end_ms - anchor.start_ms)
    overlap = max(0, min(item.end_ms, anchor.end_ms) - max(item.start_ms, anchor.start_ms))
    return overlap / duration
