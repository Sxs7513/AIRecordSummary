from __future__ import annotations

from l2_core.rag.adjudication.contracts import EvidenceOverlay
from l2_core.rag.contracts import Evidence


def apply_evidence_overlays(evidence: list[Evidence], overlays: list[EvidenceOverlay]) -> list[Evidence]:
    overlays_by_evidence: dict[tuple[int, str], list[EvidenceOverlay]] = {}
    for overlay in overlays:
        if not overlay.target_spans:
            raise ValueError(f"Evidence overlay {overlay.proposal_id} has no target spans")
        overlays_by_evidence.setdefault((overlay.evidence_index, overlay.chunk_id), []).append(overlay)

    unapplied_proposal_ids = {overlay.proposal_id for overlay in overlays}
    corrected_evidence: list[Evidence] = []
    for item in evidence:
        item_overlays = overlays_by_evidence.get((item.index, str(item.chunk.id)), [])
        if not item_overlays:
            corrected_evidence.append(item)
            continue

        replacements = [
            (span.start_char, span.end_char, overlay.original_expression, overlay.resolved_expression)
            for overlay in item_overlays
            for span in overlay.target_spans
        ]
        corrected_text = item.chunk.text
        for start_char, end_char, original_expression, resolved_expression in sorted(replacements, reverse=True):
            if corrected_text[start_char:end_char] != original_expression:
                raise ValueError(
                    "Evidence overlay target no longer matches the source text "
                    f"(evidence_index={item.index}, chunk_id={item.chunk.id}, start_char={start_char}, end_char={end_char})"
                )
            corrected_text = f"{corrected_text[:start_char]}{resolved_expression}{corrected_text[end_char:]}"
        unapplied_proposal_ids.difference_update(overlay.proposal_id for overlay in item_overlays)
        corrected_evidence.append(
            item.model_copy(update={"chunk": item.chunk.model_copy(update={"text": corrected_text})})
        )

    if unapplied_proposal_ids:
        raise ValueError(f"Evidence overlays do not match the selected evidence: {sorted(unapplied_proposal_ids)}")
    return corrected_evidence


def render_correction_notices(overlays: list[EvidenceOverlay]) -> str:
    if not overlays:
        return ""
    return "转写纠偏已应用于上述录音正文。回答时仅依据修正后的正文直接回答问题，不要在回答正文中罗列或说明纠偏过程。"
