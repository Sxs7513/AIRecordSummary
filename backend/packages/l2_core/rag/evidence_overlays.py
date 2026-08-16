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
    lines = ["转写纠偏（回答必须采用修正表达，并明确告诉用户“将原表达修正为新表达”）："]
    lines.extend(
        f"- 证据[{overlay.evidence_index}]：将“{overlay.original_expression}”修正为“{overlay.resolved_expression}”"
        f"（{overlay.status}，confidence={overlay.confidence:.3f}）"
        for overlay in overlays
    )
    return "\n".join(lines)
