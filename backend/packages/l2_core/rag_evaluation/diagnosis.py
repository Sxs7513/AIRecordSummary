from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from l2_core.rag_evaluation.contracts import EvidenceAnchor, EvidenceMatch

DIAGNOSTIC_VERSION = "gold_node_loss_v1"
StageStatus = Literal["hit", "miss", "skipped", "unknown"]


@dataclass(frozen=True, slots=True)
class OperationMatches:
    operation: str
    status: str
    matches: Sequence[EvidenceMatch]


def build_evidence_journeys(
    evidence: Sequence[EvidenceAnchor],
    operations: Sequence[OperationMatches],
) -> list[dict[str, object]]:
    """Build deterministic, Gold-level stage journeys from a complete in-memory trace."""
    grouped = _group_operations(operations)
    journeys: list[dict[str, object]] = []
    for anchor in evidence:
        channels = {name: _observe_stage(grouped[name], anchor.id) for name in ("vector", "lexical", "scope")}
        base = _combine_base_channels(channels)
        stages: dict[str, dict[str, object]] = {
            "base_retrieval": {**base, "channels": channels},
            "rrf": _observe_stage(grouped["rrf"], anchor.id),
            "expand": _observe_stage(grouped["expand"], anchor.id),
            "rerank": _observe_stage(grouped["rerank"], anchor.id),
        }
        enabled = [name for name in ("base_retrieval", "rrf", "expand", "rerank") if stages[name]["status"] != "skipped"]
        final_stage = enabled[-1] if enabled else None
        final_covered = final_stage is not None and stages[final_stage]["status"] == "hit"
        first_loss_node = _first_sustained_loss(stages, enabled, final_covered)
        visible = [name for name in enabled if stages[name]["status"] == "hit"]
        journeys.append(
            {
                "diagnostic_version": DIAGNOSTIC_VERSION,
                "evidence_id": str(anchor.id),
                "recording_id": str(anchor.recording_id),
                "source_chunk_id": str(anchor.source_chunk_id) if anchor.source_chunk_id else None,
                "quote": anchor.quote,
                "start_ms": anchor.start_ms,
                "end_ms": anchor.end_ms,
                "relevance": anchor.relevance,
                "final_covered": final_covered,
                "last_visible_node": visible[-1] if visible else None,
                "first_loss_node": first_loss_node,
                "stages": stages,
            }
        )
    return journeys


def summarize_journeys(journeys: Sequence[Mapping[str, object]]) -> dict[str, object]:
    counts: dict[str, int] = {}
    covered = 0
    unknown = 0
    for journey in journeys:
        if bool(journey.get("final_covered")):
            covered += 1
            continue
        node = str(journey.get("first_loss_node") or "unknown")
        counts[node] = counts.get(node, 0) + 1
        if node == "unknown":
            unknown += 1
    total = len(journeys)
    if total == 0:
        coverage_status = "not_applicable"
    elif covered == total:
        coverage_status = "full_coverage"
    elif covered:
        coverage_status = "partial_coverage"
    else:
        coverage_status = "no_hit"
    return {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "gold_count": total,
        "covered_gold_count": covered,
        "uncovered_gold_count": total - covered,
        "unknown_gold_count": unknown,
        "coverage_status": coverage_status,
        "first_loss_node_counts": counts,
    }


def _group_operations(operations: Sequence[OperationMatches]) -> dict[str, list[OperationMatches]]:
    grouped: dict[str, list[OperationMatches]] = {
        "vector": [],
        "lexical": [],
        "scope": [],
        "rrf": [],
        "expand": [],
        "rerank": [],
    }
    for operation in operations:
        name = operation.operation
        if name.startswith("retrieve.vector"):
            grouped["vector"].append(operation)
        elif name.startswith("retrieve.lexical"):
            grouped["lexical"].append(operation)
        elif name == "retrieve.scope":
            grouped["scope"].append(operation)
        elif name == "retrieve.rrf":
            grouped["rrf"].append(operation)
        elif name == "retrieve.expand":
            grouped["expand"].append(operation)
        elif name == "retrieve.rerank":
            grouped["rerank"].append(operation)
    return grouped


def _observe_stage(operations: Sequence[OperationMatches], evidence_id: UUID) -> dict[str, object]:
    if not operations:
        return {"status": "skipped", "best_rank": None}
    best: dict[str, object] | None = None
    has_failed_operation = False
    for operation in operations:
        if operation.status == "failed":
            has_failed_operation = True
        for rank, match in enumerate(operation.matches, start=1):
            covered = next((item for item in match.all_matches() if item.evidence_id == evidence_id), None)
            if covered is None:
                continue
            candidate: dict[str, object] = {
                "status": "hit",
                "best_rank": rank,
                "match_kind": covered.kind,
                "operation": operation.operation,
            }
            previous_rank = best.get("best_rank") if best is not None else None
            if best is None or (isinstance(previous_rank, int) and rank < previous_rank):
                best = candidate
    if best is not None:
        return best
    if has_failed_operation:
        return {"status": "unknown", "best_rank": None}
    return {"status": "miss", "best_rank": None}


def _combine_base_channels(channels: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    active = [item for item in channels.values() if item["status"] != "skipped"]
    if not active:
        return {"status": "skipped", "best_rank": None}
    hits = [item for item in active if item["status"] == "hit"]
    if hits:
        best = hits[0]
        for item in hits[1:]:
            item_rank = item.get("best_rank")
            best_rank = best.get("best_rank")
            if isinstance(item_rank, int) and isinstance(best_rank, int) and item_rank < best_rank:
                best = item
        return {
            "status": "hit",
            "best_rank": best["best_rank"],
            "match_kind": best.get("match_kind"),
            "operation": best.get("operation"),
        }
    if any(item["status"] == "unknown" for item in active):
        return {"status": "unknown", "best_rank": None}
    return {"status": "miss", "best_rank": None}


def _first_sustained_loss(
    stages: Mapping[str, Mapping[str, object]],
    enabled: Sequence[str],
    final_covered: bool,
) -> str | None:
    if final_covered:
        return None
    if not enabled:
        return "unknown"
    last_hit_index = -1
    for index, name in enumerate(enabled):
        if stages[name]["status"] == "hit":
            last_hit_index = index
    search_from = last_hit_index + 1
    for name in enabled[search_from:]:
        status = stages[name]["status"]
        if status == "unknown":
            return "unknown"
        if status == "miss":
            return name
    return "unknown"
