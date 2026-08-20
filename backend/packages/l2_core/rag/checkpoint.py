from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from enum import Enum
from pathlib import Path
from typing import cast
from uuid import UUID

from pydantic import BaseModel

from l1_foundation.streaming import SyncRedisStreamStore
from l2_core.rag.adjudication.contracts import AdjudicationAgentState, ClaimConfirmationDecision
from l2_core.rag.contracts import (
    AnswerPlan,
    Evidence,
    EvidenceGrade,
    JsonObject,
    JsonValue,
    RagGraphState,
    RagHistoryMessage,
    RagRoute,
    RagStateUpdate,
    ResolvedFilters,
    StrategyResult,
)

WORKFLOW_VERSION = "rag-v11"


def _string_set() -> set[str]:
    return set()


def rag_input_hash(query: str, limit: int, scope_recording_ids: list[UUID]) -> str:
    payload = json.dumps(
        {
            "query": query,
            "limit": limit,
            "scope_recording_ids": sorted(str(item) for item in scope_recording_ids),
            "workflow_version": WORKFLOW_VERSION,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class RagCheckpointStore:
    """Redis-backed completed-node snapshots containing a text-free RAG state."""

    def __init__(self, redis: SyncRedisStreamStore, ttl_seconds: int) -> None:
        self._redis_store = redis
        self._ttl_seconds = ttl_seconds

    def load_all(self, generation_id: UUID, input_hash: str) -> list[tuple[str, str, JsonObject]]:
        values = self._redis_store.get_states_by_pattern(f"generation:{generation_id}:rag-checkpoint:*")
        checkpoints: list[tuple[str, str, JsonObject]] = []
        for value in values.values():
            if (
                value.get("status") != "completed"
                or value.get("workflow_version") != WORKFLOW_VERSION
                or value.get("input_hash") != input_hash
                or not isinstance(value.get("node"), str)
                or not isinstance(value.get("completed_at"), str)
                or not isinstance(value.get("state"), dict)
            ):
                continue
            checkpoints.append(
                (
                    cast(str, value["completed_at"]),
                    cast(str, value["node"]),
                    cast(JsonObject, value["state"]),
                )
            )
        return sorted(checkpoints, key=lambda checkpoint: checkpoint[0])

    def save(self, generation_id: UUID, node: str, input_hash: str, state: RagGraphState) -> None:
        self._redis_store.set_state(
            self._key(generation_id, node),
            {
                "workflow_version": WORKFLOW_VERSION,
                "node": node,
                "status": "completed",
                "completed_at": datetime.now(UTC).isoformat(),
                "input_hash": input_hash,
                "state": _serialize_state_without_evidence_text(state),
            },
            ttl_seconds=self._ttl_seconds,
        )

    @staticmethod
    def _key(generation_id: UUID, node: str) -> str:
        node_key = hashlib.sha256(node.encode()).hexdigest()[:24]
        return f"generation:{generation_id}:rag-checkpoint:{node_key}"


@dataclass(slots=True)
class RagCheckpointSession:
    store: RagCheckpointStore
    generation_id: UUID
    source_generation_id: UUID | None
    input_hash: str
    hydrate_state: Callable[[JsonObject], JsonObject]
    rerun_nodes: set[str] = field(default_factory=_string_set)
    repeatable_nodes: set[str] = field(default_factory=_string_set)

    _completed_nodes: set[str] = field(default_factory=_string_set, init=False)

    def prepare(self) -> RagGraphState | None:
        """Load and hydrate the latest source state once before graph execution."""
        if self.source_generation_id is None:
            return None
        checkpoints = self.store.load_all(self.source_generation_id, self.input_hash)
        if not checkpoints:
            return None
        self._completed_nodes = {
            node
            for _, node, _ in checkpoints
            if node not in self.rerun_nodes and node not in self.repeatable_nodes
        }
        for _, node, state in checkpoints:
            self.store.save(self.generation_id, node, self.input_hash, cast(RagGraphState, state))
        latest_state = checkpoints[-1][2]
        hydrated = self.hydrate_state(latest_state)
        restored = _deserialize_state(hydrated)
        restored["run_id"] = str(self.generation_id)
        return restored

    def should_skip(self, node: str) -> bool:
        return node in self._completed_nodes

    def save(self, node: str, state: RagGraphState) -> None:
        self.store.save(self.generation_id, node, self.input_hash, state)
        if node not in self.repeatable_nodes:
            self._completed_nodes.add(node)


def completed_state(current: RagGraphState, output: RagStateUpdate) -> RagGraphState:
    merged = cast(RagGraphState, {**current, **output})
    if "token_usage" in output:
        merged["token_usage"] = current.get("token_usage", 0) + output["token_usage"]
    return merged


def _serialize_state_without_evidence_text(state: RagGraphState) -> JsonObject:
    serialized = cast(JsonObject, _json_value(dict(state)))
    raw_candidates = serialized.get("retrieval_candidates")
    if isinstance(raw_candidates, list):
        for candidate in raw_candidates:
            if isinstance(candidate, dict):
                candidate.pop("text", None)
    for state_field in ("evidence", "answer_evidence"):
        _strip_evidence_text(serialized.get(state_field))
    raw_strategy = serialized.get("strategy_result")
    if isinstance(raw_strategy, dict):
        raw_strategy["answer_context"] = ""
        if raw_strategy.get("corrected_answer_context") is not None:
            raw_strategy["corrected_answer_context"] = ""
        _strip_evidence_text(raw_strategy.get("evidence"))
    return serialized


def _strip_evidence_text(value: JsonValue | None) -> None:
    if not isinstance(value, list):
        return
    for item in value:
        if isinstance(item, dict) and isinstance((chunk := item.get("chunk")), dict):
            chunk.pop("text", None)


def _json_value(value: object) -> JsonValue:
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _json_value(item) for key, item in mapping.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in cast(list[object] | tuple[object, ...], value)]
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, UUID | Path):
        return str(value)
    if isinstance(value, Enum):
        return _json_value(value.value)
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


def _deserialize_state(value: JsonObject) -> RagGraphState:
    state: dict[str, object] = dict(value)
    state["history"] = [RagHistoryMessage.model_validate(item) for item in cast(list[object], state.get("history", []))]
    state["route"] = RagRoute.model_validate(state["route"]) if state.get("route") is not None else None
    state["filters"] = ResolvedFilters.model_validate(state["filters"]) if state.get("filters") is not None else None
    state["evidence"] = [Evidence.model_validate(item) for item in cast(list[object], state.get("evidence", []))]
    state["answer_evidence"] = [Evidence.model_validate(item) for item in cast(list[object], state.get("answer_evidence", []))]
    state["grade"] = EvidenceGrade.model_validate(state["grade"]) if state.get("grade") is not None else None
    state["original_grade"] = EvidenceGrade.model_validate(state["original_grade"]) if state.get("original_grade") is not None else None
    state["corrected_grade"] = EvidenceGrade.model_validate(state["corrected_grade"]) if state.get("corrected_grade") is not None else None
    state["answer_plan"] = AnswerPlan.model_validate(state["answer_plan"]) if state.get("answer_plan") is not None else None
    state["adjudication_agent_state"] = (
        AdjudicationAgentState.model_validate(state["adjudication_agent_state"])
        if state.get("adjudication_agent_state") is not None
        else None
    )
    state["adjudication_user_decision"] = (
        ClaimConfirmationDecision.model_validate(state["adjudication_user_decision"])
        if state.get("adjudication_user_decision") is not None
        else None
    )
    state["strategy_result"] = StrategyResult.model_validate(state["strategy_result"]) if state.get("strategy_result") is not None else None
    return cast(RagGraphState, state)


def render_evidence_text(evidence: list[Evidence]) -> str:
    blocks: list[str] = []
    for item in evidence:
        lines = [
            f"[{item.index}] 录音：{item.recording.title}",
            f"recording_id：{item.recording.id}",
            f"证据类型：{item.match_type}",
            f"时间：{item.chunk.start_ms}-{item.chunk.end_ms}ms",
        ]
        if item.facts.scope_verified:
            lines.append("录音范围：已由 route 和权限 filters 验证；无需正文再次证明时间排序或录音身份")
        if item.facts.speaker_count is not None:
            labels = "、".join(item.chunk.speaker_labels) or "无"
            lines.extend((f"结构化说话人标签数量：{item.facts.speaker_count}", f"结构化说话人标签：{labels}"))
        if item.facts.utterance_count is not None:
            lines.append(f"发言段总数：{item.facts.utterance_count}")
        lines.append(f"提供给模型的正文是否截断：{'是' if item.facts.transcript_truncated else '否'}")
        lines.append(f"录音正文：\n{item.chunk.text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
