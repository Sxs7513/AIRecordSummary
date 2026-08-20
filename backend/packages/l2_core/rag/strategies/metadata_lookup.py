from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph

from l2_core.rag.contracts import EvidenceSource, MetadataSpeaker, RagGraphState, RagStateUpdate, RecordingMetadataRow
from l2_core.rag.execution_middleware import rag_execution_middleware
from l2_core.rag.observability import started_at
from l2_core.rag.retrieval import RagRetriever
from l2_core.rag.strategies.base import StrategyResult, StructuredFact, StructuredFactKey

_FACT_FIELDS: tuple[tuple[StructuredFactKey, str], ...] = (
    ("file_name", "文件名"),
    ("duration_seconds", "时长（秒）"),
    ("created_at", "上传时间"),
    ("location", "地点"),
    ("speakers", "说话人"),
)


class MetadataLookupStrategy:
    id: Literal["metadata_lookup"] = "metadata_lookup"
    version = "1"

    def __init__(
        self,
        retriever: RagRetriever,
        *,
        node_started: Callable[[RagGraphState, str], float],
        node_completed: Callable[..., None],
        operation_completed: Callable[..., None],
    ) -> None:
        self._retriever = retriever
        self._node_started = node_started
        self._node_completed = node_completed
        self._operation_completed = operation_completed
        builder = cast(Any, StateGraph(RagGraphState))
        builder.add_node(
            "load_metadata",
            rag_execution_middleware.wrap_node(self._load, graph_name=self.id, node_name="load_metadata"),
        )
        builder.add_edge(START, "load_metadata")
        builder.add_edge("load_metadata", END)
        self._graph: Any = builder.compile()

    async def invoke(self, state: RagGraphState) -> RagStateUpdate:
        result = cast(RagGraphState, await self._graph.ainvoke(state))
        return {
            "evidence": [],
            "answer_evidence": [],
            "message": result["message"],
            "grade": None,
            "planning_required": False,
            "answer_plan": None,
            "strategy_result": result["strategy_result"],
        }

    async def _load(self, state: RagGraphState) -> RagStateUpdate:
        node_started = self._node_started(state, "load_metadata")
        route = state["route"]
        filters = state["filters"]
        if route is None or filters is None:
            raise RuntimeError("Metadata lookup requires a resolved route")
        if filters.match_none:
            rows: list[RecordingMetadataRow] = []
        else:
            operation_started = started_at()
            rows = await asyncio.to_thread(
                self._retriever.retrieve_metadata,
                filters,
                route.recording_limit,
                route.recording_rank,
            )
            self._operation_completed("load_metadata", "metadata.load", rows, operation_started)
        facts = self._facts(rows)
        sources = self._sources(rows)
        ready = bool(rows)
        message = None if ready else "没有找到符合范围的已完成录音"
        self._node_completed(
            state,
            "load_metadata",
            node_started,
            outcome="succeeded" if ready else "empty",
            strategy=self.id,
            recording_count=len(rows),
            fact_count=len(facts),
        )
        return {
            "message": message,
            "strategy_result": StrategyResult(
                status="ready" if ready else "not_found",
                answer_context=self._render(rows),
                facts=facts,
                sources=sources,
                message=message,
            ),
        }

    @staticmethod
    def _facts(rows: list[RecordingMetadataRow]) -> list[StructuredFact]:
        facts: list[StructuredFact] = []
        for row in rows:
            recording_id = row["id"]
            for key, label in _FACT_FIELDS:
                value = row.get(key)
                if key == "location" and not _has_text(value):
                    continue
                facts.append(
                    StructuredFact(
                        key=key,
                        label=label,
                        value=_fact_value(value),
                        recording_id=recording_id,
                    )
                )
        return facts

    @staticmethod
    def _render(rows: list[RecordingMetadataRow]) -> str:
        blocks: list[str] = []
        for index, row in enumerate(rows, start=1):
            lines = [f"[{index}] 录音：{row['file_name']}", f"recording_id：{row['id']}"]
            for key, label in _FACT_FIELDS:
                value = row.get(key)
                if key == "location" and not _has_text(value):
                    continue
                lines.append(f"{label}：{_render_value(value)}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    @staticmethod
    def _sources(rows: list[RecordingMetadataRow]) -> list[EvidenceSource]:
        sources: list[EvidenceSource] = []
        for index, row in enumerate(rows, start=1):
            recording_id = row["id"]
            duration_seconds = row["duration_seconds"]
            sources.append(
                {
                    "index": index,
                    "recording": {
                        "id": str(recording_id),
                        "fileName": str(row["file_name"]),
                        "location": row["location"] if _has_text(row["location"]) else None,
                        "durationSeconds": duration_seconds,
                    },
                    "chunk": {
                        "id": str(recording_id),
                        "startMs": 0,
                        "endMs": (duration_seconds or 0) * 1_000,
                        "speakerLabels": _speaker_names(row["speakers"]),
                        "isTargetPerson": False,
                        "matchedSpeakerProfiles": [],
                    },
                    "score": 1.0,
                    "matchType": "scope",
                    "facts": {"scope_verified": True},
                    "url": f"/recordings/{recording_id}",
                }
            )
        return sources


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _fact_value(value: object) -> str | int | float | bool | list[str] | list[dict[str, str | int | float]] | None:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return cast(list[dict[str, str | int | float]], value)
    return cast(str | int | float | bool | None, value)


def _render_value(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        speakers = cast(list[MetadataSpeaker], value)
        return "、".join(f"{item['name']}：{item['speaking_duration_seconds']} 秒" for item in speakers) or "未知"
    return "未知" if value is None else str(value)


def _speaker_names(value: list[MetadataSpeaker]) -> list[str]:
    return [item["name"] for item in value]
