from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph

from l2_core.rag.contracts import Evidence, EvidenceGrade, RagGraphState, RagStateUpdate
from l2_core.rag.execution_middleware import rag_execution_middleware
from l2_core.rag.observability import started_at
from l2_core.rag.retrieval import RagRetriever
from l2_core.rag.strategies.base import StrategyResult


class ScopeSummaryStrategy:
    id: Literal["scope_summary"] = "scope_summary"
    version = "1"

    def __init__(
        self,
        retriever: RagRetriever,
        *,
        node_started: Callable[[RagGraphState, str], float],
        node_completed: Callable[..., None],
        operation_completed: Callable[..., None],
        render_evidence: Callable[[list[Evidence]], str],
    ) -> None:
        self._retriever = retriever
        self._node_started = node_started
        self._node_completed = node_completed
        self._operation_completed = operation_completed
        self._render_evidence = render_evidence
        builder = cast(Any, StateGraph(RagGraphState))
        builder.add_node(
            "retrieve_scope",
            rag_execution_middleware.wrap_node(self._retrieve, graph_name=self.id, node_name="retrieve_scope"),
        )
        builder.add_node(
            "prepare_scope",
            rag_execution_middleware.wrap_node(self._prepare, graph_name=self.id, node_name="prepare_scope"),
        )
        builder.add_node(
            "finalize",
            rag_execution_middleware.wrap_node(self._finalize, graph_name=self.id, node_name="finalize"),
        )
        builder.add_edge(START, "retrieve_scope")
        builder.add_conditional_edges(
            "retrieve_scope",
            self._after_retrieve,
            {"prepare_scope": "prepare_scope", "finalize": "finalize"},
        )
        builder.add_edge("prepare_scope", "finalize")
        builder.add_edge("finalize", END)
        self._graph: Any = builder.compile()

    async def invoke(self, state: RagGraphState) -> RagStateUpdate:
        initial_tokens = state.get("token_usage", 0)
        result = cast(RagGraphState, await self._graph.ainvoke(state))
        return {
            "evidence": result["evidence"],
            "answer_evidence": result["answer_evidence"],
            "message": result["message"],
            "grade": result["grade"],
            "planning_required": result["planning_required"],
            "answer_plan": result["answer_plan"],
            "token_usage": max(0, result.get("token_usage", 0) - initial_tokens),
            "strategy_result": result["strategy_result"],
        }

    async def _retrieve(self, state: RagGraphState) -> RagStateUpdate:
        node_started = self._node_started(state, "retrieve_scope")
        route = state["route"]
        filters = state["filters"]
        if route is None or filters is None:
            raise RuntimeError("Scope summary requires a resolved route")
        if filters.match_none:
            evidence: list[Evidence] = []
        else:
            operation_started = started_at()
            evidence = await asyncio.to_thread(
                self._retriever.retrieve_scope,
                filters,
                route.recording_limit,
                route.recording_rank,
            )
            self._operation_completed("retrieve_scope", "retrieve.scope", evidence, operation_started)
        self._node_completed(
            state,
            "retrieve_scope",
            node_started,
            outcome="succeeded" if evidence else "empty",
            strategy=self.id,
            evidence_count=len(evidence),
            recording_count=len({item.recording.id for item in evidence}),
        )
        return {
            "retrieval_candidates": [],
            "evidence": evidence,
            "answer_evidence": [],
            "message": None if evidence else "没有找到符合范围的已完成录音",
        }

    @staticmethod
    def _after_retrieve(state: RagGraphState) -> Literal["prepare_scope", "finalize"]:
        if state["execution_mode"] == "retrieval" or not state["evidence"]:
            return "finalize"
        return "prepare_scope"

    @staticmethod
    async def _prepare(state: RagGraphState) -> RagStateUpdate:
        return {
            "grade": EvidenceGrade(
                verdict="direct_answer",
                reason="scope_verified",
            )
        }

    async def _finalize(self, state: RagGraphState) -> RagStateUpdate:
        answer_evidence = state["answer_evidence"] or state["evidence"]
        ready = bool(answer_evidence)
        return {
            "strategy_result": StrategyResult(
                status="ready" if ready else "not_found",
                answer_context=self._render_evidence(answer_evidence) if ready else "",
                evidence=answer_evidence,
                sources=[item.source_payload() for item in answer_evidence],
                message=state["message"],
            )
        }
