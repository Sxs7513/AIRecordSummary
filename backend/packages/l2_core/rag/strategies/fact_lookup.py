from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph

from l2_core.rag.contracts import RagGraphState
from l2_core.rag.evidence_overlays import apply_evidence_overlays, render_correction_notices
from l2_core.rag.execution_middleware import rag_execution_middleware
from l2_core.rag.strategies.base import StrategyResult
from l2_core.rag.workflows.chunk_evidence import ChunkEvidencePipeline

Node = Callable[[RagGraphState], object]


class FactLookupStrategy:
    id: Literal["fact_lookup"] = "fact_lookup"
    version = "1"

    def __init__(
        self,
        chunk_evidence_pipeline: ChunkEvidencePipeline,
        *,
        expand_retrieval_terms: Node,
        grade: Node,
        classify_query_correction_risk: Node,
        adjudication_agent: Node,
        apply_user_adjudication_decision: Node,
        adjudication_enabled: bool,
        after_grade: Callable[[RagGraphState], str],
        transition: Callable[[RagGraphState, str, str, str], None],
        rerank_enabled: bool,
        query_term_expansion_enabled: bool,
        render_evidence: Callable[[list[Any]], str],
    ) -> None:
        self._chunk_evidence_pipeline = chunk_evidence_pipeline
        self._render_evidence = render_evidence
        self._transition = transition
        self._rerank_enabled = rerank_enabled
        builder = cast(Any, StateGraph(RagGraphState))
        if adjudication_enabled:
            builder.add_node(
                "classify_query_correction_risk",
                rag_execution_middleware.wrap_node(
                    classify_query_correction_risk,
                    graph_name=self.id,
                    node_name="classify_query_correction_risk",
                ),
            )
        if query_term_expansion_enabled:
            builder.add_node(
                "expand_retrieval_terms",
                rag_execution_middleware.wrap_node(
                    expand_retrieval_terms,
                    graph_name=self.id,
                    node_name="expand_retrieval_terms",
                ),
            )
        builder.add_node(
            "chunk_evidence",
            rag_execution_middleware.wrap_node(self._acquire, graph_name=self.id, node_name="chunk_evidence"),
        )
        builder.add_node(
            "grade",
            rag_execution_middleware.wrap_node(grade, graph_name=self.id, node_name="grade"),
        )
        builder.add_node(
            "finalize",
            rag_execution_middleware.wrap_node(self._finalize, graph_name=self.id, node_name="finalize"),
        )
        if adjudication_enabled:
            builder.add_node(
                "adjudication_agent",
                rag_execution_middleware.wrap_node(
                    adjudication_agent,
                    graph_name=self.id,
                    node_name="adjudication_agent",
                ),
            )
            builder.add_node(
                "apply_user_adjudication_decision",
                rag_execution_middleware.wrap_node(
                    apply_user_adjudication_decision,
                    graph_name=self.id,
                    node_name="apply_user_adjudication_decision",
                ),
            )
        if query_term_expansion_enabled:
            if adjudication_enabled:
                builder.add_edge(START, "classify_query_correction_risk")
                builder.add_edge("classify_query_correction_risk", "expand_retrieval_terms")
            else:
                builder.add_edge(START, "expand_retrieval_terms")
            builder.add_edge("expand_retrieval_terms", "chunk_evidence")
        else:
            if adjudication_enabled:
                builder.add_edge(START, "classify_query_correction_risk")
                builder.add_edge("classify_query_correction_risk", "chunk_evidence")
            else:
                builder.add_edge(START, "chunk_evidence")
        builder.add_conditional_edges(
            "chunk_evidence",
            self._after_acquire,
            {"grade": "grade", "finalize": "finalize"},
        )
        if adjudication_enabled:
            builder.add_conditional_edges(
                "grade",
                after_grade,
                {"adjudicate": "adjudication_agent", "finalize": "finalize"},
            )
            builder.add_edge("adjudication_agent", "apply_user_adjudication_decision")
            builder.add_edge("apply_user_adjudication_decision", "finalize")
        else:
            builder.add_edge("grade", "finalize")
        builder.add_edge("finalize", END)
        self._graph: Any = builder.compile()

    async def invoke(self, state: RagGraphState) -> Mapping[str, object]:
        initial_tokens = state.get("token_usage", 0)
        result = cast(RagGraphState, await self._graph.ainvoke(state))
        return {
            "content_query": result["content_query"],
            "retrieval_expanded_query": result["retrieval_expanded_query"],
            "retrieval_lexical_queries": result["retrieval_lexical_queries"],
            "retrieval_protected_lexical_queries": result["retrieval_protected_lexical_queries"],
            "retrieval_attempt": result["retrieval_attempt"],
            "retrieval_candidates": result["retrieval_candidates"],
            "protected_chunk_ids": result["protected_chunk_ids"],
            "rerank_input_tokens": result["rerank_input_tokens"],
            "rerank_skipped_candidates": result["rerank_skipped_candidates"],
            "evidence": result["evidence"],
            "answer_evidence": result["answer_evidence"],
            "message": result["message"],
            "grade": result["grade"],
            "planning_required": result["planning_required"],
            "answer_plan": result["answer_plan"],
            "query_correction_risk": result["query_correction_risk"],
            "adjudication_agent_state": result["adjudication_agent_state"],
            "token_usage": max(0, result.get("token_usage", 0) - initial_tokens),
            "strategy_result": result["strategy_result"],
        }

    async def _acquire(self, state: RagGraphState) -> dict[str, object]:
        result = await self._chunk_evidence_pipeline.invoke(state)
        return {
            "retrieval_candidates": result["retrieval_candidates"],
            "rerank_input_tokens": result["rerank_input_tokens"],
            "rerank_skipped_candidates": result["rerank_skipped_candidates"],
            "evidence": result["evidence"],
            "answer_evidence": [],
            "message": result["message"],
        }

    def _after_acquire(self, state: RagGraphState) -> Literal["grade", "finalize"]:
        target: Literal["grade", "finalize"] = "finalize" if state["execution_mode"] == "retrieval" else "grade"
        source = ("rerank" if state["evidence"] and self._rerank_enabled else "expand_context") if state["retrieval_candidates"] else "retrieve"
        self._transition(
            state,
            source,
            "done" if target == "finalize" else "grade",
            "retrieval_terminal" if target == "finalize" else "evidence_ready",
        )
        return target

    async def _finalize(self, state: RagGraphState) -> dict[str, object]:
        answer_evidence = state["answer_evidence"] or state["evidence"]
        grade = state["grade"]
        ready = bool(answer_evidence) and (state["execution_mode"] == "retrieval" or (grade is not None and grade.verdict != "abstain"))
        answer_context = self._render_evidence(answer_evidence) if ready else ""
        corrected_answer_context: str | None = None
        sources = [item.source_payload() for item in answer_evidence]
        adjudication = state["adjudication_agent_state"]
        if ready and adjudication is not None and adjudication.overlays:
            overlays_by_index: dict[int, list[dict[str, object]]] = {}
            for overlay in adjudication.overlays:
                overlays_by_index.setdefault(overlay.evidence_index, []).append(overlay.model_dump(mode="json"))
            corrected_evidence = apply_evidence_overlays(answer_evidence, adjudication.overlays)
            corrected_answer_context = (
                f"{self._render_evidence(corrected_evidence)}\n\n{render_correction_notices(adjudication.overlays)}"
            )
            for source in sources:
                index = source.get("index")
                if isinstance(index, int) and index in overlays_by_index:
                    source["adjudication"] = overlays_by_index[index]
        strategy_result = StrategyResult(
            status="ready" if ready else "not_found",
            answer_context=answer_context,
            corrected_answer_context=corrected_answer_context,
            evidence=answer_evidence,
            sources=sources,
            message=state["message"] or (grade.reason if grade is not None and grade.verdict == "abstain" else None),
        )
        return {
            "planning_required": state["answer_plan"] is not None,
            "strategy_result": strategy_result,
        }
