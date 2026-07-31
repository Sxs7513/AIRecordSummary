from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph

from l2_core.rag.contracts import RagGraphState
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
        plan: Node,
        validate_plan: Node,
        select_planned_evidence: Node,
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
            "plan",
            rag_execution_middleware.wrap_node(plan, graph_name=self.id, node_name="plan"),
        )
        builder.add_node(
            "validate_plan",
            rag_execution_middleware.wrap_node(validate_plan, graph_name=self.id, node_name="validate_plan"),
        )
        builder.add_node(
            "select_planned_evidence",
            rag_execution_middleware.wrap_node(
                select_planned_evidence,
                graph_name=self.id,
                node_name="select_planned_evidence",
            ),
        )
        builder.add_node(
            "finalize",
            rag_execution_middleware.wrap_node(self._finalize, graph_name=self.id, node_name="finalize"),
        )
        if query_term_expansion_enabled:
            builder.add_edge(START, "expand_retrieval_terms")
            builder.add_edge("expand_retrieval_terms", "chunk_evidence")
        else:
            builder.add_edge(START, "chunk_evidence")
        builder.add_conditional_edges(
            "chunk_evidence",
            self._after_acquire,
            {"grade": "grade", "finalize": "finalize"},
        )
        builder.add_conditional_edges(
            "grade",
            after_grade,
            {"plan": "plan", "done": "finalize"},
        )
        builder.add_edge("plan", "validate_plan")
        builder.add_edge("validate_plan", "select_planned_evidence")
        builder.add_edge("select_planned_evidence", "finalize")
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
        answer_evidence = state["answer_evidence"] or (state["evidence"] if state["execution_mode"] == "retrieval" else [])
        grade = state["grade"]
        ready = bool(answer_evidence) and (state["execution_mode"] == "retrieval" or (grade is not None and grade.verdict != "abstain"))
        strategy_result = StrategyResult(
            status="ready" if ready else "not_found",
            answer_context=self._render_evidence(answer_evidence) if ready else "",
            evidence=answer_evidence,
            sources=[item.source_payload() for item in answer_evidence],
            message=state["message"] or (grade.reason if grade is not None and grade.verdict == "abstain" else None),
        )
        return {
            "planning_required": state["answer_plan"] is not None,
            "strategy_result": strategy_result,
        }
