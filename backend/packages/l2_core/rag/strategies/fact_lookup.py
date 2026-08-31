from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph

from l2_core.rag.contracts import Evidence, RagGraphState, RagStateUpdate
from l2_core.rag.evidence_overlays import apply_evidence_overlays, render_correction_notices
from l2_core.rag.execution_middleware import rag_execution_middleware
from l2_core.rag.strategies.base import StrategyResult
from l2_core.rag.workflows.chunk_evidence import ChunkEvidencePipeline

Node = Callable[[RagGraphState], Awaitable[RagStateUpdate]]


class FactLookupStrategy:
    id: Literal["fact_lookup"] = "fact_lookup"
    version = "1"

    def __init__(
        self,
        chunk_evidence_pipeline: ChunkEvidencePipeline,
        *,
        expand_retrieval_terms: Node,
        grade: Node,
        grade_variants: Node,
        classify_query_correction_risk: Node,
        adjudication_agent: Node,
        apply_user_adjudication_decision: Node,
        retry_without_history_scope: Node,
        adjudication_enabled: bool,
        after_user_adjudication: Callable[[RagGraphState], str],
        transition: Callable[[RagGraphState, str, str, str], None],
        rerank_enabled: bool,
        query_term_expansion_enabled: bool,
        render_evidence: Callable[[list[Evidence]], str],
        report_phase: Callable[[str, str], None],
    ) -> None:
        self._chunk_evidence_pipeline = chunk_evidence_pipeline
        self._render_evidence = render_evidence
        self._transition = transition
        self._rerank_enabled = rerank_enabled
        self._adjudication_enabled = adjudication_enabled
        self._retry_without_history_scope = retry_without_history_scope
        self._report_phase = report_phase
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
            "retry_without_history_scope",
            rag_execution_middleware.wrap_node(
                retry_without_history_scope,
                graph_name=self.id,
                node_name="retry_without_history_scope",
            ),
        )
        if adjudication_enabled:
            builder.add_node(
                "grade_variants",
                rag_execution_middleware.wrap_node(grade_variants, graph_name=self.id, node_name="grade_variants"),
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
        acquire_targets = {"grade": "grade", "finalize": "finalize"}
        if adjudication_enabled:
            acquire_targets["adjudicate"] = "adjudication_agent"
        builder.add_conditional_edges(
            "chunk_evidence",
            self._after_acquire,
            acquire_targets,
        )
        if adjudication_enabled:
            builder.add_conditional_edges(
                "grade",
                self._after_grade,
                {"retry": "retry_without_history_scope", "finalize": "finalize"},
            )
            builder.add_conditional_edges(
                "grade_variants",
                self._after_grade,
                {"retry": "retry_without_history_scope", "finalize": "finalize"},
            )
            builder.add_edge("adjudication_agent", "apply_user_adjudication_decision")
            builder.add_conditional_edges(
                "apply_user_adjudication_decision",
                after_user_adjudication,
                {"grade_variants": "grade_variants", "finalize": "finalize"},
            )
        else:
            builder.add_conditional_edges(
                "grade",
                self._after_grade,
                {"retry": "retry_without_history_scope", "finalize": "finalize"},
            )
        builder.add_edge("retry_without_history_scope", "chunk_evidence")
        builder.add_edge("finalize", END)
        self._graph: Any = builder.compile()

    async def invoke(self, state: RagGraphState) -> RagStateUpdate:
        initial_tokens = state.get("token_usage", 0)
        result = cast(RagGraphState, await self._graph.ainvoke(state))
        return {
            "content_query": result["content_query"],
            "retrieval_expanded_query": result["retrieval_expanded_query"],
            "retrieval_lexical_queries": result["retrieval_lexical_queries"],
            "retrieval_protected_lexical_queries": result["retrieval_protected_lexical_queries"],
            "retrieval_attempt": result["retrieval_attempt"],
            "filters": result["filters"],
            "history_scope_active": result["history_scope_active"],
            "retrieval_candidates": result["retrieval_candidates"],
            "protected_chunk_ids": result["protected_chunk_ids"],
            "rerank_input_tokens": result["rerank_input_tokens"],
            "rerank_skipped_candidates": result["rerank_skipped_candidates"],
            "evidence": result["evidence"],
            "answer_evidence": result["answer_evidence"],
            "message": result["message"],
            "grade": result["grade"],
            "original_grade": result["original_grade"],
            "corrected_grade": result["corrected_grade"],
            "planning_required": result["planning_required"],
            "answer_plan": result["answer_plan"],
            "query_correction_risk": result["query_correction_risk"],
            "adjudication_agent_state": result["adjudication_agent_state"],
            "token_usage": max(0, result.get("token_usage", 0) - initial_tokens),
            "strategy_result": result["strategy_result"],
        }

    async def _acquire(self, state: RagGraphState) -> RagStateUpdate:
        result = await self._chunk_evidence_pipeline.invoke(state)
        return {
            "retrieval_candidates": result["retrieval_candidates"],
            "rerank_input_tokens": result["rerank_input_tokens"],
            "rerank_skipped_candidates": result["rerank_skipped_candidates"],
            "evidence": result["evidence"],
            "answer_evidence": [],
            "message": result["message"],
        }

    def _after_acquire(self, state: RagGraphState) -> Literal["grade", "adjudicate", "finalize"]:
        if state["execution_mode"] == "retrieval":
            target: Literal["grade", "adjudicate", "finalize"] = "finalize"
        elif self._adjudication_enabled and state["query_correction_risk"] and state["evidence"]:
            target = "adjudicate"
        else:
            target = "grade"
        source = ("rerank" if state["evidence"] and self._rerank_enabled else "expand_context") if state["retrieval_candidates"] else "retrieve"
        self._transition(
            state,
            source,
            "done" if target == "finalize" else "adjudication_agent" if target == "adjudicate" else "grade",
            "retrieval_terminal" if target == "finalize" else "query_correction_risk" if target == "adjudicate" else "evidence_ready",
        )
        if target == "adjudicate":
            self._report_phase("correcting_asr", "正在尝试纠正 ASR 错误")
        return target

    def _after_grade(self, state: RagGraphState) -> Literal["retry", "finalize"]:
        grade = state.get("grade")
        should_retry = (
            state["history_scope_active"]
            and state["retrieval_attempt"] == 0
            and grade is not None
            and grade.verdict == "abstain"
        )
        if not should_retry:
            return "finalize"
        self._transition(state, "grade", "retry_without_history_scope", "history_scope_insufficient")
        return "retry"

    async def _finalize(self, state: RagGraphState) -> RagStateUpdate:
        answer_evidence = state["answer_evidence"] or state["evidence"]
        grade = state["grade"]
        original_grade = state.get("original_grade") or grade
        corrected_grade = state.get("corrected_grade") or grade
        adjudication = state["adjudication_agent_state"]
        pending_confirmation = adjudication is not None and adjudication.pending_confirmation is not None
        has_overlays = adjudication is not None and bool(adjudication.overlays)
        original_ready = bool(answer_evidence) and (
            state["execution_mode"] == "retrieval"
            or pending_confirmation
            or (original_grade is not None and original_grade.verdict != "abstain")
        )
        corrected_ready = bool(answer_evidence) and (
            state["execution_mode"] == "retrieval"
            or pending_confirmation
            or (corrected_grade is not None and corrected_grade.verdict != "abstain")
        )
        if not has_overlays:
            corrected_ready = False
        ready = original_ready or corrected_ready
        answer_context = self._render_evidence(answer_evidence) if original_ready else ""
        corrected_answer_context: str | None = None
        original_sources = [item.source_payload() for item in answer_evidence]
        corrected_sources = [source.copy() for source in original_sources]
        if corrected_ready and adjudication is not None and adjudication.overlays:
            overlays_by_index: dict[int, list[dict[str, object]]] = {}
            for overlay in adjudication.overlays:
                overlays_by_index.setdefault(overlay.evidence_index, []).append(overlay.model_dump(mode="json"))
            corrected_evidence = apply_evidence_overlays(answer_evidence, adjudication.overlays)
            corrected_answer_context = (
                f"{self._render_evidence(corrected_evidence)}\n\n{render_correction_notices(adjudication.overlays)}"
            )
            for source in corrected_sources:
                index = source["index"]
                if index in overlays_by_index:
                    source["adjudication"] = overlays_by_index[index]
        sources = corrected_sources if corrected_ready else original_sources
        strategy_result = StrategyResult(
            status="ready" if ready else "not_found",
            answer_context=answer_context,
            corrected_answer_context=corrected_answer_context,
            evidence=answer_evidence,
            sources=sources,
            original_sources=original_sources,
            corrected_sources=corrected_sources if corrected_ready else [],
            message=state["message"] or (
                corrected_grade.reason
                if corrected_grade is not None and corrected_grade.verdict == "abstain" and not original_ready
                else original_grade.reason if original_grade is not None and original_grade.verdict == "abstain" and not corrected_ready else None
            ),
        )
        return {
            "planning_required": state["answer_plan"] is not None,
            "strategy_result": strategy_result,
        }
