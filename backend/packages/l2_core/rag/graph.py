from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol, cast
from uuid import UUID

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph

from l1_foundation.llm import (
    ChatMessage,
    ChatRole,
    CompletionOptions,
    LlmGenerateResult,
    LlmProvider,
    ResponseFormat,
    ResponseFormatType,
    build_llm_generate_command,
)
from l1_foundation.observability import (
    InstrumentedModelClient,
    finish_span,
    start_span,
)
from l1_foundation.observability.context import current_span
from l2_core.rag.adjudication.agent import EvidenceAdjudicationAgent
from l2_core.rag.adjudication.contracts import (
    AdjudicationConfirmationBlock,
    ClaimConfirmationDecision,
    CorrectionRiskAssessment,
    EvidenceOverlay,
)
from l2_core.rag.adjudication.prompts import AuditPromptVariant, correction_risk_prompt
from l2_core.rag.adjudication.web_research import GroundedSearchClient
from l2_core.rag.citations import normalize_answer_citations
from l2_core.rag.contracts import (
    AnswerPlan,
    AnswerPlanItem,
    Evidence,
    EvidenceGrade,
    EvidenceSource,
    RagGraphState,
    RagHistoryMessage,
    RagRoute,
    RagStateUpdate,
    ResolvedFilters,
    RetrievalCandidateRow,
    RetrievalTerms,
    StrategyId,
)
from l2_core.rag.evidence_overlays import apply_evidence_overlays
from l2_core.rag.execution_middleware import RagExecutionCancelled, rag_execution_middleware
from l2_core.rag.hooks import (
    RagExecutionHook,
    RagNodeCompleted,
    RagOperationCompleted,
    current_rag_execution_hook,
    rag_execution_hook_scope,
)
from l2_core.rag.observability import elapsed_ms, log_event, started_at
from l2_core.rag.prompts import answer_plan_prompt, answer_prompt, grade_prompt, retrieval_terms_prompt, route_prompt
from l2_core.rag.retrieval import RagRetriever
from l2_core.rag.routing import AMBIGUOUS_RECORDING_SCOPE_MESSAGE, ROUTE_UNRESOLVED_MESSAGE, parse_route_response
from l2_core.rag.scope import make_filters, resolve_time_range
from l2_core.rag.strategies.fact_lookup import FactLookupStrategy
from l2_core.rag.strategies.metadata_lookup import MetadataLookupStrategy
from l2_core.rag.strategies.registry import StrategyRegistry
from l2_core.rag.strategies.scope_summary import ScopeSummaryStrategy
from l2_core.rag.streaming import ThinkTagFilter
from l2_core.rag.token_budget import RagTokenBudgetMiddleware
from l2_core.rag.workflows.chunk_evidence import ChunkEvidencePipeline

AnswerVariant = Literal["original", "corrected"]


class AggregateAnswerStream(Protocol):
    def start_aggregate_message(self) -> None: ...

    def aggregate_text(self, variant: str, value: str) -> None: ...

    def complete_aggregate_variant(
        self,
        variant: str,
        text: str,
        sources: list[dict[str, object]],
    ) -> None: ...

    def fail_aggregate_variant(self, variant: str, error: str) -> None: ...

logger = logging.getLogger("rag")

MAX_ADJUDICATION_CASES = 2
MAX_ADJUDICATION_ITERATIONS = 4
MAX_ADJUDICATION_SEARCHES = 3
INSUFFICIENT_EVIDENCE_ANSWER = "没有在录音中找到足够依据。"


class RagGraph:
    """Typed LangGraph implementation of route, retrieval, evidence checks and answer generation."""

    def __init__(
        self,
        retriever: RagRetriever,
        model_client: InstrumentedModelClient,
        online_provider: LlmProvider,
        context_size: int,
        *,
        plan_local_input_tokens: int = 4_000,
        max_total_tokens: int = 50_000,
        route_model_profile: Literal["default", "rag"] = "default",
        node_model_profile: Literal["default", "rag"] = "default",
        query_term_expansion_enabled: bool = False,
        asr_adjudication_enabled: bool = False,
        asr_adjudication_web_search_enabled: bool = False,
        asr_adjudication_auto_resolve_confidence: float = 0.95,
        asr_adjudication_audit_prompt_variant: AuditPromptVariant = "relation_rules",
        asr_adjudication_audit_model: str | None = None,
        asr_adjudication_audit_min_request_interval_seconds: float | None = None,
        grounded_search_client: GroundedSearchClient | None = None,
    ) -> None:
        self._retriever = retriever
        self._model_client = model_client
        self._online_provider = online_provider
        self._context_size = context_size
        self._plan_local_input_tokens = plan_local_input_tokens
        self._route_model_profile: Literal["default", "rag"] = route_model_profile
        self._node_model_profile: Literal["default", "rag"] = node_model_profile
        self._query_term_expansion_enabled = query_term_expansion_enabled
        self._asr_adjudication_enabled = asr_adjudication_enabled
        self._asr_adjudication_web_search_enabled = asr_adjudication_web_search_enabled and grounded_search_client is not None
        self._token_budget = RagTokenBudgetMiddleware(max_total_tokens)
        self._adjudication_agent = EvidenceAdjudicationAgent(
            model_client=model_client,
            online_provider=online_provider,
            context_size=context_size,
            token_budget=self._token_budget,
            grounded_search_client=grounded_search_client,
            web_search_enabled=self._asr_adjudication_web_search_enabled,
            auto_resolve_confidence=asr_adjudication_auto_resolve_confidence,
            max_cases=MAX_ADJUDICATION_CASES,
            max_iterations=MAX_ADJUDICATION_ITERATIONS,
            max_searches=MAX_ADJUDICATION_SEARCHES,
            audit_prompt_variant=asr_adjudication_audit_prompt_variant,
            audit_model=asr_adjudication_audit_model,
            audit_min_request_interval_seconds=asr_adjudication_audit_min_request_interval_seconds,
            node_started=self._node_started,
            node_completed=self._node_completed,
            event_logger=log_event,
        )
        self._chunk_evidence = ChunkEvidencePipeline(
            retriever,
            node_started=self._node_started,
            node_completed=self._node_completed,
            operation_completed=self._operation_completed,
            transition=self._transition,
        )
        self._strategies = StrategyRegistry(
            (
                FactLookupStrategy(
                    chunk_evidence_pipeline=self._chunk_evidence,
                    expand_retrieval_terms=self._expand_retrieval_terms,
                    grade=self._grade,
                    grade_variants=self._grade_variants,
                    classify_query_correction_risk=self._classify_query_correction_risk,
                    adjudication_agent=self._adjudication_agent.start,
                    apply_user_adjudication_decision=self._apply_user_adjudication_decision,
                    retry_without_history_scope=self._retry_without_history_scope,
                    adjudication_enabled=asr_adjudication_enabled,
                    after_user_adjudication=self._after_user_adjudication,
                    transition=self._transition,
                    rerank_enabled=bool(getattr(retriever, "rerank_enabled", False)),
                    query_term_expansion_enabled=query_term_expansion_enabled,
                    render_evidence=self._evidence_text,
                ),
                MetadataLookupStrategy(
                    retriever,
                    node_started=self._node_started,
                    node_completed=self._node_completed,
                    operation_completed=self._operation_completed,
                ),
                ScopeSummaryStrategy(
                    retriever,
                    node_started=self._node_started,
                    node_completed=self._node_completed,
                    operation_completed=self._operation_completed,
                    render_evidence=self._evidence_text,
                ),
            )
        )
        self._strategies.validate_complete({"fact_lookup", "metadata_lookup", "scope_summary"})
        builder = cast(Any, StateGraph(RagGraphState))
        builder.add_node(
            "route",
            rag_execution_middleware.wrap_node(self._route, graph_name="root", node_name="route"),
        )
        for strategy in self._strategies.all():
            node_name = f"strategy_{strategy.id}"
            builder.add_node(
                node_name,
                rag_execution_middleware.wrap_node(
                    self._strategy_node(strategy.id),
                    graph_name="root",
                    node_name=node_name,
                ),
            )
        builder.add_edge(START, "route")
        builder.add_conditional_edges(
            "route",
            self._after_route_strategy,
            {
                "fact_lookup": "strategy_fact_lookup",
                "metadata_lookup": "strategy_metadata_lookup",
                "scope_summary": "strategy_scope_summary",
                "unresolved": END,
            },
        )
        for strategy in self._strategies.all():
            builder.add_edge(f"strategy_{strategy.id}", END)
        self._graph: Any = builder.compile()

    async def run(
        self,
        query: str,
        limit: int,
        scope_recording_ids: list[UUID],
        on_phase: Callable[[str, str, int | None], None],
        on_delta: Callable[[str], None],
        history: list[RagHistoryMessage] | None = None,
        run_id: UUID | None = None,
        hook: RagExecutionHook | None = None,
        existing_answer: str | None = None,
        existing_original_answer: str | None = None,
        aggregate_stream: AggregateAnswerStream | None = None,
        restored_state: RagGraphState | None = None,
        adjudication_user_decision: ClaimConfirmationDecision | None = None,
    ) -> tuple[str, list[EvidenceSource], bool, str | None, AdjudicationConfirmationBlock | None]:
        with rag_execution_hook_scope(hook):
            return await self._run(
                query,
                limit,
                scope_recording_ids,
                on_phase,
                on_delta,
                history,
                run_id,
                existing_answer,
                existing_original_answer,
                aggregate_stream,
                restored_state,
                adjudication_user_decision,
            )

    async def _run(
        self,
        query: str,
        limit: int,
        scope_recording_ids: list[UUID],
        on_phase: Callable[[str, str, int | None], None],
        on_delta: Callable[[str], None],
        history: list[RagHistoryMessage] | None = None,
        run_id: UUID | None = None,
        existing_answer: str | None = None,
        existing_original_answer: str | None = None,
        aggregate_stream: AggregateAnswerStream | None = None,
        restored_state: RagGraphState | None = None,
        adjudication_user_decision: ClaimConfirmationDecision | None = None,
    ) -> tuple[str, list[EvidenceSource], bool, str | None, AdjudicationConfirmationBlock | None]:
        trace_id = str(run_id) if run_id is not None else "standalone"
        graph_started = started_at()
        log_event(
            "graph_started",
            trace_id,
            query_chars=len(query),
            limit=limit,
            scope_recording_count=len(scope_recording_ids),
            history_messages=len(history or []),
        )
        on_phase("routing", "正在理解问题", 10)
        try:
            initial_state = self._initial_state(trace_id, "answer", query, limit, scope_recording_ids, history)
            if restored_state is not None:
                initial_state.update(restored_state)
                initial_state.update(
                    run_id=trace_id,
                    execution_mode="answer",
                    query=query,
                    history=history or [],
                    limit=limit,
                    scope_recording_ids=[str(item) for item in scope_recording_ids],
                    adjudication_user_decision=adjudication_user_decision,
                )
            elif adjudication_user_decision is not None:
                initial_state["adjudication_user_decision"] = adjudication_user_decision
            result = await self._graph.ainvoke(initial_state)
        except asyncio.CancelledError:
            finish_span(current_span(), "cancelled", error_type="CancelledError")
            log_event(
                "graph_completed",
                trace_id,
                status="cancelled",
                stage="graph_nodes",
                elapsed_ms=elapsed_ms(graph_started),
            )
            raise
        except RagExecutionCancelled:
            finish_span(current_span(), "cancelled", error_type="RagExecutionCancelled")
            log_event(
                "graph_completed",
                trace_id,
                status="cancelled",
                stage="graph_nodes",
                elapsed_ms=elapsed_ms(graph_started),
            )
            raise
        except Exception as error:
            finish_span(current_span(), "failed", error_type=type(error).__name__)
            log_event(
                "graph_completed",
                trace_id,
                level=logging.ERROR,
                status="failed",
                stage="graph_nodes",
                error_type=type(error).__name__,
                elapsed_ms=elapsed_ms(graph_started),
            )
            raise
        state = cast(RagGraphState, result)
        if state["route_error"] is not None:
            message = AMBIGUOUS_RECORDING_SCOPE_MESSAGE if state["route_error"] == "ambiguous_recording_scope" else ROUTE_UNRESOLVED_MESSAGE
            log_event(
                "graph_completed",
                trace_id,
                status="route_error",
                reason=state["route_error"],
                elapsed_ms=elapsed_ms(graph_started),
            )
            return message, [], True, state["route_error"], None
        strategy_result = state["strategy_result"]
        if strategy_result is None:
            raise RuntimeError("RAG graph completed without a strategy result")
        evidence = strategy_result.evidence
        plan = state["answer_plan"]
        if strategy_result.status != "ready":
            log_event(
                "graph_completed",
                trace_id,
                status="insufficient_evidence",
                reason=strategy_result.message or "strategy_not_ready",
                evidence_count=len(evidence),
                elapsed_ms=elapsed_ms(graph_started),
            )
            return INSUFFICIENT_EVIDENCE_ANSWER, [], True, strategy_result.message, None
        adjudication = state.get("adjudication_agent_state")
        if adjudication is not None and adjudication.pending_confirmation is not None:
            return "", strategy_result.sources, False, None, adjudication.pending_confirmation
        on_phase("generating", "正在生成回答", 75)
        answer_started = self._node_started(state, "answer")
        original_answerable = bool(strategy_result.answer_context)
        corrected_available = strategy_result.corrected_answer_context is not None
        primary_is_corrected = corrected_available
        primary_answer_context = strategy_result.corrected_answer_context if primary_is_corrected else strategy_result.answer_context
        primary_grade = state.get("corrected_grade") if primary_is_corrected else state.get("original_grade")
        primary_sources = (
            strategy_result.corrected_sources if primary_is_corrected else strategy_result.original_sources
        ) or strategy_result.sources
        if not primary_answer_context or not primary_sources:
            raise RuntimeError("RAG graph marked a response ready without an answerable evidence variant")
        corrected_direct_answer = primary_is_corrected
        log_event(
            "answer_mode_selected",
            trace_id,
            mode="direct" if corrected_direct_answer or plan is None else "planned",
            strategy_id=state["route"].strategy_id if state["route"] is not None else None,
            retrieved_evidence_count=len(state["evidence"]),
            answer_evidence_count=len(strategy_result.sources),
            answer_evidence_text=self._evidence_log_entries(strategy_result.evidence),
        )
        comparison_answer = corrected_available and aggregate_stream is not None
        generate_original_answer = comparison_answer and original_answerable
        active_aggregate_stream = aggregate_stream if comparison_answer else None
        answer_query = state["query"]
        # if corrected_direct_answer:
        #     logger.info(
        #         "TEMP corrected answer evidence run_id=%s text=%s",
        #         trace_id,
        #         primary_answer_context,
        #     )
        prompt, values = answer_prompt(
            answer_query,
            None if corrected_direct_answer else (plan.model_dump_json() if plan is not None else None),
            primary_answer_context,
            primary_grade.verdict if primary_grade is not None else None,
            existing_answer,
        )
        messages = prompt.invoke(values).to_messages()
        original_messages: object | None = None
        if generate_original_answer:
            assert active_aggregate_stream is not None
            original_grade = state.get("original_grade")
            original_prompt, original_values = answer_prompt(
                answer_query,
                plan.model_dump_json() if plan is not None else None,
                strategy_result.answer_context,
                original_grade.verdict if original_grade is not None else None,
                existing_original_answer,
            )
            original_messages = original_prompt.invoke(original_values).to_messages()
        if comparison_answer:
            assert active_aggregate_stream is not None
            active_aggregate_stream.start_aggregate_message()
            if not original_answerable:
                active_aggregate_stream.complete_aggregate_variant(
                    "original",
                    INSUFFICIENT_EVIDENCE_ANSWER,
                    [],
                )
        first_token_logged: set[AnswerVariant] = set()

        def trace_delta(variant: AnswerVariant, delta: str) -> None:
            if variant not in first_token_logged:
                first_token_logged.add(variant)
                log_event(
                    "answer_first_token",
                    trace_id,
                    variant=variant,
                    elapsed_ms=elapsed_ms(answer_started),
                )
            if active_aggregate_stream is not None:
                active_aggregate_stream.aggregate_text(variant, delta)
            else:
                on_delta(delta)

        primary_variant: AnswerVariant = "corrected" if primary_is_corrected else "original"
        primary_delta = rag_execution_middleware.wrap_delta(lambda delta: trace_delta(primary_variant, delta))

        answer_provider = LlmProvider.LOCAL if state["route"] is not None and state["route"].strategy_id == "metadata_lookup" else self._online_provider
        original_answer: str | None = None
        original_result: LlmGenerateResult | None = None
        try:
            if generate_original_answer:
                assert active_aggregate_stream is not None
                if original_messages is None:
                    raise RuntimeError("Dual answer generation requires original messages")
                original_delta = rag_execution_middleware.wrap_delta(lambda delta: trace_delta("original", delta))
                generated = await asyncio.gather(
                    self._generate_streaming_answer(
                        state,
                        original_messages,
                        2048,
                        original_delta,
                        provider=answer_provider,
                    ),
                    self._generate_streaming_answer(
                        state,
                        messages,
                        2048,
                        primary_delta,
                        provider=answer_provider,
                    ),
                    return_exceptions=True,
                )
                original_generated, corrected_generated = generated
                if isinstance(original_generated, BaseException):
                    active_aggregate_stream.fail_aggregate_variant(
                        "original",
                        str(original_generated) or type(original_generated).__name__,
                    )
                    log_event(
                        "answer_variant_failed",
                        trace_id,
                        level=logging.WARNING,
                        variant="original",
                        error_type=type(original_generated).__name__,
                    )
                else:
                    original_answer, original_result = original_generated
                if isinstance(corrected_generated, BaseException):
                    active_aggregate_stream.fail_aggregate_variant(
                        "corrected",
                        str(corrected_generated) or type(corrected_generated).__name__,
                    )
                    raise corrected_generated
                answer, answer_result = corrected_generated
            else:
                answer, answer_result = await self._generate_streaming_answer(
                    state,
                    messages,
                    2048,
                    primary_delta,
                    provider=answer_provider,
                )
        except asyncio.CancelledError:
            finish_span(current_span(), "cancelled", error_type="CancelledError")
            log_event(
                "graph_completed",
                trace_id,
                status="cancelled",
                stage="answer",
                elapsed_ms=elapsed_ms(graph_started),
            )
            raise
        except RagExecutionCancelled:
            finish_span(current_span(), "cancelled", error_type="RagExecutionCancelled")
            log_event(
                "graph_completed",
                trace_id,
                status="cancelled",
                stage="answer",
                elapsed_ms=elapsed_ms(graph_started),
            )
            raise
        except Exception as error:
            model_execution = "local" if answer_provider == LlmProvider.LOCAL else "online"
            finish_span(
                current_span(),
                "failed",
                error_type=type(error).__name__,
                metadata={"model_execution": model_execution, "provider": answer_provider.value},
            )
            log_event(
                "node_failed",
                trace_id,
                level=logging.ERROR,
                node="answer",
                error_type=type(error).__name__,
                model_execution=model_execution,
                provider=answer_provider.value,
                elapsed_ms=elapsed_ms(answer_started),
            )
            log_event(
                "graph_completed",
                trace_id,
                level=logging.ERROR,
                status="failed",
                stage="answer",
                error_type=type(error).__name__,
                elapsed_ms=elapsed_ms(graph_started),
            )
            raise
        if not answer:
            raise RuntimeError("RAG answer model returned an empty streamed answer")
        complete_answer = f"{existing_answer or ''}{answer}"
        normalized_citations = normalize_answer_citations(complete_answer, primary_sources)
        answer = normalized_citations.text
        sources = normalized_citations.sources
        if normalized_citations.invalid_indexes:
            log_event(
                "answer_invalid_citations_removed",
                trace_id,
                level=logging.WARNING,
                invalid_indexes=list(normalized_citations.invalid_indexes),
            )
        state["token_usage"] += self._token_budget.actual_usage(answer_result)
        if comparison_answer:
            assert active_aggregate_stream is not None
            active_aggregate_stream.complete_aggregate_variant("corrected", answer, [dict(source) for source in sources])
            if original_answer is not None and original_result is not None:
                complete_original_answer = f"{existing_original_answer or ''}{original_answer}"
                normalized_original = normalize_answer_citations(
                    complete_original_answer,
                    strategy_result.original_sources,
                )
                original_answer = normalized_original.text
                active_aggregate_stream.complete_aggregate_variant(
                    "original",
                    original_answer,
                    [dict(source) for source in normalized_original.sources],
                )
                state["token_usage"] += self._token_budget.actual_usage(original_result)
                if normalized_original.invalid_indexes:
                    log_event(
                        "answer_invalid_citations_removed",
                        trace_id,
                        level=logging.WARNING,
                        variant="original",
                        invalid_indexes=list(normalized_original.invalid_indexes),
                    )
        self._node_completed(
            state,
            "answer",
            answer_started,
            answer_chars=len(answer),
            evidence_count=len(sources),
            mode="direct" if corrected_direct_answer or plan is None else "planned",
            total_tokens=state["token_usage"],
            model_execution="local" if answer_provider == LlmProvider.LOCAL else "online",
            provider=answer_provider.value,
            dual_answer=comparison_answer,
        )
        log_event(
            "graph_completed",
            trace_id,
            status="succeeded",
            evidence_count=len(sources),
            retrieved_evidence_count=len(state["evidence"]),
            planning_required=state["planning_required"],
            dual_answer=comparison_answer,
            elapsed_ms=elapsed_ms(graph_started),
        )
        return answer, sources, False, strategy_result.message, None

    async def run_retrieval(
        self,
        query: str,
        limit: int,
        scope_recording_ids: list[UUID],
        *,
        history: list[RagHistoryMessage] | None = None,
        run_id: UUID | None = None,
        hook: RagExecutionHook | None = None,
    ) -> RagGraphState:
        """Execute the production routing and retrieval nodes without grade or answer generation."""
        trace_id = str(run_id) if run_id is not None else "standalone"
        with rag_execution_hook_scope(hook):
            result = await self._graph.ainvoke(self._initial_state(trace_id, "retrieval", query, limit, scope_recording_ids, history))
        return cast(RagGraphState, result)

    async def grade_retrieval(
        self,
        state: RagGraphState,
        *,
        hook: RagExecutionHook | None = None,
    ) -> RagGraphState:
        """Run the production evidence gate against an already retrieved evaluation state."""

        with rag_execution_hook_scope(hook):
            update = await self._grade(state)
        result = cast(RagGraphState, {**state, **update})
        result["token_usage"] = state.get("token_usage", 0) + update.get("token_usage", 0)
        return result

    @staticmethod
    def _initial_state(
        trace_id: str,
        execution_mode: Literal["answer", "retrieval"],
        query: str,
        limit: int,
        scope_recording_ids: list[UUID],
        history: list[RagHistoryMessage] | None,
    ) -> RagGraphState:
        return RagGraphState(
            run_id=trace_id,
            execution_mode=execution_mode,
            query=query,
            history=history or [],
            limit=limit,
            scope_recording_ids=[str(item) for item in scope_recording_ids],
            route=None,
            route_error=None,
            filters=None,
            history_scope_active=False,
            content_query=query,
            retrieval_expanded_query=None,
            retrieval_lexical_queries=[],
            retrieval_protected_lexical_queries=[],
            retrieval_attempt=0,
            retrieval_candidates=[],
            protected_chunk_ids=[],
            rerank_input_tokens=0,
            rerank_skipped_candidates=0,
            evidence=[],
            answer_evidence=[],
            message=None,
            grade=None,
            original_grade=None,
            corrected_grade=None,
            planning_required=False,
            answer_plan=None,
            query_correction_risk=False,
            adjudication_agent_state=None,
            adjudication_user_decision=None,
            token_usage=0,
            strategy_result=None,
        )

    @staticmethod
    def _node_started(state: RagGraphState, node: str) -> float:
        start = started_at()
        start_span(node, attempt=state.get("retrieval_attempt", 0))
        log_event(
            "node_started",
            state.get("run_id", "standalone"),
            node=node,
            attempt=state.get("retrieval_attempt", 0),
        )
        return start

    @staticmethod
    def _node_completed(state: RagGraphState, node: str, start: float, **fields: object) -> None:
        fields.setdefault("model_execution", RagGraph._node_model_execution(node))
        duration = elapsed_ms(start)
        safe_metadata = {
            key: value
            for key, value in fields.items()
            if not any(sensitive in key.lower() for sensitive in ("raw", "prompt", "query", "text", "message", "transcript"))
        }
        finish_span(current_span(), "succeeded", metadata=safe_metadata)
        log_event(
            "node_completed",
            state.get("run_id", "standalone"),
            node=node,
            attempt=state.get("retrieval_attempt", 0),
            elapsed_ms=duration,
            data=fields,
        )
        current_rag_execution_hook().on_node_completed(
            RagNodeCompleted(
                node=node,
                attempt=state.get("retrieval_attempt", 0),
                elapsed_ms=duration,
                metadata=fields,
            )
        )

    @staticmethod
    def _node_model_execution(node: str) -> Literal["local", "online", "none"]:
        """Execution location for nodes with a fixed model dependency."""

        if node in {
            "route",
            "classify_query_correction_risk",
        }:
            return "online"
        if node in {"retrieve", "rerank"}:
            return "local"
        return "none"

    @staticmethod
    def _operation_completed(
        node: str,
        operation: str,
        output: object,
        start: float,
        *,
        status: str = "succeeded",
        details: Mapping[str, object] | None = None,
    ) -> None:
        current_rag_execution_hook().on_operation_completed(
            RagOperationCompleted(
                node=node,
                operation=operation,
                output=output,
                elapsed_ms=elapsed_ms(start),
                status=status,
                details=details or {},
            )
        )

    @staticmethod
    def _transition(state: RagGraphState, source: str, target: str, reason: str) -> None:
        log_event(
            "graph_transition",
            state.get("run_id", "standalone"),
            source=source,
            target=target,
            reason=reason,
            attempt=state.get("retrieval_attempt", 0),
        )

    async def _route(self, state: RagGraphState) -> RagStateUpdate:
        node_started = self._node_started(state, "route")
        prompt, values, _ = route_prompt(state["query"], self._route_history_context(state["history"]))
        result = await self._complete(
            state,
            "route",
            prompt.invoke(values).to_messages(),
            max_tokens=700,
            json_schema=RagRoute.model_json_schema(),
            provider=self._online_provider,
        )
        raw = result.text
        token_usage = self._token_budget.actual_usage(result)
        logger.debug("rag route raw output: %s", raw)
        route = parse_route_response(raw)
        if route is None:
            self._node_completed(
                state,
                "route",
                node_started,
                outcome="route_unresolved",
                query=state["query"],
                query_chars=len(state["query"]),
                response_chars=len(raw),
                raw=raw,
            )
            return {"route_error": "route_unresolved", "token_usage": token_usage}
        if route.status != "resolved":
            route_error = route.error_code or "route_unresolved"
            self._node_completed(
                state,
                "route",
                node_started,
                outcome=route.status,
                query=state["query"],
                reason=route_error,
                response_chars=len(raw),
                raw=raw,
            )
            return {"route_error": route_error, "token_usage": token_usage}
        if route.time_range is not None and resolve_time_range(route) == (None, None):
            self._node_completed(
                state,
                "route",
                node_started,
                outcome="unresolved",
                query=state["query"],
                reason="unsupported_time_expression",
                response_chars=len(raw),
                raw=raw,
            )
            return {"route_error": "unsupported_time_expression", "token_usage": token_usage}
        route_error = self._validate_selected_recording_ids(route, [UUID(value) for value in state["scope_recording_ids"]])
        if route_error is not None:
            self._node_completed(
                state,
                "route",
                node_started,
                outcome="rejected",
                query=state["query"],
                reason=route_error,
                response_chars=len(raw),
                raw=raw,
            )
            return {"route_error": route_error, "token_usage": token_usage}
        scope_ids = [UUID(value) for value in state["scope_recording_ids"]]
        history_scope_active = bool(route.history_recording_ids)
        filters = self._resolve_route_filters(route, scope_ids, include_history_scope=history_scope_active)
        self._node_completed(
            state,
            "route",
            node_started,
            outcome="resolved",
            query=state["query"],
            strategy_id=route.strategy_id,
            recording_limit=route.recording_limit,
            recording_rank=route.recording_rank,
            filtered_recording_count=len(filters.recording_ids),
            match_none=filters.match_none,
            response_chars=len(raw),
            raw=raw,
            resolved_filters=filters.model_dump(mode="json"),
        )
        return {
            "route": route,
            "route_error": None,
            "filters": filters,
            "history_scope_active": history_scope_active,
            "content_query": route.content_query.strip() if route.content_query and route.content_query.strip() else state["query"],
            "token_usage": token_usage,
        }

    async def _expand_retrieval_terms(self, state: RagGraphState) -> RagStateUpdate:
        """Prepare a scope-free content query and faithful lexical anchors."""

        node_started = self._node_started(state, "expand_retrieval_terms")
        if not self._query_term_expansion_enabled:
            self._node_completed(state, "expand_retrieval_terms", node_started, enabled=False, term_count=0, phrase_count=0, evidence_query_count=0)
            return {
                "content_query": state["content_query"],
                "retrieval_expanded_query": None,
                "retrieval_lexical_queries": [],
                "retrieval_protected_lexical_queries": [],
            }
        prompt, values, parser = retrieval_terms_prompt(state["content_query"])
        try:
            result = await self._complete(
                state,
                "expand_retrieval_terms",
                prompt.invoke(values).to_messages(),
                max_tokens=240,
                json_schema=RetrievalTerms.model_json_schema(),
                provider=self._provider_for_input(prompt.invoke(values).to_messages(), 1_000),
            )
            terms = parser.parse(result.text)
        except Exception as error:
            log_event(
                "node_warning",
                state.get("run_id", "standalone"),
                level=logging.WARNING,
                node="expand_retrieval_terms",
                reason="term_extraction_fallback",
                error_type=type(error).__name__,
            )
            self._node_completed(
                state,
                "expand_retrieval_terms",
                node_started,
                enabled=True,
                fallback=True,
                term_count=0,
                phrase_count=0,
                evidence_query_count=0,
            )
            return {
                "content_query": state["content_query"],
                "retrieval_expanded_query": None,
                "retrieval_lexical_queries": [],
                "retrieval_protected_lexical_queries": [],
            }
        content_query = terms.content_query.strip() or state["content_query"]
        anchors = list(dict.fromkeys([*terms.phrases, *terms.terms]))
        expanded_query = " ".join(anchors).strip() or None
        evidence_queries = list(dict.fromkeys(terms.evidence_queries))
        lexical_queries = list(dict.fromkeys([*(item for item in anchors if item != content_query), *evidence_queries]))
        protected_lexical_queries = [item for item in lexical_queries if self._is_protected_lexical_query(item)]
        self._node_completed(
            state,
            "expand_retrieval_terms",
            node_started,
            enabled=True,
            fallback=False,
            term_count=len(terms.terms),
            phrase_count=len(terms.phrases),
            evidence_query_count=len(evidence_queries),
            expanded_query_present=expanded_query is not None,
            extracted_terms=terms.terms,
            extracted_phrases=terms.phrases,
            extracted_evidence_queries=evidence_queries,
            lexical_queries=lexical_queries,
            protected_lexical_queries=protected_lexical_queries,
            content_query=content_query,
        )
        return {
            "content_query": content_query,
            "retrieval_expanded_query": expanded_query if expanded_query != content_query else None,
            "retrieval_lexical_queries": lexical_queries,
            "retrieval_protected_lexical_queries": protected_lexical_queries,
            "token_usage": self._token_budget.actual_usage(result),
        }

    async def _classify_query_correction_risk(self, state: RagGraphState) -> RagStateUpdate:
        node_started = self._node_started(state, "classify_query_correction_risk")
        if not self._asr_adjudication_enabled or state["execution_mode"] != "answer":
            self._node_completed(
                state,
                "classify_query_correction_risk",
                node_started,
                enabled=False,
                has_risk=False,
            )
            return {"query_correction_risk": False, "adjudication_agent_state": None}
        prompt, values = correction_risk_prompt(state["query"])
        messages = prompt.invoke(values).to_messages()
        try:
            result = await self._complete(
                state,
                "classify_query_correction_risk",
                messages,
                max_tokens=80,
                json_schema=CorrectionRiskAssessment.model_json_schema(),
                provider=self._online_provider,
            )
            assessment = CorrectionRiskAssessment.model_validate_json(result.text)
        except Exception as error:
            log_event(
                "node_warning",
                state.get("run_id", "standalone"),
                level=logging.WARNING,
                node="classify_query_correction_risk",
                reason="classification_fallback",
                error_type=type(error).__name__,
            )
            self._node_completed(
                state,
                "classify_query_correction_risk",
                node_started,
                enabled=True,
                has_risk=False,
                fallback=True,
            )
            return {"query_correction_risk": False}
        self._node_completed(
            state,
            "classify_query_correction_risk",
            node_started,
            enabled=True,
            has_risk=assessment.has_risk,
            fallback=False,
        )
        return {
            "query_correction_risk": assessment.has_risk,
            "token_usage": self._token_budget.actual_usage(result),
        }

    async def _apply_user_adjudication_decision(self, state: RagGraphState) -> RagStateUpdate:
        node_started = self._node_started(state, "apply_user_adjudication_decision")
        adjudication = state["adjudication_agent_state"]
        user_decision = state["adjudication_user_decision"]
        if adjudication is None or adjudication.pending_confirmation is None or user_decision is None:
            self._node_completed(
                state,
                "apply_user_adjudication_decision",
                node_started,
                applied=False,
                pending=adjudication is not None and adjudication.pending_confirmation is not None,
            )
            return {}
        pending = adjudication.pending_confirmation
        if user_decision.request_id != pending.request_id:
            raise ValueError("Adjudication decision does not match the pending request")
        pending_by_id = {item.id: item for item in pending.items}
        decisions_by_id = {item.item_id: item for item in user_decision.decisions}
        if set(decisions_by_id) != set(pending_by_id):
            raise ValueError("Adjudication decision must resolve every pending item exactly once")
        overlays = list(adjudication.overlays)
        accepted = 0
        kept = 0
        unresolved = 0
        for item_id, item_decision in decisions_by_id.items():
            pending_item = pending_by_id[item_id]
            if item_decision.action == "accept_candidate":
                candidate = next(
                    (candidate for candidate in pending_item.candidates if candidate.id == item_decision.candidate_id),
                    None,
                )
                if candidate is None:
                    raise ValueError("Adjudication candidate does not belong to the pending item")
                overlays.append(
                    EvidenceOverlay(
                        proposal_id=candidate.id,
                        evidence_index=pending_item.evidence_index,
                        chunk_id=str(pending_item.chunk_id),
                        original_expression=pending_item.original_expression,
                        resolved_expression=candidate.expression,
                        target_spans=pending_item.target_spans,
                        status="user_confirmed",
                        confidence=candidate.confidence,
                        source_urls=candidate.source_urls,
                    )
                )
                accepted += 1
            elif item_decision.action == "keep_original":
                kept += 1
            else:
                unresolved += 1
        updated = adjudication.model_copy(
            update={
                "overlays": overlays,
                "pending_confirmation": None,
                "applied_user_decision": user_decision,
            }
        )
        self._node_completed(
            state,
            "apply_user_adjudication_decision",
            node_started,
            applied=True,
            accepted=accepted,
            kept=kept,
            unresolved=unresolved,
        )
        return {"adjudication_agent_state": updated}

    @staticmethod
    def _is_protected_lexical_query(value: str) -> bool:
        return value.strip() not in {"最近", "今天", "昨天", "这次", "这条", "那个", "这个", "是否", "是不是", "有没有"}

    def _resolve_route_filters(
        self,
        route: RagRoute,
        scope_ids: list[UUID],
        *,
        include_history_scope: bool,
    ) -> ResolvedFilters:
        preliminary = make_filters(route, None, scope_ids, include_history_scope=include_history_scope)
        recording_ids = self._retriever.resolve_recording_scope(preliminary, route.recording_limit, route.recording_rank)
        return make_filters(
            route,
            recording_ids,
            scope_ids,
            include_history_scope=include_history_scope,
        ).model_copy(update={"recording_scope_resolved": True})

    async def _retry_without_history_scope(self, state: RagGraphState) -> RagStateUpdate:
        route = state["route"]
        if route is None:
            raise RuntimeError("History-scope retry requires a resolved route")
        filters = self._resolve_route_filters(
            route,
            [UUID(value) for value in state["scope_recording_ids"]],
            include_history_scope=False,
        )
        self._node_completed(
            state,
            "retry_without_history_scope",
            self._node_started(state, "retry_without_history_scope"),
            history_recording_count=len(route.history_recording_ids),
            expanded_recording_count=len(filters.recording_ids),
        )
        return {
            "filters": filters,
            "history_scope_active": False,
            "retrieval_attempt": state["retrieval_attempt"] + 1,
            "retrieval_candidates": [],
            "protected_chunk_ids": [],
            "rerank_input_tokens": 0,
            "rerank_skipped_candidates": 0,
            "evidence": [],
            "answer_evidence": [],
            "message": None,
            "grade": None,
            "original_grade": None,
            "corrected_grade": None,
        }

    @staticmethod
    def _after_route(state: RagGraphState) -> Literal["retrieve", "unresolved"]:
        target: Literal["retrieve", "unresolved"] = "unresolved" if state["route_error"] is not None else "retrieve"
        RagGraph._transition(state, "route", target, state["route_error"] or "route_resolved")
        return target

    def _strategy_node(self, strategy_id: StrategyId) -> Callable[[RagGraphState], object]:
        async def invoke(state: RagGraphState) -> object:
            return await self._strategies.get(strategy_id).invoke(state)

        return invoke

    @staticmethod
    def _after_route_strategy(
        state: RagGraphState,
    ) -> Literal["fact_lookup", "metadata_lookup", "scope_summary", "unresolved"]:
        route = state["route"]
        if state["route_error"] is not None or route is None or route.strategy_id is None:
            RagGraph._transition(state, "route", "unresolved", state["route_error"] or "missing_strategy")
            return "unresolved"
        RagGraph._transition(state, "route", route.strategy_id, "route_resolved")
        return route.strategy_id

    async def _retrieve(self, state: RagGraphState) -> RagStateUpdate:
        node_started = self._node_started(state, "retrieve")
        route = state["route"]
        filters = state["filters"]
        if route is None or filters is None:
            raise RuntimeError("RAG graph route was not initialized")
        if filters.match_none:
            self._node_completed(
                state,
                "retrieve",
                node_started,
                outcome="empty",
                reason="match_none",
                strategy=route.strategy,
                evidence_count=0,
            )
            return {
                "retrieval_candidates": [],
                "evidence": [],
                "answer_evidence": [],
                "message": "没有找到符合范围的已完成录音",
            }
        if route.strategy == "scope_summary":
            operation_started = started_at()
            evidence = self._retriever.retrieve_scope(filters, route.recording_limit, route.recording_rank)
            self._operation_completed("retrieve", "retrieve.scope", evidence, operation_started)
            candidates: list[RetrievalCandidateRow] = []
        elif route.strategy == "chunk_search":
            query = state["content_query"]
            candidates = await self._retrieve_candidates(query, filters, state["limit"], state.get("run_id", "standalone"))
            evidence = []
        else:
            raise RuntimeError("Resolved RAG route has no retrieval strategy")
        self._node_completed(
            state,
            "retrieve",
            node_started,
            outcome="succeeded" if evidence or candidates else "empty",
            strategy=route.strategy,
            evidence_count=len(evidence),
            candidate_count=len(candidates),
            recording_count=len({item.recording.id for item in evidence}),
            requested_limit=state["limit"],
        )
        return {
            "retrieval_candidates": candidates,
            "evidence": evidence,
            "answer_evidence": [],
            "rerank_input_tokens": 0,
            "rerank_skipped_candidates": 0,
            "message": None if evidence or candidates else "没有找到足够相关的录音片段",
        }

    def _after_retrieve(self, state: RagGraphState) -> Literal["expand_context", "grade", "done"]:
        route = state["route"]
        if route is not None and route.strategy == "chunk_search" and state["retrieval_candidates"]:
            target: Literal["expand_context", "grade", "done"] = "expand_context"
            reason = "candidates_ready"
        elif state["execution_mode"] == "retrieval":
            target = "done"
            reason = "retrieval_terminal"
        else:
            target = "grade"
            reason = "scope_evidence_ready" if state["evidence"] else "no_candidates"
        self._transition(state, "retrieve", target, reason)
        return target

    async def _retrieve_candidates(
        self,
        query: str,
        filters: ResolvedFilters,
        limit: int,
        run_id: str,
        *,
        expanded_query: str | None = None,
        lexical_queries: list[str] | None = None,
        protected_lexical_queries: list[str] | None = None,
    ) -> list[RetrievalCandidateRow]:
        return await self._chunk_evidence.retrieve_candidates(
            query,
            filters,
            limit,
            run_id,
            expanded_query=expanded_query,
            lexical_queries=lexical_queries,
            protected_lexical_queries=protected_lexical_queries,
        )

    async def _rerank(self, state: RagGraphState) -> RagStateUpdate:
        return await self._chunk_evidence.rerank(state)

    async def _expand_context(self, state: RagGraphState) -> RagStateUpdate:
        return await self._chunk_evidence.expand_context(state)

    def _after_expand_context(self, state: RagGraphState) -> Literal["rerank", "grade", "done"]:
        if state["evidence"] and bool(getattr(self._retriever, "rerank_enabled", False)):
            target: Literal["rerank", "grade", "done"] = "rerank"
            reason = "expanded_evidence_ready"
        elif state["execution_mode"] == "retrieval":
            target = "done"
            reason = "retrieval_terminal"
        else:
            target = "grade"
            reason = "rerank_disabled_or_empty"
        self._transition(state, "expand_context", target, reason)
        return target

    @staticmethod
    def _after_rerank(state: RagGraphState) -> Literal["grade", "done"]:
        if state["execution_mode"] == "retrieval":
            RagGraph._transition(state, "rerank", "done", "retrieval_terminal")
            return "done"
        RagGraph._transition(state, "rerank", "grade", "rerank_completed_or_degraded")
        return "grade"

    async def _grade(self, state: RagGraphState) -> RagStateUpdate:
        evidence = state.get("answer_evidence") or state["evidence"]
        grade, token_usage = await self._grade_evidence(state, evidence, node="grade")
        return {"grade": grade, "original_grade": grade, "corrected_grade": None, "token_usage": token_usage}

    async def _grade_variants(self, state: RagGraphState) -> RagStateUpdate:
        """Grade original and adjudicated evidence concurrently after correction is final."""

        original_evidence = state.get("answer_evidence") or state["evidence"]
        adjudication = state.get("adjudication_agent_state")
        if adjudication is None or not adjudication.overlays:
            grade, token_usage = await self._grade_evidence(state, original_evidence, node="grade_original")
            return {
                "grade": grade,
                "original_grade": grade,
                "corrected_grade": None,
                "token_usage": token_usage,
            }
        corrected_evidence = apply_evidence_overlays(original_evidence, adjudication.overlays)
        original_result, corrected_result = await asyncio.gather(
            self._grade_evidence(state, original_evidence, node="grade_original"),
            self._grade_evidence(state, corrected_evidence, node="grade_corrected"),
        )
        original_grade, original_tokens = original_result
        corrected_grade, corrected_tokens = corrected_result
        return {
            "grade": corrected_grade,
            "original_grade": original_grade,
            "corrected_grade": corrected_grade,
            "token_usage": original_tokens + corrected_tokens,
        }

    async def _grade_evidence(
        self,
        state: RagGraphState,
        evidence: list[Evidence],
        *,
        node: str,
    ) -> tuple[EvidenceGrade, int]:
        node_started = self._node_started(state, node)
        route = state["route"]
        grade_query = state["query"]
        if not evidence:
            grade = EvidenceGrade(
                verdict="abstain",
                reason="empty_evidence",
            )
            self._node_completed(
                state,
                node,
                node_started,
                outcome="insufficient",
                reason=grade.reason,
                evidence_count=0,
                evidence_refs=[],
                model_execution="skipped",
                provider=None,
            )
            return grade, 0
        if route is not None and route.strategy == "scope_summary" and route.recording_limit:
            covered = {item.recording.id for item in evidence}
            if len(covered) < route.recording_limit:
                grade = EvidenceGrade(
                    verdict="abstain",
                    reason="incomplete_recording_scope",
                )
                self._node_completed(
                    state,
                    node,
                    node_started,
                    outcome="insufficient",
                    reason=grade.reason,
                    evidence_count=len(evidence),
                    evidence_refs=self._grade_evidence_refs(evidence),
                    covered_recordings=len(covered),
                    required_recordings=route.recording_limit,
                    model_execution="skipped",
                    provider=None,
                )
                return grade, 0
        prompt, values, parser = grade_prompt(grade_query, self._grade_evidence_text(evidence))
        messages = prompt.invoke(values).to_messages()
        provider = LlmProvider.LOCAL
        result = await self._complete(
            state,
            node,
            messages,
            max_tokens=500,
            json_schema=EvidenceGrade.model_json_schema(),
            provider=provider,
            enable_thinking=True
        )
        raw = result.text
        logger.debug("rag grade raw output: %s", raw)
        try:
            grade = parser.parse(raw)
        except Exception as error:
            log_event(
                "node_warning",
                state.get("run_id", "standalone"),
                level=logging.WARNING,
                node=node,
                reason="grader_parse_fallback",
                error_type=type(error).__name__,
                response_chars=len(raw),
            )
            grade = EvidenceGrade(
                verdict="abstain",
                reason="grader_parse_fallback",
            )
        self._node_completed(
            state,
            node,
            node_started,
            outcome=grade.verdict,
            verdict=grade.verdict,
            grade_query=grade_query,
            reason=grade.reason,
            evidence_count=len(evidence),
            evidence_refs=self._grade_evidence_refs(evidence),
            response_chars=len(raw),
            model_execution="local" if provider == LlmProvider.LOCAL else "online",
            provider=provider.value,
        )
        return grade, self._token_budget.actual_usage(result)

    @staticmethod
    def _grade_evidence_refs(evidence: list[Evidence]) -> list[dict[str, str]]:
        """Return log-safe evidence identities without transcript content."""

        refs: list[dict[str, str]] = []
        for item in evidence:
            ref = {
                "recording_id": str(item.recording.id),
                "chunk_id": str(item.chunk.id),
            }
            refs.append(ref)
        return refs

    @staticmethod
    def _grade_evidence_text(evidence: list[Evidence]) -> str:
        """Provide grade with transcript content and its recording context."""

        blocks: list[str] = []
        for item in evidence:
            lines = [f"证据 {item.index}：", f"录音名字：{item.recording.title}"]
            if item.recording.created_at is not None:
                lines.append(f"录音发生时间：{item.recording.created_at.isoformat()}")
            if item.recording.location and item.recording.location.strip():
                lines.append(f"录音发生地点：{item.recording.location}")
            lines.append(f"录音正文：\n{item.chunk.retrieval_text()}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    @staticmethod
    def _evidence_log_entries(evidence: list[Evidence]) -> list[dict[str, object]]:
        """Diagnostic evidence details for application logs; excluded from span metadata."""

        return [
            {
                "index": item.index,
                "recording_id": str(item.recording.id),
                "chunk_id": str(item.chunk.id),
                "start_ms": item.chunk.start_ms,
                "end_ms": item.chunk.end_ms,
            }
            for item in evidence
        ]

    @staticmethod
    async def _decide_plan(state: RagGraphState) -> RagStateUpdate:
        node_started = RagGraph._node_started(state, "decide_plan")
        grade = state["grade"]
        route = state["route"]
        if grade is None or grade.verdict == "abstain":
            raise RuntimeError("Answer planning decision requires an answerable evidence assessment")
        recording_count = len({item.recording.id for item in state["evidence"]})
        scope_override = route is not None and route.strategy == "scope_summary" and recording_count > 1
        planning_required = scope_override
        RagGraph._node_completed(
            state,
            "decide_plan",
            node_started,
            planning_required=planning_required,
            model_required=False,
            scope_override=scope_override,
            recording_count=recording_count,
            reason=grade.reason,
        )
        return {"planning_required": planning_required}

    @staticmethod
    def _after_plan_decision(state: RagGraphState) -> Literal["plan", "select_direct_evidence"]:
        target: Literal["plan", "select_direct_evidence"] = "plan" if state["planning_required"] else "select_direct_evidence"
        RagGraph._transition(
            state,
            "decide_plan",
            target,
            "planning_required" if state["planning_required"] else "planning_not_required",
        )
        return target

    async def _plan(self, state: RagGraphState) -> RagStateUpdate:
        node_started = self._node_started(state, "plan")
        if not state["evidence"]:
            raise RuntimeError("Answer planning requires evidence")
        grade = state["grade"]
        if grade is None or grade.verdict == "abstain":
            raise RuntimeError("Answer planning requires an answerable evidence assessment")
        prompt, values, parser = answer_plan_prompt(
            state["query"],
            self._evidence_text(state["evidence"]),
            grade.verdict,
        )
        messages = prompt.invoke(values).to_messages()
        provider = self._provider_for_input(messages, self._plan_local_input_tokens)
        result = await self._complete(
            state,
            "plan",
            messages,
            max_tokens=900,
            json_schema=AnswerPlan.model_json_schema(),
            provider=provider,
        )
        raw = result.text
        logger.debug("rag answer plan raw output: %s", raw)
        fallback_used = False
        try:
            plan = parser.parse(raw)
        except Exception as error:
            fallback_used = True
            log_event(
                "node_warning",
                state.get("run_id", "standalone"),
                level=logging.WARNING,
                node="plan",
                reason="plan_parse_fallback",
                error_type=type(error).__name__,
                response_chars=len(raw),
            )
            plan = self._fallback_plan(state["evidence"])
        self._node_completed(
            state,
            "plan",
            node_started,
            plan_items=len(plan.items),
            fallback_used=fallback_used,
            evidence_count=len(state["evidence"]),
            response_chars=len(raw),
            next_node="validate_plan",
            model_execution="local" if provider == LlmProvider.LOCAL else "online",
            provider=provider.value,
        )
        return {"answer_plan": plan, "token_usage": self._token_budget.actual_usage(result)}

    @staticmethod
    async def _validate_plan(state: RagGraphState) -> RagStateUpdate:
        node_started = RagGraph._node_started(state, "validate_plan")
        plan = state["answer_plan"]
        valid_indexes = {item.index for item in state["evidence"]}
        fallback_used = False
        removed_indexes = 0
        removed_items = 0
        cleaned_items: list[AnswerPlanItem] = []
        if plan is not None:
            for item in plan.items:
                indexes = list(dict.fromkeys(index for index in item.evidence_indexes if index in valid_indexes))
                removed_indexes += len(item.evidence_indexes) - len(indexes)
                if not indexes:
                    removed_items += 1
                    continue
                cleaned_items.append(item.model_copy(update={"evidence_indexes": indexes}))
        if cleaned_items:
            plan = AnswerPlan(items=cleaned_items)
        else:
            fallback_used = True
            log_event(
                "node_warning",
                state.get("run_id", "standalone"),
                level=logging.WARNING,
                node="validate_plan",
                reason="invalid_evidence_indexes",
            )
            plan = RagGraph._fallback_plan(state["evidence"])
        RagGraph._node_completed(
            state,
            "validate_plan",
            node_started,
            plan_items=len(plan.items),
            selected_evidence_count=len({index for item in plan.items for index in item.evidence_indexes}),
            fallback_used=fallback_used,
            removed_indexes=removed_indexes,
            removed_items=removed_items,
            next_node="select_planned_evidence",
        )
        return {"answer_plan": plan}

    @staticmethod
    async def _select_planned_evidence(state: RagGraphState) -> RagStateUpdate:
        node_started = RagGraph._node_started(state, "select_planned_evidence")
        plan = state["answer_plan"]
        if plan is None:
            raise RuntimeError("Planned evidence selection requires a validated answer plan")
        planned_indexes = {index for item in plan.items for index in item.evidence_indexes}
        evidence = list(state["evidence"])
        if not evidence:
            raise RuntimeError("Planned answer requires evidence")
        RagGraph._node_completed(
            state,
            "select_planned_evidence",
            node_started,
            retrieved_evidence_count=len(state["evidence"]),
            selected_evidence_count=len(evidence),
            selected_indexes=[item.index for item in evidence],
            planned_indexes=sorted(planned_indexes),
            plan_prunes_evidence=False,
        )
        return {"answer_evidence": evidence}

    @staticmethod
    def _fallback_plan(evidence: list[Evidence]) -> AnswerPlan:
        if not evidence:
            raise RuntimeError("Cannot build an answer plan without evidence")
        return AnswerPlan(items=[AnswerPlanItem(statement=item.chunk.text[:240], evidence_indexes=[item.index]) for item in evidence[:3]])

    @staticmethod
    def _evidence_text(evidence: list[Evidence]) -> str:
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
                lines.append(f"结构化说话人标签数量：{item.facts.speaker_count}")
                lines.append(f"结构化说话人标签：{labels}")
            if item.facts.utterance_count is not None:
                lines.append(f"发言段总数：{item.facts.utterance_count}")
            lines.append(f"提供给模型的正文是否截断：{'是' if item.facts.transcript_truncated else '否'}")
            lines.append(f"录音正文：\n{item.chunk.text}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    @staticmethod
    def _history_text(history: list[RagHistoryMessage], include_sources: bool = False) -> str:
        lines: list[str] = []
        for message in history:
            if message.content.strip():
                lines.append(f"{'用户' if message.role == 'user' else '助手'}：{message.content}")
            if include_sources and message.role == "assistant" and message.sources:
                lines.append("该回答引用的录音 source（仅用于判断范围）：")
                for source in message.sources:
                    lines.append(f"- recording_id：{source.recording_id}")
        return "\n".join(lines)

    @staticmethod
    def _route_history_context(history: list[RagHistoryMessage]) -> str:
        turns: list[dict[str, object]] = []
        for message in history:
            turn: dict[str, object] = {"role": message.role, "content": message.content}
            if message.role == "assistant":
                turn["sources"] = [{"recording_id": str(source.recording_id)} for source in message.sources]
            turns.append(turn)
        return json.dumps(
            turns,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _validate_selected_recording_ids(route: RagRoute, scope_recording_ids: list[UUID]) -> str | None:
        selected_ids = set(route.history_recording_ids)
        if not selected_ids:
            return None
        if not selected_ids.issubset(set(scope_recording_ids)):
            return "referenced_recording_unavailable"
        return None

    async def _complete(
        self,
        state: RagGraphState,
        node: str,
        messages: object,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
        *,
        provider: LlmProvider,
        local_model_profile: Literal["default", "rag"] | None = None,
        enable_thinking: bool = False
    ) -> LlmGenerateResult:
        self._token_budget.before_model(state.get("token_usage", 0), node)
        options = CompletionOptions(
            max_tokens=max_tokens,
            response_format=ResponseFormat(
                type=ResponseFormatType.JSON_SCHEMA,
                json_schema=json_schema,
                strict=True,
            )
            if json_schema is not None
            else ResponseFormat(),
            enbale_thinking=enable_thinking,
        )
        result = await self._model_client.execute(
            build_llm_generate_command(
                provider,
                self._worker_messages(cast(list[BaseMessage], messages)),
                options,
                context_size=self._context_size,
                stream=False,
                model_profile=(local_model_profile or self._node_model_profile) if provider == LlmProvider.LOCAL else "default",
            ),
            result_type=LlmGenerateResult,
        )
        return result

    async def _generate_streaming_answer(
        self,
        state: RagGraphState,
        messages: object,
        max_tokens: int,
        on_delta: Callable[[str], None],
        *,
        provider: LlmProvider,
    ) -> tuple[str, LlmGenerateResult]:
        async def generate() -> tuple[str, LlmGenerateResult]:
            self._token_budget.before_model(state.get("token_usage", 0), "answer")
            chunks: list[str] = []

            def publish_visible(delta: str) -> None:
                chunks.append(delta)
                on_delta(delta)

            visible_stream = ThinkTagFilter(publish_visible)
            result = await self._model_client.execute_streaming(
                build_llm_generate_command(
                    provider,
                    self._worker_messages(cast(list[BaseMessage], messages)),
                    CompletionOptions(max_tokens=max_tokens, temperature=0.1),
                    context_size=self._context_size,
                    stream=True,
                ),
                result_type=LlmGenerateResult,
                on_delta=visible_stream.feed,
            )
            visible_stream.finish()
            return "".join(chunks).strip(), result

        return await generate()

    def _provider_for_input(self, messages: object, local_limit: int) -> LlmProvider:
        worker_messages = self._worker_messages(cast(list[BaseMessage], messages))
        estimated = self._token_budget.estimate_input_tokens(worker_messages)
        return LlmProvider.LOCAL if estimated <= local_limit else self._online_provider

    @staticmethod
    def _worker_messages(messages: list[BaseMessage]) -> list[ChatMessage]:
        output: list[ChatMessage] = []
        for message in messages:
            role = ChatRole.ASSISTANT if message.type == "ai" else ChatRole.SYSTEM if message.type == "system" else ChatRole.USER
            content = message.content if isinstance(message.content, str) else str(message.content)
            output.append(ChatMessage(role, content))
        return output
    @staticmethod
    def _after_user_adjudication(state: RagGraphState) -> Literal["grade_variants", "finalize"]:
        adjudication = state.get("adjudication_agent_state")
        if adjudication is not None and adjudication.pending_confirmation is not None:
            RagGraph._transition(state, "apply_user_adjudication_decision", "finalize", "awaiting_user_confirmation")
            return "finalize"
        RagGraph._transition(state, "apply_user_adjudication_decision", "grade_variants", "adjudication_completed")
        return "grade_variants"
