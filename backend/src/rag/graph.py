from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any, Literal, cast
from uuid import UUID

from langgraph.graph import END, START, StateGraph

from rag.contracts import AnswerPlan, AnswerPlanItem, Evidence, EvidenceGrade, RagGraphState, RagHistoryMessage, RagRoute, ResolvedFilters
from rag.model import RagLanguageModel
from rag.observability import elapsed_ms, log_event, started_at
from rag.prompts import answer_plan_prompt, answer_prompt, grade_prompt, route_prompt
from rag.retrieval import RagRetriever
from rag.routing import AMBIGUOUS_RECORDING_SCOPE_MESSAGE, ROUTE_UNRESOLVED_MESSAGE, parse_route_response
from rag.scope import make_filters, resolve_date_range
from rag.streaming import ThinkTagFilter
from task_runtime.resources import ResourceQueue
from task_runtime.scheduler import ResourceScheduler

MAX_RETRIEVAL_ATTEMPTS = 1
logger = logging.getLogger("rag")


class RagGraph:
    """Typed LangGraph implementation of route, retrieval, evidence checks and answer generation."""

    def __init__(self, retriever: RagRetriever, model: RagLanguageModel, scheduler: ResourceScheduler | None = None) -> None:
        self._retriever = retriever
        self._model = model
        self._scheduler = scheduler
        builder = cast(Any, StateGraph(RagGraphState))
        builder.add_node("route", self._route)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("grade", self._grade)
        builder.add_node("rewrite", self._rewrite)
        builder.add_node("decide_plan", self._decide_plan)
        builder.add_node("plan", self._plan)
        builder.add_node("validate_plan", self._validate_plan)
        builder.add_node("select_direct_evidence", self._select_direct_evidence)
        builder.add_node("select_planned_evidence", self._select_planned_evidence)
        builder.add_edge(START, "route")
        builder.add_conditional_edges("route", self._after_route, {"retrieve": "retrieve", "unresolved": END})
        builder.add_edge("retrieve", "grade")
        builder.add_conditional_edges("grade", self._after_grade, {"rewrite": "rewrite", "decide_plan": "decide_plan", "done": END})
        builder.add_conditional_edges("rewrite", self._after_rewrite, {"retrieve": "retrieve", "grade": "grade"})
        builder.add_conditional_edges(
            "decide_plan",
            self._after_plan_decision,
            {"plan": "plan", "select_direct_evidence": "select_direct_evidence"},
        )
        builder.add_edge("plan", "validate_plan")
        builder.add_edge("validate_plan", "select_planned_evidence")
        builder.add_edge("select_direct_evidence", END)
        builder.add_edge("select_planned_evidence", END)
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
    ) -> tuple[str, list[dict[str, object]], bool, str | None]:
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
            result = await self._graph.ainvoke(
                RagGraphState(
                    run_id=trace_id,
                    query=query,
                    history=history or [],
                    limit=limit,
                    scope_recording_ids=[str(item) for item in scope_recording_ids],
                    route=None,
                    route_error=None,
                    filters=None,
                    retrieval_query=query,
                    retrieval_attempt=0,
                    evidence=[],
                    answer_evidence=[],
                    message=None,
                    grade=None,
                    planning_required=False,
                    answer_plan=None,
                )
            )
        except Exception as error:
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
        evidence = state["evidence"]
        if state["route_error"] is not None:
            message = AMBIGUOUS_RECORDING_SCOPE_MESSAGE if state["route_error"] == "ambiguous_recording_scope" else ROUTE_UNRESOLVED_MESSAGE
            log_event(
                "graph_completed",
                trace_id,
                status="route_error",
                reason=state["route_error"],
                elapsed_ms=elapsed_ms(graph_started),
            )
            return message, [], True, state["route_error"]
        grade = state["grade"]
        plan = state["answer_plan"]
        if not evidence or grade is None or not grade.sufficient:
            log_event(
                "graph_completed",
                trace_id,
                status="insufficient_evidence",
                reason=state["message"] or (grade.reason if grade is not None else "missing_grade"),
                evidence_count=len(evidence),
                elapsed_ms=elapsed_ms(graph_started),
            )
            return "没有在录音中找到足够依据。", [], True, state["message"]
        if state["planning_required"] and plan is None:
            raise RuntimeError("RAG answer planning was required but no valid plan was produced")
        answer_evidence = state["answer_evidence"]
        if not answer_evidence:
            raise RuntimeError("RAG graph completed without selecting answer evidence")
        on_phase("generating", "正在生成回答", 75)
        answer_started = self._node_started(state, "answer")
        log_event(
            "answer_mode_selected",
            trace_id,
            mode="planned" if plan is not None else "direct",
            retrieved_evidence_count=len(evidence),
            answer_evidence_count=len(answer_evidence),
        )
        prompt, values = answer_prompt(
            query,
            plan.model_dump_json() if plan is not None else None,
            self._evidence_text(answer_evidence),
            self._history_text(state["history"]),
        )
        messages = prompt.invoke(values).to_messages()
        first_token_logged = False

        def traced_delta(delta: str) -> None:
            nonlocal first_token_logged
            if not first_token_logged:
                first_token_logged = True
                log_event("answer_first_token", trace_id, elapsed_ms=elapsed_ms(answer_started))
            on_delta(delta)

        try:
            answer = await self._generate_streaming_answer(messages, 2048, traced_delta)
        except Exception as error:
            log_event(
                "node_failed",
                trace_id,
                level=logging.ERROR,
                node="answer",
                error_type=type(error).__name__,
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
            raise RuntimeError("Local RAG model returned an empty streamed answer")
        self._node_completed(
            state,
            "answer",
            answer_started,
            answer_chars=len(answer),
            evidence_count=len(answer_evidence),
            mode="planned" if plan is not None else "direct",
        )
        log_event(
            "graph_completed",
            trace_id,
            status="succeeded",
            evidence_count=len(answer_evidence),
            retrieved_evidence_count=len(evidence),
            planning_required=state["planning_required"],
            elapsed_ms=elapsed_ms(graph_started),
        )
        return answer, [item.source_payload() for item in answer_evidence], False, state["message"]

    @staticmethod
    def _node_started(state: RagGraphState, node: str) -> float:
        start = started_at()
        log_event(
            "node_started",
            state.get("run_id", "standalone"),
            node=node,
            attempt=state.get("retrieval_attempt", 0),
        )
        return start

    @staticmethod
    def _node_completed(state: RagGraphState, node: str, start: float, **fields: object) -> None:
        log_event(
            "node_completed",
            state.get("run_id", "standalone"),
            node=node,
            attempt=state.get("retrieval_attempt", 0),
            elapsed_ms=elapsed_ms(start),
            data=fields,
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

    async def _route(self, state: RagGraphState) -> dict[str, object]:
        node_started = self._node_started(state, "route")
        prompt, values, _ = route_prompt(
            state["retrieval_query"], self._route_history_messages(state["history"]), self._route_history_sources(state["history"])
        )
        raw = await self._complete(prompt.invoke(values).to_messages(), max_tokens=700, json_schema=RagRoute.model_json_schema())
        logger.debug("rag route raw output: %s", raw)
        route = parse_route_response(raw)
        if route is None:
            self._node_completed(
                state,
                "route",
                node_started,
                outcome="route_unresolved",
                query_chars=len(state["retrieval_query"]),
                response_chars=len(raw),
                raw=raw,
            )
            return {"route_error": "route_unresolved"}
        if route.status != "resolved":
            route_error = route.error_code or "route_unresolved"
            self._node_completed(
                state,
                "route",
                node_started,
                outcome=route.status,
                reason=route_error,
                response_chars=len(raw),
                raw=raw,
            )
            return {"route_error": route_error}
        if route.time_range is not None and resolve_date_range(route) == (None, None):
            self._node_completed(
                state,
                "route",
                node_started,
                outcome="unresolved",
                reason="unsupported_time_expression",
                response_chars=len(raw),
                raw=raw,
            )
            return {"route_error": "unsupported_time_expression"}
        route_error = self._validate_selected_recording_ids(route, [UUID(value) for value in state["scope_recording_ids"]])
        if route_error is not None:
            self._node_completed(
                state,
                "route",
                node_started,
                outcome="rejected",
                reason=route_error,
                response_chars=len(raw),
                raw=raw,
            )
            return {"route_error": route_error}
        scope_ids = [UUID(value) for value in state["scope_recording_ids"]]
        preliminary = make_filters(route, None, scope_ids)
        recording_ids = self._retriever.resolve_recording_scope(preliminary, route.recording_limit, route.recording_rank)
        filters = make_filters(route, recording_ids, scope_ids).model_copy(update={"recording_scope_resolved": True})
        self._node_completed(
            state,
            "route",
            node_started,
            outcome="resolved",
            strategy=route.strategy,
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
        }

    @staticmethod
    def _after_route(state: RagGraphState) -> Literal["retrieve", "unresolved"]:
        target: Literal["retrieve", "unresolved"] = "unresolved" if state["route_error"] is not None else "retrieve"
        RagGraph._transition(state, "route", target, state["route_error"] or "route_resolved")
        return target

    async def _retrieve(self, state: RagGraphState) -> dict[str, object]:
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
            return {"evidence": [], "answer_evidence": [], "message": "没有找到符合范围的已完成录音"}
        if route.strategy == "scope_summary":
            evidence = self._retriever.retrieve_scope(filters, route.recording_limit, route.recording_rank)
        elif route.strategy == "chunk_search":
            query = route.topic or state["retrieval_query"]
            evidence = await self._retrieve_chunks(query, filters, state["limit"], state.get("run_id", "standalone"))
        else:
            raise RuntimeError("Resolved RAG route has no retrieval strategy")
        self._node_completed(
            state,
            "retrieve",
            node_started,
            outcome="succeeded" if evidence else "empty",
            strategy=route.strategy,
            evidence_count=len(evidence),
            recording_count=len({item.recording.id for item in evidence}),
            requested_limit=state["limit"],
        )
        return {
            "evidence": evidence,
            "answer_evidence": [],
            "message": None if evidence else "没有找到足够相关的录音片段",
        }

    async def _retrieve_chunks(
        self,
        query: str,
        filters: ResolvedFilters,
        limit: int,
        run_id: str,
    ) -> list[Evidence]:
        if not self._retriever.hybrid_search_enabled:
            if self._scheduler is None:
                return await asyncio.to_thread(self._retriever.retrieve_chunks, query, filters, limit, run_id)
            return await self._scheduler.submit(
                ResourceQueue.GPU_HIGH,
                lambda: self._retriever.retrieve_chunks(query, filters, limit, run_id),
            )

        started = started_at()
        lexical_task = asyncio.create_task(
            asyncio.to_thread(self._retriever.retrieve_lexical_candidates, query, filters)
        )
        vector_rows: list[dict[str, object]] = []
        lexical_rows: list[dict[str, object]] = []
        vector_error: Exception | None = None
        lexical_error: Exception | None = None
        try:
            embedding = (
                await asyncio.to_thread(self._retriever.generate_query_embedding, query)
                if self._scheduler is None
                else await self._scheduler.submit(ResourceQueue.GPU_HIGH, lambda: self._retriever.generate_query_embedding(query))
            )
            vector_rows = await asyncio.to_thread(self._retriever.retrieve_vector_candidates, embedding, filters)
        except Exception as error:
            vector_error = error
            log_event(
                "retrieval_branch_failed",
                run_id,
                level=logging.WARNING,
                exc_info=True,
                branch="vector",
                error_type=type(error).__name__,
            )
        try:
            lexical_rows = await lexical_task
        except Exception as error:
            lexical_error = error
            log_event(
                "retrieval_branch_failed",
                run_id,
                level=logging.WARNING,
                exc_info=True,
                branch="lexical",
                error_type=type(error).__name__,
            )
        if vector_error is not None and lexical_error is not None:
            raise RuntimeError("Both RAG hybrid retrieval branches failed") from vector_error
        fused = self._retriever.fuse_candidates(vector_rows, lexical_rows, limit)
        evidence = await asyncio.to_thread(self._retriever.expand_candidates, fused)
        overlap = len({row["chunk_id"] for row in vector_rows} & {row["chunk_id"] for row in lexical_rows})
        log_event(
            "hybrid_retrieval_completed",
            run_id,
            query_chars=len(query),
            scope_recording_count=len(filters.recording_ids),
            vector_candidates=len(vector_rows),
            lexical_candidates=len(lexical_rows),
            overlap=overlap,
            fused_candidates=len(fused),
            evidence_count=len(evidence),
            vector_degraded=vector_error is not None,
            lexical_degraded=lexical_error is not None,
            elapsed_ms=elapsed_ms(started),
        )
        return evidence

    async def _grade(self, state: RagGraphState) -> dict[str, object]:
        node_started = self._node_started(state, "grade")
        evidence = state["evidence"]
        route = state["route"]
        grade_query = state["query"]
        retrieval_fallback = (route.topic if route is not None else None) or state["retrieval_query"]
        if not evidence:
            grade = EvidenceGrade(sufficient=False, rewrite_query=retrieval_fallback, reason="empty_evidence")
            self._node_completed(state, "grade", node_started, outcome="insufficient", reason=grade.reason, evidence_count=0)
            return {"grade": grade}
        if route is not None and route.strategy == "scope_summary" and route.recording_limit:
            covered = {item.recording.id for item in evidence}
            if len(covered) < route.recording_limit:
                grade = EvidenceGrade(sufficient=False, rewrite_query=retrieval_fallback, reason="incomplete_recording_scope")
                self._node_completed(
                    state,
                    "grade",
                    node_started,
                    outcome="insufficient",
                    reason=grade.reason,
                    evidence_count=len(evidence),
                    covered_recordings=len(covered),
                    required_recordings=route.recording_limit,
                )
                return {"grade": grade}
        prompt, values, parser = grade_prompt(grade_query, self._evidence_text(evidence))
        raw = await self._complete(prompt.invoke(values).to_messages(), max_tokens=500, json_schema=EvidenceGrade.model_json_schema())
        logger.debug("rag grade raw output: %s", raw)
        try:
            grade = parser.parse(raw)
        except Exception as error:
            log_event(
                "node_warning",
                state.get("run_id", "standalone"),
                level=logging.WARNING,
                node="grade",
                reason="grader_parse_fallback",
                error_type=type(error).__name__,
                response_chars=len(raw),
            )
            grade = EvidenceGrade(sufficient=True, reason="grader_parse_fallback")
        self._node_completed(
            state,
            "grade",
            node_started,
            outcome="sufficient" if grade.sufficient else "insufficient",
            reason=grade.reason,
            planning_required=grade.planning_required,
            evidence_count=len(evidence),
            response_chars=len(raw),
        )
        return {"grade": grade}

    @staticmethod
    def _after_grade(state: RagGraphState) -> Literal["rewrite", "decide_plan", "done"]:
        grade = state["grade"]
        route = state["route"]
        if grade is not None and grade.sufficient:
            target: Literal["rewrite", "decide_plan", "done"] = "decide_plan"
            reason = "evidence_sufficient"
        elif route is not None and grade is not None and state["retrieval_attempt"] < MAX_RETRIEVAL_ATTEMPTS:
            target = "rewrite"
            reason = grade.reason or "evidence_insufficient"
        else:
            target = "done"
            reason = grade.reason if grade is not None else "missing_grade"
        RagGraph._transition(state, "grade", target, reason)
        return target

    @staticmethod
    async def _decide_plan(state: RagGraphState) -> dict[str, object]:
        node_started = RagGraph._node_started(state, "decide_plan")
        grade = state["grade"]
        route = state["route"]
        if grade is None or not grade.sufficient:
            raise RuntimeError("Answer planning decision requires a sufficient evidence grade")
        recording_count = len({item.recording.id for item in state["evidence"]})
        scope_override = route is not None and route.strategy == "scope_summary" and recording_count > 1
        planning_required = grade.planning_required or scope_override
        RagGraph._node_completed(
            state,
            "decide_plan",
            node_started,
            planning_required=planning_required,
            model_required=grade.planning_required,
            scope_override=scope_override,
            recording_count=recording_count,
            reason=grade.planning_reason,
        )
        return {"planning_required": planning_required}

    @staticmethod
    def _after_plan_decision(state: RagGraphState) -> Literal["plan", "select_direct_evidence"]:
        target: Literal["plan", "select_direct_evidence"] = (
            "plan" if state["planning_required"] else "select_direct_evidence"
        )
        RagGraph._transition(
            state,
            "decide_plan",
            target,
            "planning_required" if state["planning_required"] else "planning_not_required",
        )
        return target

    @staticmethod
    async def _rewrite(state: RagGraphState) -> dict[str, object]:
        node_started = RagGraph._node_started(state, "rewrite")
        grade = state["grade"]
        route = state["route"]
        rewritten = (grade.rewrite_query if grade is not None else None) or (route.topic if route is not None else None) or state["query"]
        rewritten_route = route.model_copy(update={"topic": rewritten}) if route is not None and route.strategy == "chunk_search" else route
        update: dict[str, object] = {
            "retrieval_query": rewritten,
            "retrieval_attempt": state["retrieval_attempt"] + 1,
            "route": rewritten_route,
            "grade": None,
            "planning_required": False,
            "answer_plan": None,
            "answer_evidence": [],
        }
        if rewritten_route is None or rewritten_route.strategy == "chunk_search":
            update["evidence"] = []
        RagGraph._node_completed(
            state,
            "rewrite",
            node_started,
            strategy=route.strategy if route is not None else None,
            rewritten_query_chars=len(rewritten),
            filters_preserved=state["filters"] is not None,
            next_attempt=state["retrieval_attempt"] + 1,
            evidence_cleared="evidence" in update,
        )
        return update

    @staticmethod
    def _after_rewrite(state: RagGraphState) -> Literal["retrieve", "grade"]:
        route = state["route"]
        target: Literal["retrieve", "grade"] = "grade" if route is not None and route.strategy == "scope_summary" else "retrieve"
        RagGraph._transition(
            state,
            "rewrite",
            target,
            "reuse_scope_evidence" if target == "grade" else "retrieve_rewritten_query",
        )
        return target

    async def _plan(self, state: RagGraphState) -> dict[str, object]:
        node_started = self._node_started(state, "plan")
        if not state["evidence"]:
            raise RuntimeError("Answer planning requires evidence")
        prompt, values, parser = answer_plan_prompt(state["query"], self._evidence_text(state["evidence"]))
        raw = await self._complete(prompt.invoke(values).to_messages(), max_tokens=900, json_schema=AnswerPlan.model_json_schema())
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
        )
        return {"answer_plan": plan}

    @staticmethod
    async def _validate_plan(state: RagGraphState) -> dict[str, object]:
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
    async def _select_direct_evidence(state: RagGraphState) -> dict[str, object]:
        node_started = RagGraph._node_started(state, "select_direct_evidence")
        evidence = list(state["evidence"])
        RagGraph._node_completed(
            state,
            "select_direct_evidence",
            node_started,
            retrieved_evidence_count=len(evidence),
            selected_evidence_count=len(evidence),
        )
        return {"answer_evidence": evidence}

    @staticmethod
    async def _select_planned_evidence(state: RagGraphState) -> dict[str, object]:
        node_started = RagGraph._node_started(state, "select_planned_evidence")
        plan = state["answer_plan"]
        if plan is None:
            raise RuntimeError("Planned evidence selection requires a validated answer plan")
        selected_indexes = {index for item in plan.items for index in item.evidence_indexes}
        evidence = [item for item in state["evidence"] if item.index in selected_indexes]
        if not evidence:
            raise RuntimeError("Validated answer plan selected no evidence")
        RagGraph._node_completed(
            state,
            "select_planned_evidence",
            node_started,
            retrieved_evidence_count=len(state["evidence"]),
            selected_evidence_count=len(evidence),
            selected_indexes=sorted(selected_indexes),
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
                    time_range = "" if source.start_ms is None or source.end_ms is None else f"，时间范围：{source.start_ms}-{source.end_ms}ms"
                    lines.append(f"- recording_id：{source.recording_id}，标题：{source.title}{time_range}")
        return "\n".join(lines)

    @staticmethod
    def _route_history_messages(history: list[RagHistoryMessage]) -> str:
        return json.dumps(
            [{"role": message.role, "content": message.content} for message in history],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _route_history_sources(history: list[RagHistoryMessage]) -> str:
        return json.dumps(
            [
                {
                    "message_index": message_index,
                    "recording_id": str(source.recording_id),
                    "title": source.title,
                    "start_ms": source.start_ms,
                    "end_ms": source.end_ms,
                }
                for message_index, message in enumerate(history)
                if message.role == "assistant"
                for source in message.sources
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _validate_selected_recording_ids(route: RagRoute, scope_recording_ids: list[UUID]) -> str | None:
        selected_ids = set(route.inferred_filters.recording_ids)
        if not selected_ids:
            return None
        if not selected_ids.issubset(set(scope_recording_ids)):
            return "referenced_recording_unavailable"
        return None

    async def _complete(self, messages: object, max_tokens: int, json_schema: dict[str, Any] | None = None) -> str:
        if self._scheduler is None:
            return await self._model.complete(cast(Any, messages), max_tokens=max_tokens, json_schema=json_schema)
        return await self._scheduler.submit(
            ResourceQueue.GPU_HIGH,
            lambda: self._model.complete(cast(Any, messages), max_tokens=max_tokens, json_schema=json_schema),
        )

    async def _generate_streaming_answer(self, messages: object, max_tokens: int, on_delta: Callable[[str], None]) -> str:
        async def generate() -> str:
            chunks: list[str] = []

            def publish_visible(delta: str) -> None:
                chunks.append(delta)
                on_delta(delta)

            visible_stream = ThinkTagFilter(publish_visible)
            async for delta in self._model.stream(cast(Any, messages), max_tokens=max_tokens):
                if delta:
                    visible_stream.feed(delta)
            visible_stream.finish()
            return "".join(chunks).strip()

        if self._scheduler is None:
            return await generate()
        return await self._scheduler.submit(ResourceQueue.GPU_HIGH, generate)
