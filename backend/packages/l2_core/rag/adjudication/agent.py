from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

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
from l1_foundation.observability import InstrumentedModelClient
from l2_core.rag.adjudication.contracts import (
    AdjudicationAgentState,
    AdjudicationProposal,
    AdjudicationReview,
    CandidateDecision,
    CandidateDecisionBatch,
    EvidenceAdjudicationCaseState,
    EvidenceOverlay,
    ExpressionAudit,
    ExpressionAuditItem,
    ExpressionTargetSpan,
    GroundedResearchFinding,
)
from l2_core.rag.adjudication.prompts import (
    AuditPromptVariant,
    adjudication_agent_prompt,
    evidence_review_prompt,
    expression_audit_prompt,
)
from l2_core.rag.adjudication.web_research import GroundedSearchClient
from l2_core.rag.contracts import AnswerPlan, AnswerPlanItem, Evidence, RagGraphState, RagStateUpdate
from l2_core.rag.execution_middleware import rag_execution_middleware
from l2_core.rag.token_budget import RagTokenBudgetMiddleware

logger = logging.getLogger("rag")
StructuredCompleter = Callable[[list[BaseMessage], dict[str, Any]], Awaitable[LlmGenerateResult]]
EventLogger = Callable[..., None]
_MAX_REFERENCE_EVIDENCE = 5


class NodeStarted(Protocol):
    def __call__(self, state: RagGraphState, node: str) -> float: ...


class NodeCompleted(Protocol):
    def __call__(self, state: RagGraphState, node: str, start: float, **fields: object) -> None: ...


@dataclass(frozen=True, slots=True)
class AdjudicationCaseContext:
    query: str
    plan: AnswerPlan
    evidence: Evidence
    reference_evidence: list[Evidence]
    run_id: str


@dataclass(frozen=True, slots=True)
class AdjudicationTransition:
    state: AdjudicationAgentState
    outcome: str
    operation: str | None = None
    completion: LlmGenerateResult | None = None
    warning_error_type: str | None = None
    terminal: bool = False


class EvidenceAdjudicationAgent:
    """Bounded candidate-decision agent for one isolated Evidence case at a time."""

    def __init__(
        self,
        *,
        model_client: InstrumentedModelClient,
        online_provider: LlmProvider,
        context_size: int,
        token_budget: RagTokenBudgetMiddleware,
        grounded_search_client: GroundedSearchClient | None,
        web_search_enabled: bool,
        auto_resolve_confidence: float,
        max_cases: int,
        max_iterations: int,
        max_searches: int,
        audit_prompt_variant: AuditPromptVariant = "relation_rules",
        audit_model: str | None = None,
        audit_min_request_interval_seconds: float | None = None,
        node_started: NodeStarted,
        node_completed: NodeCompleted,
        event_logger: EventLogger,
    ) -> None:
        self._model_client = model_client
        self._online_provider = online_provider
        self._context_size = context_size
        self._token_budget = token_budget
        self._grounded_search_client = grounded_search_client
        self._web_search_enabled = web_search_enabled
        self._auto_resolve_confidence = auto_resolve_confidence
        self._max_cases = max_cases
        self._max_iterations = max_iterations
        self._max_searches = max_searches
        self._audit_prompt_variant: AuditPromptVariant = audit_prompt_variant
        self._audit_model = audit_model
        self._audit_min_request_interval_seconds = audit_min_request_interval_seconds
        self._node_started = node_started
        self._node_completed = node_completed
        self._event_logger = event_logger

        builder = cast(Any, StateGraph(RagGraphState))
        builder.add_node(
            "adjudication_agent_step",
            rag_execution_middleware.wrap_node(
                self._agent_step_node,
                graph_name="fact_lookup",
                node_name="adjudication_agent_step",
            ),
        )
        builder.add_node(
            "adjudication_execute_operation",
            rag_execution_middleware.wrap_node(
                self._execute_operation_node,
                graph_name="fact_lookup",
                node_name="adjudication_execute_operation",
            ),
        )
        builder.add_edge(START, "adjudication_agent_step")
        builder.add_conditional_edges(
            "adjudication_agent_step",
            self._after_agent_step,
            {"execute": "adjudication_execute_operation", "next": "adjudication_agent_step", "done": END},
        )
        builder.add_conditional_edges(
            "adjudication_execute_operation",
            self._after_operation,
            {"next": "adjudication_agent_step", "done": END},
        )
        self._graph: Any = builder.compile()

    async def start(self, state: RagGraphState) -> RagStateUpdate:
        """Initialize if needed, then run the checkpointable internal agent graph to completion."""

        initial_tokens = state.get("token_usage", 0)
        agent = state["adjudication_agent_state"]
        answer_evidence = state["answer_evidence"] or state["evidence"]
        if agent is None:
            risk = state["query_correction_risk"]
            if not risk:
                self._event_logger(
                    "adjudication_agent_skipped",
                    state.get("run_id", "standalone"),
                    reason="has_risk_false",
                )
                return {"adjudication_agent_state": None, "token_usage": 0}
            if not answer_evidence:
                raise RuntimeError("Evidence adjudication requires final evidence")
            agent = self.initialize(
                risk,
                answer_evidence,
                web_search_enabled=self._web_search_enabled,
            )
        run_id = state.get("run_id", "standalone")
        self._event_logger(
            "adjudication_agent_started",
            run_id,
            has_risk=agent.risk,
            case_count=len(agent.cases),
            current_case=agent.current_case,
            web_search_enabled=agent.web_search_enabled,
        )
        inner_state = cast(
            RagGraphState,
            {**state, "answer_evidence": answer_evidence, "adjudication_agent_state": agent},
        )
        result = cast(RagGraphState, await self._graph.ainvoke(inner_state))
        completed = result["adjudication_agent_state"]
        self._event_logger(
            "adjudication_agent_completed",
            run_id,
            status=completed.status if completed is not None else "skipped",
            overlay_count=len(completed.overlays) if completed is not None else 0,
            confirmation_count=(
                len(completed.pending_confirmation.items)
                if completed is not None and completed.pending_confirmation is not None
                else 0
            ),
        )
        return {
            "answer_evidence": answer_evidence,
            "adjudication_agent_state": result["adjudication_agent_state"],
            "token_usage": max(0, result.get("token_usage", 0) - initial_tokens),
        }

    async def _agent_step_node(self, state: RagGraphState) -> RagStateUpdate:
        node_started = self._node_started(state, "adjudication_agent_step")
        agent = state["adjudication_agent_state"]
        if agent is None:
            self._node_completed(state, "adjudication_agent_step", node_started, outcome="completed")
            return {}
        context = self._context(state, agent)

        async def complete(messages: list[BaseMessage], schema: dict[str, Any]) -> LlmGenerateResult:
            return await self._complete_candidate_decisions(state, messages, schema)

        transition = await self.decide_candidate_actions(
            agent,
            context,
            complete=complete,
        )
        if transition.warning_error_type is not None:
            self._warning(state, "adjudication_agent_step", "action_selection_failed", transition)
        self._event_logger(
            "adjudication_agent_next_operation_selected",
            state.get("run_id", "standalone"),
            case_index=agent.current_case,
            outcome=transition.outcome,
            operation=transition.operation,
            error_type=transition.warning_error_type,
        )
        self._node_completed(
            state,
            "adjudication_agent_step",
            node_started,
            outcome=transition.outcome,
            operation=transition.operation,
            model_execution="online" if transition.completion is not None else "none",
        )
        output: RagStateUpdate = {"adjudication_agent_state": transition.state}
        if transition.completion is not None:
            output["token_usage"] = self._token_budget.actual_usage(transition.completion)
        return output

    async def _execute_operation_node(self, state: RagGraphState) -> RagStateUpdate:
        node_started = self._node_started(state, "adjudication_execute_operation")
        agent = state["adjudication_agent_state"]
        if agent is None:
            return {}
        context = self._context(state, agent)

        async def complete_audit(messages: list[BaseMessage], schema: dict[str, Any]) -> LlmGenerateResult:
            return await self._complete_expression_audit(state, messages, schema)

        async def reconstruct(messages: list[BaseMessage], schema: dict[str, Any]) -> LlmGenerateResult:
            return await self._complete_candidate_reconstruction(state, messages, schema)

        current_case = agent.cases[agent.current_case] if agent.current_case < len(agent.cases) else None
        pending_setup_phase = current_case.pending_setup_phase if current_case is not None else None
        pending_decisions = current_case.pending_decisions if current_case is not None else None
        operation = pending_setup_phase or ("candidate_actions" if pending_decisions is not None else None)
        self._event_logger(
            "adjudication_agent_operation_started",
            state.get("run_id", "standalone"),
            case_index=agent.current_case,
            operation=operation,
            iteration=agent.cases[agent.current_case].iteration if agent.current_case < len(agent.cases) else 0,
            search_count=agent.cases[agent.current_case].search_count if agent.current_case < len(agent.cases) else 0,
        )
        if pending_setup_phase is not None:
            transition = await self.execute_setup_phase(
                agent,
                context,
                complete_audit=complete_audit,
                reconstruct=reconstruct,
            )
        elif pending_decisions is not None:
            transition = await self.execute_candidate_actions(agent, context, reconstruct=reconstruct)
        else:
            raise RuntimeError("Adjudication operation execution requires a pending phase or decisions")
        if transition.warning_error_type is not None:
            self._warning(state, "adjudication_execute_operation", "operation_failed", transition)
        self._event_logger(
            "adjudication_agent_operation_completed",
            state.get("run_id", "standalone"),
            case_index=agent.current_case,
            operation=transition.operation,
            outcome=transition.outcome,
            terminal=transition.terminal,
            succeeded=transition.warning_error_type is None,
            error_type=transition.warning_error_type,
        )
        self._node_completed(
            state,
            "adjudication_execute_operation",
            node_started,
            outcome=transition.outcome,
            operation=transition.operation,
            succeeded=transition.warning_error_type is None,
            terminal=transition.terminal,
            model_execution="online" if transition.completion is not None else "none",
        )
        output: RagStateUpdate = {"adjudication_agent_state": transition.state}
        if transition.completion is not None:
            output["token_usage"] = self._token_budget.actual_usage(transition.completion)
        return output

    def initialize(self, risk: bool, evidence: list[Evidence], *, web_search_enabled: bool) -> AdjudicationAgentState:
        return AdjudicationAgentState(
            risk=risk,
            cases=[
                EvidenceAdjudicationCaseState(evidence_index=item.index, chunk_id=item.chunk.id)
                for item in evidence[: self._max_cases]
            ],
            web_search_enabled=web_search_enabled,
        )

    async def decide_candidate_actions(
        self,
        state: AdjudicationAgentState,
        context: AdjudicationCaseContext,
        *,
        complete: StructuredCompleter,
    ) -> AdjudicationTransition:
        case = self._current_case(state)
        if case is None:
            return AdjudicationTransition(state, "completed", terminal=True)

        if case.pending_setup_phase is not None:
            return AdjudicationTransition(state, "resume_setup", operation=case.pending_setup_phase)
        if case.pending_decisions is not None:
            return AdjudicationTransition(state, "resume_actions", operation="candidate_actions")
        setup_phase = self._required_setup_phase(case)
        if setup_phase is not None:
            stepped = case.model_copy(update={"pending_setup_phase": setup_phase})
            return AdjudicationTransition(
                self._replace_case(state, stepped),
                "setup_phase",
                operation=setup_phase,
            )
        if not case.proposals:
            return AdjudicationTransition(
                self._finish_case_without_confirmation(state, case, reason="candidates_converged"),
                "candidates_converged",
                terminal=True,
            )
        if case.iteration >= self._max_iterations:
            return AdjudicationTransition(
                self._finish_case_without_confirmation(state, case, reason="iteration_budget_exhausted"),
                "budget_exhausted",
                warning_error_type="IterationBudgetExhausted",
                terminal=True,
            )

        prompt, values = adjudication_agent_prompt(
            context.query,
            state.risk,
            context.plan,
            context.evidence,
            context.reference_evidence,
            case,
            self._max_iterations,
            self._max_searches,
            control={
                "web_search_enabled": state.web_search_enabled and self._grounded_search_client is not None,
                "remaining_searches": max(0, self._max_searches - case.search_count),
            },
        )
        completion: LlmGenerateResult | None = None
        try:
            completion = await complete(
                prompt.invoke(values).to_messages(),
                CandidateDecisionBatch.model_json_schema(),
            )
            self._log_structured_completion(context, state.current_case, case.iteration, "candidate_decisions", completion)
            decisions = CandidateDecisionBatch.model_validate_json(completion.text)
            decisions, missing_ids = self._normalize_candidate_decisions(case, decisions, context)
            if missing_ids:
                retry_case = case.model_copy(
                    update={"proposals": [proposal for proposal in case.proposals if proposal.id in missing_ids]}
                )
                logger.warning(
                    "Evidence 裁决 Agent Candidate 决策缺失，开始单次补全 run_id=%s case=%d missing_proposal_ids=%s",
                    context.run_id,
                    state.current_case,
                    missing_ids,
                )
                retry_prompt, retry_values = adjudication_agent_prompt(
                    context.query,
                    state.risk,
                    context.plan,
                    context.evidence,
                    context.reference_evidence,
                    retry_case,
                    self._max_iterations,
                    self._max_searches,
                    control={
                        "web_search_enabled": state.web_search_enabled and self._grounded_search_client is not None,
                        "remaining_searches": max(0, self._max_searches - case.search_count),
                    },
                )
                retry_completion = await complete(
                    retry_prompt.invoke(retry_values).to_messages(),
                    CandidateDecisionBatch.model_json_schema(),
                )
                self._log_structured_completion(
                    context,
                    state.current_case,
                    case.iteration,
                    "candidate_decisions_retry",
                    retry_completion,
                )
                completion = retry_completion
                retry_decisions = CandidateDecisionBatch.model_validate_json(retry_completion.text)
                retry_decisions, remaining_missing_ids = self._normalize_candidate_decisions(
                    retry_case, retry_decisions, context
                )
                if remaining_missing_ids:
                    raise ValueError(f"candidate decisions are missing active proposals: {remaining_missing_ids}")
                decisions = CandidateDecisionBatch(decisions=[*decisions.decisions, *retry_decisions.decisions])
            self._validate_candidate_decisions(state, case, decisions, context)
        except Exception as error:
            failed = case.model_copy(update={"iteration": case.iteration + 1, "error": type(error).__name__})
            exhausted = failed.iteration >= self._max_iterations
            logger.warning(
                "Evidence 裁决 Agent Candidate 决策失败 run_id=%s case=%d iteration=%d exhausted=%s error=%s",
                context.run_id,
                state.current_case,
                failed.iteration,
                exhausted,
                json.dumps(self._error_for_log(error), ensure_ascii=False, separators=(",", ":")),
            )
            return AdjudicationTransition(
                (
                    self._finish_case_without_confirmation(state, failed, reason=f"agent:{type(error).__name__}")
                    if exhausted
                    else self._replace_case(state, failed)
                ),
                "action_selection_failed",
                completion=completion,
                warning_error_type=type(error).__name__,
                terminal=exhausted,
            )
        stepped = case.model_copy(update={"pending_decisions": decisions, "iteration": case.iteration + 1})
        return AdjudicationTransition(
            self._replace_case(state, stepped),
            "candidate_actions",
            operation="candidate_actions",
            completion=completion,
        )

    async def execute_setup_phase(
        self,
        state: AdjudicationAgentState,
        context: AdjudicationCaseContext,
        *,
        complete_audit: StructuredCompleter,
        reconstruct: StructuredCompleter,
    ) -> AdjudicationTransition:
        """Execute the deterministic audit -> initial reconstruction prefix for a case."""

        case = self._current_case(state)
        if case is None:
            return AdjudicationTransition(state, "completed", terminal=True)
        phase = case.pending_setup_phase
        if phase is None:
            raise RuntimeError("Adjudication setup execution requires a pending setup phase")
        completion: LlmGenerateResult | None = None
        try:
            if phase == "audit":
                prompt, values = expression_audit_prompt(
                    context.query,
                    context.plan,
                    context.evidence,
                    context.reference_evidence,
                    focus="",
                    variant=self._audit_prompt_variant,
                )
                completion = await complete_audit(
                    prompt.invoke(values).to_messages(),
                    ExpressionAudit.model_json_schema(),
                )
                self._log_structured_completion(context, state.current_case, case.iteration, phase, completion)
                audit = self._valid_audit(context, ExpressionAudit.model_validate_json(completion.text))
                logger.info(
                    "Evidence 裁决 Agent 最终审计 run_id=%s case=%d item_count=%d items=%s",
                    context.run_id,
                    state.current_case,
                    len(audit.items),
                    json.dumps([item.model_dump(mode="json") for item in audit.items], ensure_ascii=False, separators=(",", ":")),
                )

                completed = case.model_copy(update={"expression_audit": audit, "pending_setup_phase": None})
                if not audit.items:
                    completed = completed.model_copy(update={"status": "rejected"})
                    return AdjudicationTransition(
                        self._advance_case(state, completed),
                        "audit_empty",
                        operation=phase,
                        completion=completion,
                        terminal=True,
                    )
                return AdjudicationTransition(
                    self._replace_case(state, completed),
                    "audit_completed",
                    operation=phase,
                    completion=completion,
                )

            if case.expression_audit is None:
                raise RuntimeError("Initial candidate reconstruction requires expression audit")
            prompt, values = evidence_review_prompt(
                context.query,
                context.plan,
                context.evidence,
                reference_evidence=context.reference_evidence,
                expression_audit=case.expression_audit,
                findings=json.dumps([item.model_dump(mode="json") for item in case.findings], ensure_ascii=False),
                focus="",
            )
            completion = await reconstruct(
                prompt.invoke(values).to_messages(),
                AdjudicationReview.model_json_schema(),
            )
            self._log_structured_completion(context, state.current_case, case.iteration, phase, completion)
            review, rejected_proposals = self._parse_review(completion.text)
            if rejected_proposals:
                logger.warning(
                    "Evidence 裁决 Agent 过滤无效候选 run_id=%s case=%d rejected=%s",
                    context.run_id,
                    state.current_case,
                    json.dumps(rejected_proposals, ensure_ascii=False, separators=(",", ":")),
                )
            proposals = self._valid_proposals(case, context, review.proposals)
            required_audit_ids = {item.id for item in case.expression_audit.items}
            missing_audit_ids = required_audit_ids - {proposal.audit_item_id for proposal in proposals}
            if missing_audit_ids:
                logger.warning(
                    "Evidence 裁决 Agent 首次候选重建缺少审计项，开始单次重试 run_id=%s case=%d missing_audit_ids=%s",
                    context.run_id,
                    state.current_case,
                    sorted(missing_audit_ids),
                )
                missing_audit = ExpressionAudit(
                    items=[item for item in case.expression_audit.items if item.id in missing_audit_ids]
                )
                retry_prompt, retry_values = evidence_review_prompt(
                    context.query,
                    context.plan,
                    context.evidence,
                    reference_evidence=context.reference_evidence,
                    expression_audit=missing_audit,
                    findings=json.dumps([item.model_dump(mode="json") for item in case.findings], ensure_ascii=False),
                    focus="首次重建遗漏了以下 Audit item；仅重建这些 item，并确保每项输出 1 至 2 个 proposal："
                    + json.dumps(sorted(missing_audit_ids), ensure_ascii=False),
                )
                completion = await reconstruct(
                    retry_prompt.invoke(retry_values).to_messages(),
                    AdjudicationReview.model_json_schema(),
                )
                self._log_structured_completion(
                    context, state.current_case, case.iteration, "initial_reconstruct_retry", completion
                )
                retry_review, retry_rejected_proposals = self._parse_review(completion.text)
                if retry_rejected_proposals:
                    logger.warning(
                        "Evidence 裁决 Agent 过滤重试无效候选 run_id=%s case=%d rejected=%s",
                        context.run_id,
                        state.current_case,
                        json.dumps(retry_rejected_proposals, ensure_ascii=False, separators=(",", ":")),
                    )
                for proposal in self._valid_proposals(case, context, retry_review.proposals):
                    if proposal.audit_item_id not in missing_audit_ids or len(proposals) >= 40:
                        continue
                    proposals.append(proposal)

            completed = case.model_copy(
                update={
                    "proposals": proposals,
                    "initial_reconstruction_completed": True,
                    "pending_setup_phase": None,
                }
            )
            return AdjudicationTransition(
                self._replace_case(state, completed),
                "initial_reconstruct_completed",
                operation=phase,
                completion=completion,
            )
        except Exception as error:
            error_name = type(error).__name__
            logger.warning(
                "Evidence 裁决 Agent 固定阶段失败 run_id=%s case=%d operation=%s error=%s",
                context.run_id,
                state.current_case,
                phase,
                json.dumps(self._error_for_log(error), ensure_ascii=False, separators=(",", ":")),
            )
            failed = case.model_copy(update={"pending_setup_phase": None, "error": error_name})
            return AdjudicationTransition(
                self._finish_case_without_confirmation(state, failed, reason=f"setup:{phase}:{error_name}"),
                "setup_failed",
                operation=phase,
                completion=completion,
                warning_error_type=error_name,
                terminal=True,
            )

    async def execute_candidate_actions(
        self,
        state: AdjudicationAgentState,
        context: AdjudicationCaseContext,
        *,
        reconstruct: StructuredCompleter,
    ) -> AdjudicationTransition:
        case = self._current_case(state)
        if case is None:
            return AdjudicationTransition(state, "completed", terminal=True)
        batch = case.pending_decisions
        if batch is None:
            raise RuntimeError("Candidate action execution requires pending decisions")
        self._validate_candidate_decisions(state, case, batch, context)

        proposals_by_id = {proposal.id: proposal for proposal in case.proposals}
        decisions_by_audit: dict[str, list[CandidateDecision]] = {}
        for decision in batch.decisions:
            audit_item_id = proposals_by_id[decision.proposal_id].audit_item_id
            decisions_by_audit.setdefault(audit_item_id, []).append(decision)

        selected_accepted, overlap_rejected_ids = self._select_non_overlapping_accepted(state, case, batch, context)
        selected_accepted_ids = {decision.proposal_id for decision in selected_accepted}
        if overlap_rejected_ids:
            logger.warning(
                "Evidence 裁决 Agent 接受候选重叠，保留更高 candidate_score run_id=%s case=%d discarded_proposal_ids=%s",
                context.run_id,
                state.current_case,
                sorted(overlap_rejected_ids),
            )
        accepted: list[CandidateDecision] = []
        searched: list[CandidateDecision] = []
        reconstruction_decisions: list[CandidateDecision] = []
        rebuilt_ids: set[str] = set()
        retained_ids: set[str] = set()
        for audit_item_id, group in decisions_by_audit.items():
            group_accepted = [decision for decision in group if decision.proposal_id in selected_accepted_ids]
            if group_accepted:
                accepted.extend(group_accepted)
                continue
            if any(decision.action == "accept" for decision in group):
                continue
            group_searched = [decision for decision in group if decision.action == "web_search"]
            if group_searched:
                searched.extend(group_searched)
                retained_ids.update(decision.proposal_id for decision in group_searched)
                continue
            reconstruction_decisions.extend(group)
            group_ids = {
                proposal.id for proposal in case.proposals if proposal.audit_item_id == audit_item_id
            }
            rebuilt_ids.update(group_ids)
            retained_ids.update(group_ids)

        rejected = [
            decision
            for decision in batch.decisions
            if decision.action == "reject" or decision.proposal_id in overlap_rejected_ids
        ]
        accepted_ids = {decision.proposal_id for decision in accepted}
        rejected_ids = {decision.proposal_id for decision in rejected}

        overlays = list(state.overlays)
        for decision in accepted:
            proposal = proposals_by_id[decision.proposal_id]
            overlays.append(
                EvidenceOverlay(
                    proposal_id=proposal.id,
                    evidence_index=proposal.evidence_index,
                    chunk_id=proposal.chunk_id,
                    original_expression=proposal.original_expression,
                    resolved_expression=proposal.proposed_expression,
                    target_spans=self._proposal_target_spans(context.evidence, case, proposal),
                    status="auto_resolved",
                    confidence=decision.confidence,
                    source_urls=self._source_urls(case, proposal.id),
                )
            )

        remaining = [
            proposal
            for proposal in case.proposals
            if proposal.id in retained_ids
        ]
        findings = [finding for finding in case.findings if finding.proposal_id in retained_ids]
        attempted_queries = list(case.attempted_queries)
        action_errors: list[str] = []
        reconstruction_completion: LlmGenerateResult | None = None

        search_tasks = [
            self._search_candidate(context, proposals_by_id[decision.proposal_id])
            for decision in searched
        ]
        reconstruction_task = (
            self._reconstruct_selected_candidates(
                context,
                case,
                reconstruction_decisions,
                case_index=state.current_case,
                reconstruct=reconstruct,
            )
            if reconstruction_decisions
            else None
        )
        combined_tasks: list[Awaitable[object]] = [*search_tasks]
        if reconstruction_task is not None:
            combined_tasks.append(reconstruction_task)
        results = await asyncio.gather(*combined_tasks, return_exceptions=True)

        for decision, result in zip(searched, results[: len(search_tasks)], strict=True):
            proposal = proposals_by_id[decision.proposal_id]
            normalized_query = " ".join(proposal.search_query.split())
            attempted_queries.append(normalized_query)
            if isinstance(result, BaseException):
                action_errors.append(type(result).__name__)
                logger.warning(
                    "Evidence 裁决 Agent Candidate Web Search 失败 run_id=%s case=%d proposal=%s error=%s",
                    context.run_id,
                    state.current_case,
                    proposal.id,
                    json.dumps(self._error_for_log(result), ensure_ascii=False, separators=(",", ":")),
                )
                findings.append(
                    GroundedResearchFinding(
                        proposal_id=proposal.id,
                        query=normalized_query,
                        summary=f"联网搜索失败，未获得可用证据（{type(result).__name__}）。",
                    )
                )
            else:
                findings.append(cast(GroundedResearchFinding, result))

        if reconstruction_task is not None:
            reconstruction_result = results[-1]
            if isinstance(reconstruction_result, BaseException):
                action_errors.append(type(reconstruction_result).__name__)
                logger.warning(
                    "Evidence 裁决 Agent Candidate 重建失败 run_id=%s case=%d proposals=%s error=%s",
                    context.run_id,
                    state.current_case,
                    sorted(rebuilt_ids),
                    json.dumps(
                        self._error_for_log(reconstruction_result),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
            else:
                reconstruction_completion, reconstructed = cast(
                    tuple[LlmGenerateResult, list[AdjudicationProposal]],
                    reconstruction_result,
                )
                remaining = [proposal for proposal in remaining if proposal.id not in rebuilt_ids]
                findings = [finding for finding in findings if finding.proposal_id not in rebuilt_ids]
                existing_ids = {proposal.id for proposal in remaining}
                for proposal in reconstructed:
                    if proposal.id in existing_ids or len(remaining) >= 10:
                        continue
                    remaining.append(proposal)
                    existing_ids.add(proposal.id)

        updated_case = case.model_copy(
            update={
                "proposals": remaining,
                "findings": findings,
                "attempted_queries": list(dict.fromkeys(attempted_queries)),
                "search_count": case.search_count + len(searched),
                "pending_decisions": None,
                "decision_history": [*case.decision_history, batch],
                "accepted_proposal_ids": list(dict.fromkeys([*case.accepted_proposal_ids, *accepted_ids])),
                "rejected_proposal_ids": list(dict.fromkeys([*case.rejected_proposal_ids, *rejected_ids])),
                "status": "researching" if remaining else "resolved" if accepted_ids or case.accepted_proposal_ids else "rejected",
                "error": action_errors[-1] if action_errors else None,
            }
        )
        updated_state = state.model_copy(update={"overlays": overlays})
        updated_state = (
            self._advance_case(updated_state, updated_case)
            if not remaining
            else self._replace_case(updated_state, updated_case)
        )
        return AdjudicationTransition(
            updated_state,
            "candidate_actions_executed",
            operation="candidate_actions",
            completion=reconstruction_completion,
            warning_error_type="CandidateActionError" if action_errors else None,
            terminal=not remaining,
        )

    async def _search_candidate(
        self,
        context: AdjudicationCaseContext,
        proposal: AdjudicationProposal,
    ) -> GroundedResearchFinding:
        if self._grounded_search_client is None:
            raise RuntimeError("grounded search client unavailable")
        query = " ".join(proposal.search_query.split())

        finding = await self._grounded_search_client.search(proposal.id, query)
        logger.info(
            "Evidence Agent Web Search Output run_id=%s recording_id=%s chunk_id=%s "
            "evidence_index=%d proposal=%s query=%s summary_preview=%s",
            context.run_id,
            context.evidence.recording.id,
            context.evidence.chunk.id,
            context.evidence.index,
            proposal.id,
            query,
            " ".join(finding.summary.split())[:50],
        )
        return finding

    async def _reconstruct_selected_candidates(
        self,
        context: AdjudicationCaseContext,
        case: EvidenceAdjudicationCaseState,
        decisions: list[CandidateDecision],
        *,
        case_index: int,
        reconstruct: StructuredCompleter,
    ) -> tuple[LlmGenerateResult, list[AdjudicationProposal]]:
        if case.expression_audit is None:
            raise RuntimeError("Candidate reconstruction requires expression audit")
        proposals_by_id = {proposal.id: proposal for proposal in case.proposals}
        audit_ids = {proposals_by_id[decision.proposal_id].audit_item_id for decision in decisions}
        selected_audit = ExpressionAudit(items=[item for item in case.expression_audit.items if item.id in audit_ids])
        focus = json.dumps(
            [
                {
                    "audit_item_id": proposals_by_id[decision.proposal_id].audit_item_id,
                    "rejected_proposal_id": decision.proposal_id,
                    "feedback": decision.reconstruct_focus or decision.reason,
                }
                for decision in decisions
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        prompt, values = evidence_review_prompt(
            context.query,
            context.plan,
            context.evidence,
            reference_evidence=context.reference_evidence,
            expression_audit=selected_audit,
            findings=json.dumps([item.model_dump(mode="json") for item in case.findings], ensure_ascii=False),
            focus=focus,
        )
        completion = await reconstruct(
            prompt.invoke(values).to_messages(),
            AdjudicationReview.model_json_schema(),
        )
        self._log_structured_completion(context, case_index, case.iteration, "candidate_reconstruct", completion)
        review, rejected_proposals = self._parse_review(completion.text)
        if rejected_proposals:
            logger.warning(
                "Evidence 裁决 Agent 过滤重建候选 run_id=%s rejected=%s",
                context.run_id,
                json.dumps(rejected_proposals, ensure_ascii=False, separators=(",", ":")),
            )
        proposals = [
            proposal
            for proposal in self._valid_proposals(case, context, review.proposals)
            if proposal.audit_item_id in audit_ids
        ]

        return completion, proposals

    def _validate_candidate_decisions(
        self,
        state: AdjudicationAgentState,
        case: EvidenceAdjudicationCaseState,
        batch: CandidateDecisionBatch,
        context: AdjudicationCaseContext,
    ) -> None:
        proposal_ids = {proposal.id for proposal in case.proposals}
        decision_ids = [decision.proposal_id for decision in batch.decisions]
        if len(decision_ids) != len(set(decision_ids)) or set(decision_ids) != proposal_ids:
            raise ValueError("candidate decisions must cover every active proposal exactly once")
        proposals_by_id = {proposal.id: proposal for proposal in case.proposals}
        decisions_by_audit: dict[str, list[CandidateDecision]] = {}
        for decision in batch.decisions:
            audit_item_id = proposals_by_id[decision.proposal_id].audit_item_id
            decisions_by_audit.setdefault(audit_item_id, []).append(decision)
        search_decisions = [
            decision
            for group in decisions_by_audit.values()
            if not any(item.action == "accept" for item in group)
            for decision in group
            if decision.action == "web_search"
        ]
        if search_decisions and (not state.web_search_enabled or self._grounded_search_client is None):
            raise ValueError("web_search action is unavailable")
        if len(search_decisions) > max(0, self._max_searches - case.search_count):
            raise ValueError("candidate decisions exceed the remaining web search budget")

        selected_accepted, _ = self._select_non_overlapping_accepted(state, case, batch, context)
        for decision in batch.decisions:
            proposal = proposals_by_id[decision.proposal_id]
            if decision in search_decisions:
                query = " ".join(proposal.search_query.split())
                if query in case.attempted_queries:
                    raise ValueError("candidate decision repeats an attempted web search")
        for decision in selected_accepted:
            proposal = proposals_by_id[decision.proposal_id]
            spans = self._proposal_target_spans(context.evidence, case, proposal)
            if not spans:
                raise ValueError("accepted candidate has no valid target spans")

    @staticmethod
    def _select_non_overlapping_accepted(
        state: AdjudicationAgentState,
        case: EvidenceAdjudicationCaseState,
        batch: CandidateDecisionBatch,
        context: AdjudicationCaseContext,
    ) -> tuple[list[CandidateDecision], set[str]]:
        """Keep the highest-scoring accepted candidate when overlays target overlapping spans."""

        proposals_by_id = {proposal.id: proposal for proposal in case.proposals}
        by_audit: dict[str, list[CandidateDecision]] = {}
        for decision in batch.decisions:
            if decision.action != "accept":
                continue
            by_audit.setdefault(proposals_by_id[decision.proposal_id].audit_item_id, []).append(decision)
        winners = [max(group, key=lambda decision: decision.candidate_score) for group in by_audit.values()]
        decision_order = {decision.proposal_id: index for index, decision in enumerate(batch.decisions)}
        winners.sort(key=lambda decision: (-decision.candidate_score, decision_order[decision.proposal_id]))

        occupied: dict[tuple[int, str], list[tuple[int, int]]] = {}
        for overlay in state.overlays:
            occupied.setdefault((overlay.evidence_index, overlay.chunk_id), []).extend(
                (span.start_char, span.end_char) for span in overlay.target_spans
            )
        selected: list[CandidateDecision] = []
        discarded_ids: set[str] = set()
        for decision in winners:
            proposal = proposals_by_id[decision.proposal_id]
            spans = EvidenceAdjudicationAgent._proposal_target_spans(context.evidence, case, proposal)
            if not spans:
                raise ValueError("accepted candidate has no valid target spans")
            key = (proposal.evidence_index, proposal.chunk_id)
            occupied_spans = occupied.setdefault(key, [])
            if any(
                target.start_char < end_char and start_char < target.end_char
                for target in spans
                for start_char, end_char in occupied_spans
            ):
                discarded_ids.add(proposal.id)
                continue
            occupied_spans.extend((target.start_char, target.end_char) for target in spans)
            selected.append(decision)
        return selected, discarded_ids

    @staticmethod
    def _normalize_candidate_decisions(
        case: EvidenceAdjudicationCaseState,
        batch: CandidateDecisionBatch,
        context: AdjudicationCaseContext,
    ) -> tuple[CandidateDecisionBatch, list[str]]:
        """Discard fabricated/repeated decisions and return any active candidate IDs still uncovered."""

        active_ids = [proposal.id for proposal in case.proposals]
        active_id_set = set(active_ids)
        if len(active_ids) != len(active_id_set):
            raise ValueError("active candidate proposal ids must be unique")

        kept: list[CandidateDecision] = []
        seen: set[str] = set()
        # 模型幻觉输出了相同的 proposalId
        duplicate_ids: list[str] = []
        for decision in batch.decisions:
            proposal_id = decision.proposal_id
            if proposal_id not in active_id_set:
                continue
            if proposal_id in seen:
                duplicate_ids.append(proposal_id)
                continue
            seen.add(proposal_id)
            kept.append(decision)

        missing_ids = sorted(active_id_set - seen)
        if duplicate_ids:
            logger.warning(
                "Evidence 裁决 Agent Candidate 决策后处理 run_id=%s evidence_index=%d discarded_duplicate_proposal_ids=%s missing_proposal_ids=%s",
                context.run_id,
                context.evidence.index,
                duplicate_ids,
                missing_ids,
            )
        return CandidateDecisionBatch(decisions=kept), missing_ids

    @staticmethod
    def _required_setup_phase(case: EvidenceAdjudicationCaseState) -> Literal["audit", "initial_reconstruct"] | None:
        """Return the next fixed phase without creating a synthetic tool call."""

        if case.expression_audit is None:
            return "audit"
        if not case.initial_reconstruction_completed:
            return "initial_reconstruct"
        return None

    @staticmethod
    def _finish_case_without_confirmation(
        state: AdjudicationAgentState,
        case: EvidenceAdjudicationCaseState,
        *,
        reason: str,
    ) -> AdjudicationAgentState:
        status: Literal["resolved", "rejected"] = "resolved" if case.accepted_proposal_ids else "rejected"
        completed = case.model_copy(
            update={
                "proposals": [],
                "pending_decisions": None,
                "pending_setup_phase": None,
                "status": status,
                "error": reason,
            }
        )
        return EvidenceAdjudicationAgent._advance_case(state, completed)

    @staticmethod
    def _current_case(state: AdjudicationAgentState) -> EvidenceAdjudicationCaseState | None:
        if state.status == "completed" or state.current_case >= len(state.cases):
            return None
        return state.cases[state.current_case]

    @staticmethod
    def _replace_case(state: AdjudicationAgentState, case: EvidenceAdjudicationCaseState) -> AdjudicationAgentState:
        cases = list(state.cases)
        cases[state.current_case] = case
        return state.model_copy(update={"cases": cases})

    @staticmethod
    def _advance_case(state: AdjudicationAgentState, case: EvidenceAdjudicationCaseState) -> AdjudicationAgentState:
        cases = list(state.cases)
        cases[state.current_case] = case
        next_case = state.current_case + 1
        return state.model_copy(
            update={
                "cases": cases,
                "current_case": next_case,
                "status": "completed" if next_case >= len(cases) else "running",
            }
        )

    @staticmethod
    def _valid_proposals(
        case: EvidenceAdjudicationCaseState,
        context: AdjudicationCaseContext,
        proposals: list[AdjudicationProposal],
    ) -> list[AdjudicationProposal]:
        reference_indexes = {item.index for item in context.reference_evidence}
        audits_by_id: dict[str, ExpressionAuditItem] = (
            {item.id: item for item in case.expression_audit.items} if case.expression_audit is not None else {}
        )
        valid: list[AdjudicationProposal] = []
        counts_by_audit: dict[str, int] = {}
        for item in proposals:
            audit_item = audits_by_id.get(item.audit_item_id)
            if (
                audit_item is None
                or item.original_expression != audit_item.expression
                or item.evidence_index != case.evidence_index
                or item.chunk_id != str(case.chunk_id)
                or not EvidenceAdjudicationAgent._target_spans(context.evidence.chunk.text, audit_item)
                or counts_by_audit.get(item.audit_item_id, 0) >= 2
                or len(valid) >= 40
            ):
                continue
            valid.append(
                item.model_copy(
                    update={
                        "supporting_evidence_index": (
                            item.supporting_evidence_index
                            if item.supporting_evidence_index in reference_indexes
                            else None
                        )
                    }
                )
            )
            counts_by_audit[item.audit_item_id] = counts_by_audit.get(item.audit_item_id, 0) + 1
        return valid

    @classmethod
    def _parse_review(cls, text: str) -> tuple[AdjudicationReview, list[dict[str, object]]]:
        """Validate proposals independently so one invalid candidate does not discard its siblings."""

        raw: object = json.loads(text)
        if not isinstance(raw, Mapping):
            raise ValueError("Adjudication review must be a JSON object")
        payload = {str(key): value for key, value in cast(Mapping[object, object], raw).items()}
        raw_proposals = payload.get("proposals")
        if not isinstance(raw_proposals, list):
            raise ValueError("Adjudication review proposals must be a list")
        proposals: list[AdjudicationProposal] = []
        rejected: list[dict[str, object]] = []
        for index, item in enumerate(cast(list[object], raw_proposals)):
            try:
                proposals.append(AdjudicationProposal.model_validate(item))
            except Exception as error:
                rejected.append(
                    {
                        "index": index,
                        "error": cls._error_for_log(error),
                        "input": cls._value_for_log(item),
                    }
                )
        payload["proposals"] = [item.model_dump(mode="json") for item in proposals]
        return AdjudicationReview.model_validate(payload), rejected

    @staticmethod
    def _valid_audit(context: AdjudicationCaseContext, audit: ExpressionAudit) -> ExpressionAudit:
        reference_indexes = {item.index for item in context.reference_evidence}
        items: list[ExpressionAuditItem] = []
        for item in audit.items:
            if item.expression not in context.evidence.chunk.text:
                continue
            if item.context_quote not in context.evidence.chunk.text or item.expression not in item.context_quote:
                continue
            if not EvidenceAdjudicationAgent._target_spans(context.evidence.chunk.text, item):
                continue
            supporting_index = (
                item.supporting_evidence_index if item.supporting_evidence_index in reference_indexes else None
            )
            items.append(
                item.model_copy(
                    update={
                        "supporting_evidence_index": supporting_index,
                    }
                )
            )
        covered_starts: dict[str, set[int]] = {}
        for item in items:
            covered_starts.setdefault(item.expression, set()).update(
                span.start_char for span in EvidenceAdjudicationAgent._target_spans(context.evidence.chunk.text, item)
            )
        expanded = list(items)
        for item in items:
            starts = EvidenceAdjudicationAgent._match_starts(context.evidence.chunk.text, item.expression)
            covered = covered_starts[item.expression]
            for start in starts:
                if start in covered:
                    continue
                context_quote = EvidenceAdjudicationAgent._unique_occurrence_context(
                    context.evidence.chunk.text, start, len(item.expression)
                )
                if context_quote is None:
                    logger.warning(
                        "Evidence 裁决 Agent 无法为遗漏表达生成唯一上下文 run_id=%s audit_item_id=%s start_char=%d",
                        context.run_id,
                        item.id,
                        start,
                    )
                    continue
                if len(expanded) >= 20:
                    logger.warning(
                        "Evidence 裁决 Agent 补充审计项达到上限 run_id=%s expression=%s",
                        context.run_id,
                        item.expression,
                    )
                    return ExpressionAudit(items=expanded)
                derived = item.model_copy(
                    update={
                        "id": f"{item.id[:60]}-occ-{start}",
                        "context_quote": context_quote,
                    }
                )
                derived_starts = {
                    span.start_char
                    for span in EvidenceAdjudicationAgent._target_spans(context.evidence.chunk.text, derived)
                }
                covered.update(derived_starts)
                expanded.append(derived)
        return ExpressionAudit(items=expanded)

    @staticmethod
    def _unique_occurrence_context(text: str, start_char: int, expression_length: int) -> str | None:
        """Return a sentence-aligned, unique context quote containing one uncovered occurrence."""

        delimiters = "\n。！？!?"
        left = max(text.rfind(delimiter, 0, start_char) for delimiter in delimiters) + 1
        right_candidates = [text.find(delimiter, start_char + expression_length) for delimiter in delimiters]
        right = min((index + 1 for index in right_candidates if index >= 0), default=len(text))
        while True:
            quote = text[left:right]
            if quote and text.count(quote) == 1:
                return quote
            if right - left >= 1_000 or (left == 0 and right == len(text)):
                return None
            previous = max(text.rfind(delimiter, 0, max(0, left - 1)) for delimiter in delimiters)
            next_candidates = [text.find(delimiter, right) for delimiter in delimiters]
            following = min((index + 1 for index in next_candidates if index >= 0), default=len(text))
            if left > 0 and (right == len(text) or start_char - previous <= following - start_char):
                left = previous + 1
            elif right < len(text):
                right = following
            else:
                return None

    @staticmethod
    def _target_spans(text: str, audit_item: ExpressionAuditItem) -> list[ExpressionTargetSpan]:
        spans: set[tuple[int, int]] = set()
        for quote_start in EvidenceAdjudicationAgent._match_starts(text, audit_item.context_quote):
            for expression_start in EvidenceAdjudicationAgent._match_starts(
                audit_item.context_quote,
                audit_item.expression,
            ):
                start_char = quote_start + expression_start
                spans.add((start_char, start_char + len(audit_item.expression)))
        return [ExpressionTargetSpan(start_char=start, end_char=end) for start, end in sorted(spans)]

    @staticmethod
    def _match_starts(text: str, expression: str) -> list[int]:
        starts: list[int] = []
        offset = 0
        while (found := text.find(expression, offset)) >= 0:
            starts.append(found)
            offset = found + 1
        return starts

    @staticmethod
    def _log_structured_completion(
        context: AdjudicationCaseContext,
        case_index: int,
        iteration: int,
        operation: str,
        completion: LlmGenerateResult,
    ) -> None:
        try:
            value: object = json.loads(completion.text)
            payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except json.JSONDecodeError:
            payload = json.dumps(
                {"invalid_json": True, "raw_text": completion.text},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        logger.info(
            "Evidence Correct Agent Output run_id=%s case=%d iteration=%d operation=%s "
            "provider=%s model=%s payload=%s",
            context.run_id,
            case_index,
            iteration,
            operation,
            completion.provider.value,
            completion.model,
            payload,
        )

    @classmethod
    def _error_for_log(cls, error: BaseException) -> object:
        errors = getattr(error, "errors", None)
        if callable(errors):
            return cls._value_for_log(errors(include_url=False))
        return {"type": type(error).__name__, "message": cls._value_for_log(str(error))}

    @classmethod
    def _value_for_log(cls, value: object) -> object:
        if isinstance(value, Mapping):
            mapping = cast(Mapping[object, object], value)
            return {str(key): cls._value_for_log(item) for key, item in mapping.items()}
        if isinstance(value, list | tuple):
            sequence = cast(list[object] | tuple[object, ...], value)
            return [cls._value_for_log(item) for item in sequence]
        if isinstance(value, BaseException):
            return {
                "type": type(value).__name__,
                "message": cls._value_for_log(str(value)),
            }
        if isinstance(value, str) and len(value) > 200:
            return f"{value[:50]}…<truncated,len={len(value)}>"
        if value is None or isinstance(value, bool | int | float | str):
            return value
        return cls._value_for_log(str(value))

    @staticmethod
    def _source_urls(case: EvidenceAdjudicationCaseState, proposal_id: str) -> list[str]:
        return list(
            dict.fromkeys(
                source.url
                for finding in case.findings
                if finding.proposal_id == proposal_id
                for source in finding.sources
            )
        )

    @staticmethod
    def _proposal_target_spans(
        evidence: Evidence,
        case: EvidenceAdjudicationCaseState,
        proposal: AdjudicationProposal,
    ) -> list[ExpressionTargetSpan]:
        if case.expression_audit is None:
            return []
        audit_item = next((item for item in case.expression_audit.items if item.id == proposal.audit_item_id), None)
        if audit_item is None:
            return []
        return EvidenceAdjudicationAgent._target_spans(evidence.chunk.text, audit_item)

    @staticmethod
    def _proposal_log_payload(
        evidence: Evidence,
        case: EvidenceAdjudicationCaseState,
        proposal: AdjudicationProposal,
    ) -> dict[str, object]:
        audit_item = (
            next(
                (item for item in case.expression_audit.items if item.id == proposal.audit_item_id),
                None,
            )
            if case.expression_audit is not None
            else None
        )
        return {
            "recording_id": str(evidence.recording.id),
            **proposal.model_dump(mode="json"),
            "context_quote": audit_item.context_quote if audit_item is not None else None,
            "target_spans": [
                span.model_dump(mode="json")
                for span in EvidenceAdjudicationAgent._proposal_target_spans(evidence, case, proposal)
            ],
        }

    def _context(self, state: RagGraphState, agent: AdjudicationAgentState) -> AdjudicationCaseContext:
        if agent.current_case >= len(agent.cases):
            raise RuntimeError("Adjudication agent has no current Evidence case")
        case = agent.cases[agent.current_case]
        evidence = next(
            (
                item
                for item in state["answer_evidence"]
                if item.index == case.evidence_index and item.chunk.id == case.chunk_id
            ),
            None,
        )
        if evidence is None:
            raise RuntimeError("Adjudication case evidence is missing from answer_evidence")
        plan = state["answer_plan"]
        relevant = [item for item in plan.items if evidence.index in item.evidence_indexes] if plan is not None else []
        case_plan = AnswerPlan(
            items=relevant or [AnswerPlanItem(statement=state["query"], evidence_indexes=[evidence.index])]
        )
        return AdjudicationCaseContext(
            query=state["query"],
            plan=case_plan,
            evidence=evidence,
            reference_evidence=[
                item
                for item in state["answer_evidence"]
                if not (item.index == evidence.index and item.chunk.id == evidence.chunk.id)
            ][:_MAX_REFERENCE_EVIDENCE],
            run_id=state.get("run_id", "standalone"),
        )

    async def _complete_expression_audit(
        self,
        state: RagGraphState,
        messages: list[BaseMessage],
        schema: dict[str, Any],
    ) -> LlmGenerateResult:
        return await self._complete_structured(
            state,
            messages,
            schema,
            node="adjudication_audit_expressions",
            max_tokens=5_000,
            model=self._audit_model,
            min_request_interval_seconds=self._audit_min_request_interval_seconds,
        )

    async def _complete_candidate_reconstruction(
        self,
        state: RagGraphState,
        messages: list[BaseMessage],
        schema: dict[str, Any],
    ) -> LlmGenerateResult:
        return await self._complete_structured(
            state,
            messages,
            schema,
            node="adjudication_reconstruct_candidates",
            max_tokens=8_000,
        )

    async def _complete_candidate_decisions(
        self,
        state: RagGraphState,
        messages: list[BaseMessage],
        schema: dict[str, Any],
    ) -> LlmGenerateResult:
        return await self._complete_structured(
            state,
            messages,
            schema,
            node="adjudication_candidate_decisions",
            max_tokens=8_000,
        )

    async def _complete_structured(
        self,
        state: RagGraphState,
        messages: list[BaseMessage],
        schema: dict[str, Any],
        *,
        node: str,
        max_tokens: int,
        model: str | None = None,
        min_request_interval_seconds: float | None = None,
    ) -> LlmGenerateResult:
        self._token_budget.before_model(state.get("token_usage", 0), node)
        return await self._model_client.execute(
            build_llm_generate_command(
                self._online_provider,
                self._worker_messages(messages),
                CompletionOptions(
                    max_tokens=max_tokens,
                    model=model,
                    min_request_interval_seconds=min_request_interval_seconds,
                    response_format=ResponseFormat(
                        type=ResponseFormatType.JSON_SCHEMA,
                        json_schema=schema,
                        strict=True,
                    ),
                ),
                context_size=self._context_size,
                stream=False,
            ),
            result_type=LlmGenerateResult,
        )

    @staticmethod
    def _worker_messages(messages: list[BaseMessage]) -> list[ChatMessage]:
        output: list[ChatMessage] = []
        for message in messages:
            role = ChatRole.ASSISTANT if message.type == "ai" else ChatRole.SYSTEM if message.type == "system" else ChatRole.USER
            content = message.content if isinstance(message.content, str) else str(message.content)
            output.append(ChatMessage(role, content))
        return output

    @staticmethod
    def _after_agent_step(state: RagGraphState) -> Literal["execute", "next", "done"]:
        agent = state["adjudication_agent_state"]
        if agent is None or agent.status == "completed" or agent.current_case >= len(agent.cases):
            return "done"
        case = agent.cases[agent.current_case]
        return (
            "execute"
            if case.pending_setup_phase is not None
            or case.pending_decisions is not None
            else "next"
        )

    @staticmethod
    def _after_operation(state: RagGraphState) -> Literal["next", "done"]:
        agent = state["adjudication_agent_state"]
        return "done" if agent is None or agent.status == "completed" else "next"

    def _warning(
        self,
        state: RagGraphState,
        node: str,
        reason: str,
        transition: AdjudicationTransition,
    ) -> None:
        self._event_logger(
            "node_warning",
            state.get("run_id", "standalone"),
            level=logging.WARNING,
            node=node,
            reason=reason,
            operation=transition.operation,
            error_type=transition.warning_error_type,
        )
