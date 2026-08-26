from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, cast
from uuid import UUID

from sqlalchemy import Engine, text

from l1_foundation.model_ref import OnlineModelRef
from l1_foundation.observability import InstrumentedModelClient
from l1_foundation.settings import Settings
from l1_foundation.worker import WorkerClient
from l2_core.rag.adjudication.agent import EvidenceAdjudicationAgent
from l2_core.rag.adjudication.contracts import AdjudicationAgentState
from l2_core.rag.adjudication.web_research import (
    ChromeAiOverviewSearchClient,
    GeminiGroundedSearchClient,
    GroundedSearchClient,
)
from l2_core.rag.contracts import Evidence, EvidenceChunk, EvidenceRecording, RagGraphState
from l2_core.rag.token_budget import RagTokenBudgetMiddleware
from l2_core.rag_adjudication_evaluation.scoring import CorrectionScoringResult, GoldCorrection, score_corrections


@dataclass(frozen=True, slots=True)
class _CaseScoreCounts:
    exact: int
    fuzzy: int
    strict_predictions: int
    relaxed_predictions: int
    gold: int
    predictions: int
    exact_weight: float
    fuzzy_weight: float
    gold_weight: float
    strict_prediction_credit: float
    relaxed_prediction_credit: float
    missed_gold: int
    incorrect_predictions: int
    succeeded: bool


class RagAdjudicationEvaluationRunner:
    def __init__(self, engine: Engine, settings: Settings, worker_client: WorkerClient) -> None:
        self._engine = engine
        self._settings = settings
        self._worker_client = worker_client

    async def execute(self, run_id: UUID) -> None:
        with self._engine.connect() as connection:
            run = dict(
                connection.execute(
                    text("select * from evaluation_runs where id=:id and evaluator_type='rag_adjudication'"),
                    {"id": run_id},
                )
                .mappings()
                .one()
            )
            cases = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select * from rag_adjudication_evaluation_cases
                        where dataset_version_id=:version order by id
                        """
                    ),
                    {"version": run["dataset_version_id"]},
                ).mappings()
            ]
        exact_total = 0
        fuzzy_total = 0
        gold_total = 0
        prediction_total = 0
        strict_prediction_total = 0
        relaxed_prediction_total = 0
        exact_weight_total = 0.0
        fuzzy_weight_total = 0.0
        gold_weight_total = 0.0
        strict_prediction_credit_total = 0.0
        relaxed_prediction_credit_total = 0.0
        missed_gold_total = 0
        incorrect_prediction_total = 0
        completed = 0
        failed = 0
        config = cast(Mapping[str, Any], run.get("config_snapshot") or {})
        fuzzy_threshold = float(config.get("fuzzy_threshold", 90.0))
        expression_fuzzy_threshold = float(config.get("expression_fuzzy_threshold", 80.0))
        importance_weights = cast(Mapping[str, Any], config.get("gold_importance_weights") or {})
        online_model = OnlineModelRef.parse(str(config.get("online_default_model", self._settings.rag_online_default_model)))
        audit_model = OnlineModelRef.parse(str(config.get("audit_model", self._settings.rag_asr_adjudication_audit_model)))
        construct_model = OnlineModelRef.parse(str(config.get("construct_model", self._settings.rag_asr_adjudication_construct_model)))
        decision_model = OnlineModelRef.parse(str(config.get("decision_model", self._settings.rag_asr_adjudication_decision_model)))
        search_client = self._grounded_search_client()
        try:
            for case in cases:
                if self._cancel_requested(run_id):
                    self._finish_cancelled(run_id)
                    return
                counts = await self._execute_case(
                    run_id,
                    case,
                    search_client,
                    fuzzy_threshold,
                    expression_fuzzy_threshold,
                    online_model,
                    audit_model,
                    construct_model,
                    decision_model,
                    importance_weights,
                )
                exact_total += counts.exact
                fuzzy_total += counts.fuzzy
                strict_prediction_total += counts.strict_predictions
                relaxed_prediction_total += counts.relaxed_predictions
                gold_total += counts.gold
                prediction_total += counts.predictions
                exact_weight_total += counts.exact_weight
                fuzzy_weight_total += counts.fuzzy_weight
                gold_weight_total += counts.gold_weight
                strict_prediction_credit_total += counts.strict_prediction_credit
                relaxed_prediction_credit_total += counts.relaxed_prediction_credit
                missed_gold_total += counts.missed_gold
                incorrect_prediction_total += counts.incorrect_predictions
                completed += int(counts.succeeded)
                failed += int(not counts.succeeded)
                with self._engine.begin() as connection:
                    connection.execute(
                        text(
                            """
                            update evaluation_runs set completed_case_count=:completed,
                                failed_case_count=:failed, updated_at=now()
                            where id=:id
                            """
                        ),
                        {"id": run_id, "completed": completed, "failed": failed},
                    )
        finally:
            if search_client is not None:
                await search_client.close()
        with self._engine.begin() as connection:
            for metric in build_metric_rows(
                exact_total,
                fuzzy_total,
                gold_total,
                prediction_total,
                strict_matched_predictions=strict_prediction_total,
                relaxed_matched_predictions=relaxed_prediction_total,
                exact_weight=exact_weight_total,
                fuzzy_weight=fuzzy_weight_total,
                gold_weight=gold_weight_total,
                strict_prediction_credit=strict_prediction_credit_total,
                relaxed_prediction_credit=relaxed_prediction_credit_total,
                diagnostic_missed_gold=missed_gold_total,
                diagnostic_incorrect_predictions=incorrect_prediction_total,
            ):
                connection.execute(
                    text(
                        """
                        insert into rag_adjudication_evaluation_metric_values (
                            evaluation_run_id, metric_name, metric_version, value,
                            passed_count, sample_count, details
                        ) values (
                            :id, :name, '5', :value, :passed, :total, cast(:details as jsonb)
                        )
                        on conflict (evaluation_run_id, metric_name, metric_version)
                        do update set value=excluded.value, passed_count=excluded.passed_count,
                                      sample_count=excluded.sample_count, details=excluded.details
                        """
                    ),
                    {"id": run_id, **metric},
                )
            connection.execute(
                text(
                    """
                    update evaluation_runs set status=:status, finished_at=now(), updated_at=now(),
                        error_message=case when :status='failed' then 'All adjudication cases failed' else null end
                    where id=:id
                    """
                ),
                {"id": run_id, "status": "failed" if failed and not completed else "succeeded"},
            )

    async def _execute_case(
        self,
        run_id: UUID,
        case: Mapping[str, Any],
        search_client: GroundedSearchClient | None,
        fuzzy_threshold: float,
        expression_fuzzy_threshold: float,
        online_model: OnlineModelRef,
        audit_model: OnlineModelRef,
        construct_model: OnlineModelRef,
        decision_model: OnlineModelRef,
        importance_weights: Mapping[str, Any],
    ) -> _CaseScoreCounts:
        case_id = cast(UUID, case["id"])
        evidence_rows, gold_rows = self._load_case(case_id)
        result_id = self._start_case(run_id, case_id)
        started = perf_counter()
        trace_events: list[dict[str, Any]] = []

        def append_trace_event(event: dict[str, Any]) -> None:
            trace_events.append({"sequence": len(trace_events) + 1, **event})

        try:
            evidence = [self._evidence(row, index + 1) for index, row in enumerate(evidence_rows)]
            target_count = sum(row["role"] == "target" for row in evidence_rows)
            agent = EvidenceAdjudicationAgent(
                model_client=InstrumentedModelClient(self._worker_client),
                online_model=online_model,
                context_size=self._settings.rag_context_size,
                token_budget=RagTokenBudgetMiddleware(self._settings.rag_run_max_total_tokens),
                grounded_search_client=search_client,
                web_search_enabled=self._settings.rag_asr_adjudication_web_search_enabled,
                auto_resolve_confidence=self._settings.rag_asr_adjudication_auto_resolve_confidence,
                max_cases=target_count,
                max_iterations=4,
                max_searches=3,
                audit_prompt_variant=self._settings.rag_asr_adjudication_audit_prompt_variant,
                audit_model=audit_model,
                construct_model=construct_model,
                decision_model=decision_model,
                audit_min_request_interval_seconds=self._settings.rag_asr_adjudication_audit_min_request_interval_seconds,
                node_started=_node_started,
                node_completed=_node_completed,
                event_logger=lambda *_args, **_kwargs: None,
                structured_completion_logger=append_trace_event,
            )
            state = self._state(str(result_id), str(case["query"]), evidence)
            output = await agent.start(state)
            agent_state = output.get("adjudication_agent_state")
            if agent_state is None:
                raise RuntimeError("Adjudication agent returned no state")
            evidence_by_id = {row["id"]: row for row in evidence_rows}
            gold = [
                GoldCorrection(
                    id=row["id"],
                    evidence_index=int(row["evidence_index"]),
                    chunk_id=str(row["runtime_chunk_id"]),
                    source_text=str(evidence_by_id[row["frozen_evidence_id"]]["text"]),
                    start_char=int(row["start_char"]),
                    end_char=int(row["end_char"]),
                    accepted_expressions=tuple(row["accepted_expressions"]),
                    importance=cast(Literal["important", "minor"], row["importance"]),
                    weight=float(importance_weights.get(str(row["importance"]), 1.0 if row["importance"] == "important" else 0.5)),
                )
                for row in gold_rows
            ]
            source_texts = {(index + 1, str(row["source_chunk_id"] or row["id"])): str(row["text"]) for index, row in enumerate(evidence_rows)}
            scores = score_corrections(
                gold,
                agent_state.overlays,
                fuzzy_threshold=fuzzy_threshold,
                expression_fuzzy_threshold=expression_fuzzy_threshold,
                source_texts=source_texts,
            )
            self._finish_case(
                result_id,
                round((perf_counter() - started) * 1000),
                agent_state,
                output.get("token_usage", 0),
                scores,
                trace_events,
            )
            return _CaseScoreCounts(
                exact=scores.exact_count,
                fuzzy=scores.fuzzy_count,
                strict_predictions=scores.strict_prediction_count,
                relaxed_predictions=scores.relaxed_prediction_count,
                gold=len(scores.corrections),
                predictions=len(scores.predictions),
                exact_weight=scores.exact_weight,
                fuzzy_weight=scores.fuzzy_weight,
                gold_weight=scores.gold_weight,
                strict_prediction_credit=scores.strict_prediction_credit,
                relaxed_prediction_credit=scores.relaxed_prediction_credit,
                missed_gold=scores.missed_gold_count,
                incorrect_predictions=scores.incorrect_prediction_count,
                succeeded=True,
            )
        except Exception as error:
            self._fail_case(
                result_id,
                round((perf_counter() - started) * 1000),
                error,
                [cast(UUID, row["id"]) for row in gold_rows],
                trace_events,
            )
            return _CaseScoreCounts(
                exact=0,
                fuzzy=0,
                strict_predictions=0,
                relaxed_predictions=0,
                gold=len(gold_rows),
                predictions=0,
                exact_weight=0.0,
                fuzzy_weight=0.0,
                gold_weight=sum(float(importance_weights.get(str(row["importance"]), 1.0 if row["importance"] == "important" else 0.5)) for row in gold_rows),
                strict_prediction_credit=0.0,
                relaxed_prediction_credit=0.0,
                missed_gold=len(gold_rows),
                incorrect_predictions=0,
                succeeded=False,
            )

    def _load_case(self, case_id: UUID) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with self._engine.connect() as connection:
            evidence = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select * from rag_adjudication_evaluation_evidence
                        where evaluation_case_id=:id
                        order by case when role='target' then 0 else 1 end, position, id
                        """
                    ),
                    {"id": case_id},
                ).mappings()
            ]
            index_by_id = {row["id"]: index + 1 for index, row in enumerate(evidence)}
            gold = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select corrections.*, evidence.source_chunk_id,
                               evidence.id as frozen_evidence_id
                        from rag_adjudication_evaluation_corrections corrections
                        join rag_adjudication_evaluation_evidence evidence
                          on evidence.id=corrections.target_evidence_id
                        where evidence.evaluation_case_id=:id order by corrections.id
                        """
                    ),
                    {"id": case_id},
                ).mappings()
            ]
            for row in gold:
                row["evidence_index"] = index_by_id[row["frozen_evidence_id"]]
                row["runtime_chunk_id"] = row["source_chunk_id"] or row["frozen_evidence_id"]
            return evidence, gold

    @staticmethod
    def _evidence(row: Mapping[str, Any], index: int) -> Evidence:
        chunk_id = row["source_chunk_id"] or row["id"]
        return Evidence(
            index=index,
            recording=EvidenceRecording(
                id=row["source_recording_id"],
                title=str(row["recording_title"]),
                file_name=str(row["recording_file_name"]),
            ),
            chunk=EvidenceChunk(
                id=chunk_id,
                text=str(row["text"]),
                start_ms=int(row["start_ms"]),
                end_ms=int(row["end_ms"]),
            ),
            score=1.0,
            match_type="hybrid",
            url=f"/recordings/{row['source_recording_id']}?t={int(row['start_ms'])}",
        )

    @staticmethod
    def _state(run_id: str, query: str, evidence: list[Evidence]) -> RagGraphState:
        return cast(
            RagGraphState,
            {
                "run_id": run_id,
                "execution_mode": "answer",
                "query": query,
                "history": [],
                "limit": len(evidence),
                "scope_recording_ids": [],
                "route": None,
                "route_error": None,
                "filters": None,
                "content_query": query,
                "retrieval_expanded_query": None,
                "retrieval_lexical_queries": [],
                "retrieval_protected_lexical_queries": [],
                "retrieval_attempt": 0,
                "retrieval_candidates": [],
                "protected_chunk_ids": [],
                "rerank_input_tokens": 0,
                "rerank_skipped_candidates": 0,
                "evidence": evidence,
                "answer_evidence": evidence,
                "message": None,
                "grade": None,
                "planning_required": False,
                "answer_plan": None,
                "query_correction_risk": True,
                "adjudication_agent_state": None,
                "adjudication_user_decision": None,
                "token_usage": 0,
                "strategy_result": None,
            },
        )

    def _start_case(self, run_id: UUID, case_id: UUID) -> UUID:
        with self._engine.begin() as connection:
            return cast(
                UUID,
                connection.execute(
                    text(
                        """
                        insert into rag_adjudication_evaluation_case_results (
                            evaluation_run_id, evaluation_case_id, status
                        ) values (:run_id, :case_id, 'running')
                        on conflict (evaluation_run_id, evaluation_case_id)
                        do update set status='running', agent_state=null, overlays='[]'::jsonb,
                                      pending_confirmation=null, trace_events='[]'::jsonb,
                                      error_type=null, error_message=null,
                                      updated_at=now()
                        returning id
                        """
                    ),
                    {"run_id": run_id, "case_id": case_id},
                ).scalar_one(),
            )

    def _finish_case(
        self,
        result_id: UUID,
        latency_ms: int,
        state: AdjudicationAgentState,
        token_usage: int,
        scores: CorrectionScoringResult,
        trace_events: list[dict[str, Any]],
    ) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    update rag_adjudication_evaluation_case_results
                    set status='succeeded', latency_ms=:latency, token_usage=:tokens,
                        agent_state=cast(:state as jsonb), overlays=cast(:overlays as jsonb),
                        pending_confirmation=cast(:confirmation as jsonb),
                        trace_events=cast(:trace_events as jsonb), updated_at=now()
                    where id=:id
                    """
                ),
                {
                    "id": result_id,
                    "latency": latency_ms,
                    "tokens": token_usage,
                    "state": _json(state.model_dump(mode="json")),
                    "overlays": _json([item.model_dump(mode="json") for item in state.overlays]),
                    "confirmation": _json(state.pending_confirmation.model_dump(mode="json")) if state.pending_confirmation else None,
                    "trace_events": _json(trace_events),
                },
            )
            connection.execute(
                text("delete from rag_adjudication_evaluation_correction_results where case_result_id=:id"),
                {"id": result_id},
            )
            connection.execute(
                text("delete from rag_adjudication_evaluation_prediction_results where case_result_id=:id"),
                {"id": result_id},
            )
            for score in scores.corrections:
                connection.execute(
                    text(
                        """
                        insert into rag_adjudication_evaluation_correction_results (
                            case_result_id, gold_correction_id, matched_proposal_id,
                            passed, actual_expression, details
                        ) values (:result, :gold, :proposal, :passed, :actual, cast(:details as jsonb))
                        """
                    ),
                    {
                        "result": result_id,
                        "gold": score.gold_id,
                        "proposal": score.matched_proposal_id,
                        "passed": score.passed,
                        "actual": score.actual_expression,
                        "details": _json(
                            {
                                "match_kind": score.match_kind,
                                "similarity": score.similarity,
                                "match_basis": score.match_basis,
                                "matched_accepted_expression": score.matched_accepted_expression,
                                "actual_local_text": score.actual_local_text,
                                "expected_local_text": score.expected_local_text,
                                "gold_weight": score.gold_weight,
                            }
                        ),
                    },
                )
            for score in scores.predictions:
                prediction = score.prediction
                connection.execute(
                    text(
                        """
                        insert into rag_adjudication_evaluation_prediction_results (
                            case_result_id, matched_gold_correction_id, proposal_id,
                            evidence_index, chunk_id, start_char, end_char,
                            original_expression, resolved_expression, match_kind, similarity, details
                        ) values (
                            :result, :gold, :proposal, :evidence_index, :chunk_id,
                            :start_char, :end_char, :original, :resolved, :match_kind, :similarity,
                            cast(:details as jsonb)
                        )
                        """
                    ),
                    {
                        "result": result_id,
                        "gold": score.matched_gold_ids[0] if score.matched_gold_ids else None,
                        "proposal": prediction.proposal_id,
                        "evidence_index": prediction.evidence_index,
                        "chunk_id": prediction.chunk_id,
                        "start_char": prediction.start_char,
                        "end_char": prediction.end_char,
                        "original": prediction.original_expression,
                        "resolved": prediction.resolved_expression,
                        "match_kind": score.match_kind,
                        "similarity": score.similarity,
                        "details": _json(
                            {
                                "matched_gold_correction_ids": score.matched_gold_ids,
                                "strict_credit": score.strict_credit,
                                "relaxed_credit": score.relaxed_credit,
                            }
                        ),
                    },
                )

    def _fail_case(
        self,
        result_id: UUID,
        latency_ms: int,
        error: Exception,
        gold_ids: list[UUID],
        trace_events: list[dict[str, Any]],
    ) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    update rag_adjudication_evaluation_case_results
                    set status='failed', latency_ms=:latency, error_type=:type,
                        error_message=:message, trace_events=cast(:trace_events as jsonb),
                        updated_at=now() where id=:id
                    """
                ),
                {
                    "id": result_id,
                    "latency": latency_ms,
                    "type": type(error).__name__,
                    "message": str(error)[-2000:],
                    "trace_events": _json(trace_events),
                },
            )
            connection.execute(
                text("delete from rag_adjudication_evaluation_correction_results where case_result_id=:id"),
                {"id": result_id},
            )
            connection.execute(
                text("delete from rag_adjudication_evaluation_prediction_results where case_result_id=:id"),
                {"id": result_id},
            )
            for gold_id in gold_ids:
                connection.execute(
                    text(
                        """
                        insert into rag_adjudication_evaluation_correction_results (
                            case_result_id, gold_correction_id, passed, details
                        ) values (:result, :gold, false, cast(:details as jsonb))
                        """
                    ),
                    {"result": result_id, "gold": gold_id, "details": _json({"error": type(error).__name__})},
                )

    def _cancel_requested(self, run_id: UUID) -> bool:
        with self._engine.connect() as connection:
            return bool(
                connection.execute(
                    text("select cancel_requested from evaluation_runs where id=:id"),
                    {"id": run_id},
                ).scalar_one()
            )

    def _finish_cancelled(self, run_id: UUID) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    update evaluation_runs set status='cancelled', finished_at=now(), updated_at=now()
                    where id=:id
                    """
                ),
                {"id": run_id},
            )

    def _grounded_search_client(self) -> GroundedSearchClient | None:
        settings = self._settings
        if not settings.rag_asr_adjudication_web_search_enabled:
            return None
        if settings.rag_asr_adjudication_search_provider == "chrome_ai_overview":
            return ChromeAiOverviewSearchClient(
                timeout_seconds=settings.rag_asr_adjudication_chrome_aio_timeout_seconds,
                poll_interval_seconds=settings.rag_asr_adjudication_chrome_aio_poll_interval_seconds,
            )
        if not settings.gemini_api_key:
            return None
        return GeminiGroundedSearchClient(
            api_key=settings.gemini_api_key,
            model=settings.rag_asr_adjudication_search_model,
            base_url=settings.gemini_native_base_url,
            timeout_seconds=min(settings.gemini_timeout_seconds, 60.0),
        )


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def build_metric_rows(
    exact: int,
    fuzzy: int,
    gold: int,
    predictions: int,
    *,
    strict_matched_predictions: int | None = None,
    relaxed_matched_predictions: int | None = None,
    exact_weight: float | None = None,
    fuzzy_weight: float | None = None,
    gold_weight: float | None = None,
    strict_prediction_credit: float | None = None,
    relaxed_prediction_credit: float | None = None,
    diagnostic_missed_gold: int | None = None,
    diagnostic_incorrect_predictions: int | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    weighted_exact = float(exact) if exact_weight is None else exact_weight
    weighted_fuzzy = float(fuzzy) if fuzzy_weight is None else fuzzy_weight
    weighted_gold = float(gold) if gold_weight is None else gold_weight
    scopes = (
        (
            "strict",
            exact,
            exact if strict_matched_predictions is None else strict_matched_predictions,
            weighted_exact,
            float(exact if strict_matched_predictions is None else strict_matched_predictions)
            if strict_prediction_credit is None
            else strict_prediction_credit,
        ),
        (
            "relaxed",
            exact + fuzzy,
            exact + fuzzy if relaxed_matched_predictions is None else relaxed_matched_predictions,
            weighted_exact + weighted_fuzzy,
            float(exact + fuzzy if relaxed_matched_predictions is None else relaxed_matched_predictions)
            if relaxed_prediction_credit is None
            else relaxed_prediction_credit,
        ),
    )
    for scope, matched_gold, matched_predictions, matched_gold_weight, prediction_credit in scopes:
        false_positive = predictions - matched_predictions
        false_negative = gold - matched_gold
        weighted_false_positive = predictions - prediction_credit
        weighted_false_negative = weighted_gold - matched_gold_weight
        precision = prediction_credit / predictions if predictions else 0.0
        recall = matched_gold_weight / weighted_gold if weighted_gold else 0.0
        f1_denominator = prediction_credit * weighted_gold + matched_gold_weight * predictions
        f1 = 2 * prediction_credit * matched_gold_weight / f1_denominator if f1_denominator else 0.0
        details = _json(
            {
                "scope": scope,
                "matched_gold_count": matched_gold,
                "matched_prediction_count": matched_predictions,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "exact_count": exact,
                "fuzzy_count": fuzzy,
                "gold_count": gold,
                "prediction_count": predictions,
                "matched_gold_weight": matched_gold_weight,
                "gold_weight": weighted_gold,
                "prediction_credit": prediction_credit,
                "weighted_false_positive": weighted_false_positive,
                "weighted_false_negative": weighted_false_negative,
                "missed_gold_count": false_negative if diagnostic_missed_gold is None else diagnostic_missed_gold,
                "incorrect_prediction_count": false_positive
                if diagnostic_incorrect_predictions is None
                else diagnostic_incorrect_predictions,
            }
        )
        rows.extend(
            [
                {"name": f"correction_precision_{scope}", "value": precision, "passed": matched_predictions, "total": predictions, "details": details},
                {"name": f"correction_recall_{scope}", "value": recall, "passed": matched_gold, "total": gold, "details": details},
                {"name": f"correction_f1_{scope}", "value": f1, "passed": matched_gold, "total": gold + predictions, "details": details},
            ]
        )
    return rows


def _node_started(state: RagGraphState, node: str) -> float:
    del state, node
    return perf_counter()


def _node_completed(state: RagGraphState, node: str, start: float, **fields: object) -> None:
    del state, node, start, fields
