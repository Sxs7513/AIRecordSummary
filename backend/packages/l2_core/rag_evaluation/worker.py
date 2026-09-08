from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from threading import Event, Thread
from time import perf_counter
from typing import cast
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import Engine, text

from l1_foundation.model_ref import OnlineModelRef
from l1_foundation.observability import InstrumentedModelClient
from l1_foundation.settings import Settings
from l1_foundation.worker import ComputeCommand, ExecutionScope, SyncWorkerClient, WorkerClient, execution_scope
from l2_core.rag.contracts import Evidence, EvidenceGrade, EvidenceSource
from l2_core.rag.graph import RagGraph
from l2_core.rag.hooks import RagNodeCompleted, RagOperationCompleted
from l2_core.rag.retrieval import RagRetriever
from l2_core.rag_adjudication_evaluation.runner import RagAdjudicationEvaluationRunner
from l2_core.rag_evaluation.answer_annotations import AnswerAnnotation, AnswerVerdict, answerability_matches
from l2_core.rag_evaluation.answer_evaluator import AnswerEvaluationResult, AnswerEvaluator
from l2_core.rag_evaluation.answer_metrics import answer_case_metrics
from l2_core.rag_evaluation.contracts import EvidenceAnchor, RankedItem, RetrievalMetrics
from l2_core.rag_evaluation.diagnosis import (
    DIAGNOSTIC_VERSION,
    OperationMatches,
    build_evidence_journeys,
    summarize_journeys,
)
from l2_core.rag_evaluation.evidence_matcher import match_ranked_item
from l2_core.rag_evaluation.metrics import mean_metrics, percentile, retrieval_metrics

logger = logging.getLogger("evaluation")
MAX_SAVED_CANDIDATES = 50
FINAL_METRICS_KEY = "__final__"
RETRIEVAL_METRIC_VERSION = "2"
ANSWER_METRIC_VERSION = "4"


class EvaluationTraceHook:
    """In-memory hook; persistence remains owned by the evaluation worker."""

    def __init__(self) -> None:
        self.nodes: list[RagNodeCompleted] = []
        self.operations: list[RagOperationCompleted] = []

    def on_node_completed(self, event: RagNodeCompleted) -> None:
        self.nodes.append(event)

    def on_operation_completed(self, event: RagOperationCompleted) -> None:
        self.operations.append(event)


class _AsyncSyncWorkerClient:
    """Expose a thread-owned synchronous Compute client to async RAG graph nodes."""

    def __init__(self, client: SyncWorkerClient) -> None:
        self._client = client

    async def __aenter__(self) -> _AsyncSyncWorkerClient:
        return self

    async def __aexit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        return None

    async def execute[InputT: BaseModel, ResultT: BaseModel](
        self,
        command: ComputeCommand[InputT],
        *,
        result_type: type[ResultT],
        on_progress: Callable[[float, str | None], None] | None = None,
    ) -> ResultT:
        return await asyncio.to_thread(
            self._client.execute,
            command,
            result_type=result_type,
            on_progress=on_progress,
        )

    async def execute_streaming[InputT: BaseModel, ResultT: BaseModel](
        self,
        command: ComputeCommand[InputT],
        *,
        result_type: type[ResultT],
        on_progress: Callable[[float, str | None], None] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> ResultT:
        return await asyncio.to_thread(
            self._client.execute_streaming,
            command,
            result_type=result_type,
            on_progress=on_progress,
            on_delta=on_delta,
        )


class RagEvaluationWorker:
    """Claim queued retrieval runs and evaluate production retrieval stages."""

    def __init__(self, engine: Engine, settings: Settings, worker_client: SyncWorkerClient) -> None:
        self._engine = engine
        self._settings = settings
        self._worker_client = worker_client
        self._poll_seconds = settings.asr_lab_worker_poll_seconds
        self._stop_event = Event()

    def run_once(self) -> bool:
        if self._stop_event.is_set():
            return False
        run_id = self._claim()
        if run_id is None:
            return False
        logger.info("RAG 评测：领取任务 run_id=%s", run_id)
        try:
            self._execute(run_id)
        except Exception as error:
            logger.exception("RAG 评测：任务失败 run_id=%s", run_id)
            with self._engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        update evaluation_runs
                        set status = 'failed', error_message = :error,
                            finished_at = now(), updated_at = now()
                        where id = :run_id
                        """
                    ),
                    {"run_id": run_id, "error": str(error)[-2000:]},
                )
        return True

    def run_forever(self) -> None:
        recovered = self.recover_stale_runs()
        if recovered:
            logger.warning("RAG 评测：已恢复 %s 个超时运行任务", recovered)
        logger.info("RAG evaluation worker started")
        while not self._stop_event.is_set():
            if not self.run_once():
                self._stop_event.wait(self._poll_seconds)
        logger.info("RAG evaluation worker stopped")

    def stop(self) -> None:
        self._stop_event.set()

    def _claim(self) -> UUID | None:
        with self._engine.begin() as connection:
            value = connection.execute(
                text(
                    """
                    with candidate as (
                        select id from evaluation_runs
                        where status = 'queued'
                          and evaluator_type in ('rag_retrieval', 'rag_adjudication')
                        order by created_at
                        for update skip locked
                        limit 1
                    )
                    update evaluation_runs runs
                    set status = 'running', started_at = coalesce(started_at, now()),
                        error_message = null, updated_at = now()
                    from candidate
                    where runs.id = candidate.id
                    returning runs.id
                    """
                )
            ).scalar_one_or_none()
        return None if value is None else UUID(str(value))

    def _execute(self, run_id: UUID) -> None:
        heartbeat_stop = Event()
        heartbeat = Thread(
            target=self._heartbeat_until_stopped,
            args=(run_id, heartbeat_stop),
            name=f"rag-evaluation-heartbeat-{run_id}",
            daemon=True,
        )
        heartbeat.start()
        try:
            with execution_scope(ExecutionScope(kind="evaluation", id=run_id)):
                asyncio.run(self._execute_async(run_id))
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=1)

    def recover_stale_runs(self) -> int:
        """Requeue work abandoned by a stopped worker without touching live heartbeats."""
        with self._engine.begin() as connection:
            result = connection.execute(
                text(
                    """
                    update evaluation_runs
                    set status = 'queued', started_at = null, finished_at = null,
                        completed_case_count = 0, failed_case_count = 0,
                        error_message = 'Recovered after evaluation worker heartbeat timeout', updated_at = now()
                    where status = 'running'
                      and evaluator_type in ('rag_retrieval', 'rag_adjudication')
                      and cancel_requested = false
                      and updated_at < now() - (:stale_after_seconds * interval '1 second')
                    """
                ),
                {"stale_after_seconds": self._settings.rag_evaluation_stale_run_seconds},
            )
            return int(result.rowcount)

    def _heartbeat_until_stopped(self, run_id: UUID, stop_event: Event) -> None:
        while not stop_event.is_set():
            try:
                with self._engine.begin() as connection:
                    connection.execute(
                        text("update evaluation_runs set updated_at = now() where id = :run_id and status = 'running'"),
                        {"run_id": run_id},
                    )
            except Exception:
                logger.exception("RAG 评测：更新心跳失败 run_id=%s", run_id)
            stop_event.wait(min(30.0, self._settings.rag_evaluation_stale_run_seconds / 3))

    async def _execute_async(self, run_id: UUID) -> None:
        with self._engine.connect() as connection:
            evaluator_type = str(
                connection.execute(
                    text("select evaluator_type from evaluation_runs where id = :run_id"),
                    {"run_id": run_id},
                ).scalar_one()
            )
        if evaluator_type == "rag_adjudication":
            async with self._model_worker_client(self._settings) as model_worker:
                await RagAdjudicationEvaluationRunner(
                    self._engine,
                    self._settings,
                    cast(WorkerClient, model_worker),
                ).execute(run_id)
            return
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    delete from rag_evaluation_metric_values
                    where evaluation_run_id = :run_id and scope in ('run', 'operation', 'tag')
                    """
                ),
                {"run_id": run_id},
            )
        with self._engine.connect() as connection:
            run = dict(
                connection.execute(
                    text(
                        """
                        select runs.*, specs.corpus_snapshot_id
                        from evaluation_runs runs
                        join rag_evaluation_run_specs specs on specs.evaluation_run_id = runs.id
                        where runs.id = :run_id
                        """
                    ),
                    {"run_id": run_id},
                )
                .mappings()
                .one()
            )
            cases = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select * from rag_evaluation_cases
                        where dataset_version_id = :version_id and split = :split
                        order by id
                        """
                    ),
                    {"version_id": run["dataset_version_id"], "split": run["split"]},
                ).mappings()
            ]
            workspace_recording_ids = [
                cast(UUID, value)
                for value in connection.execute(
                    text("select id from recordings where workspace_id = :workspace_id and status = 'completed' order by id"),
                    {"workspace_id": run["workspace_id"]},
                ).scalars()
            ]
            snapshot_chunks = {
                cast(UUID, row["source_chunk_id"]): cast(UUID, row["id"])
                for row in connection.execute(
                    text(
                        """
                        select id, source_chunk_id from rag_corpus_snapshot_chunks
                        where corpus_snapshot_id = :snapshot_id and source_chunk_id is not null
                        """
                    ),
                    {"snapshot_id": run["corpus_snapshot_id"]},
                ).mappings()
            }

        settings = self._settings_for_run(cast(Mapping[str, object], run["config_snapshot"]))
        retriever = RagRetriever(self._engine, settings, self._worker_client)
        metrics_by_operation: dict[str, list[RetrievalMetrics]] = defaultdict(list)
        case_latencies: list[int] = []
        grade_answerability_results: list[float] = []
        answer_metrics_by_name: dict[str, list[float]] = defaultdict(list)
        completed = failed = 0

        async with self._model_worker_client(settings) as model_worker:
            graph = RagGraph(
                retriever,
                InstrumentedModelClient(cast(WorkerClient, model_worker)),
                online_model=OnlineModelRef.parse(settings.rag_online_default_model),
                context_size=settings.rag_context_size,
                plan_local_input_tokens=settings.rag_plan_local_input_tokens,
                max_total_tokens=settings.rag_run_max_total_tokens,
                route_model_profile=settings.rag_route_model_profile,
                node_model_profile=settings.rag_node_model_profile,
                query_term_expansion_enabled=settings.rag_query_term_expansion_enabled,
            )
            answer_evaluator = AnswerEvaluator(
                InstrumentedModelClient(cast(WorkerClient, model_worker)),
                OnlineModelRef.parse(settings.rag_answer_judge_model),
                context_size=settings.rag_context_size,
                max_output_tokens=settings.rag_answer_judge_max_output_tokens,
            )
            try:
                for case in cases:
                    if self._cancel_requested(run_id):
                        self._finish_cancelled(run_id)
                        return
                    started = perf_counter()
                    case_id = cast(UUID, case["id"])
                    result_id = self._start_case(run_id, case_id, str(case["query"]))
                    try:
                        evidence = self._load_evidence(case_id)
                        annotation = AnswerAnnotation.model_validate(case["answer_annotation"])
                        scope_ids = self._scope_recording_ids(cast(Mapping[str, object], case["scope"]), workspace_recording_ids)
                        hook = EvaluationTraceHook()
                        answer, sources, not_enough_evidence, message, confirmation = await graph.run(
                            query=str(case["query"]),
                            limit=settings.rag_fused_candidate_limit,
                            scope_recording_ids=scope_ids,
                            on_phase=lambda _name, _label, _progress: None,
                            on_delta=lambda _delta: None,
                            run_id=result_id,
                            hook=hook,
                        )
                        if confirmation is not None:
                            raise RuntimeError("RAG answer evaluation must not enter adjudication confirmation")
                        route_node = next((item for item in hook.nodes if item.node == "route"), None)
                        route_error = str(route_node.metadata.get("reason") or "") if route_node is not None else ""
                        grade_node = next((item for item in reversed(hook.nodes) if item.node == "grade_corrected"), None)
                        if grade_node is None:
                            grade_node = next((item for item in reversed(hook.nodes) if item.node == "grade"), None)
                        if grade_node is None:
                            grade_node = next((item for item in reversed(hook.nodes) if item.node == "grade_original"), None)
                        raw_verdict = grade_node.metadata.get("verdict") if grade_node is not None else None
                        actual_verdict = "abstain" if not_enough_evidence else _answer_verdict(raw_verdict or "direct_answer")
                        grade = EvidenceGrade(
                            verdict=actual_verdict,
                            reason=str(grade_node.metadata.get("reason") or "") if grade_node is not None else (message or route_error),
                        )
                        grade_answerability_correct = answerability_matches(annotation.expected_verdict, actual_verdict)
                        operations = hook.operations or [self._empty_terminal_operation(hook, route_error or None)]
                        final_metric: RetrievalMetrics | None = None
                        diagnostic_operations: list[OperationMatches] = []
                        for sequence, operation in enumerate(operations):
                            items = _operation_items(operation)
                            diagnostic_operations.append(
                                OperationMatches(
                                    operation=operation.operation,
                                    status="failed" if operation.status == "failed" else "succeeded",
                                    matches=[match_ranked_item(item, evidence) for item in items],
                                )
                            )
                            metric = self._save_ranked_step(
                                result_id,
                                sequence,
                                operation.operation,
                                items,
                                evidence,
                                snapshot_chunks,
                                round(operation.elapsed_ms),
                                status="failed" if operation.status == "failed" else "succeeded",
                                details={"node": operation.node, "execution_status": operation.status, **operation.details},
                            )
                            metrics_by_operation[operation.operation].append(metric)
                            final_metric = metric
                        if final_metric is not None:
                            metrics_by_operation[FINAL_METRICS_KEY].append(final_metric)
                        evidence_journeys = build_evidence_journeys(evidence, diagnostic_operations)
                        evidence_diagnosis = summarize_journeys(evidence_journeys)
                        self._save_evidence_diagnosis(result_id, case_id, evidence_journeys)
                        self._save_grade_step(
                            result_id,
                            len(operations),
                            grade,
                            grade_answerability_correct,
                            round(grade_node.elapsed_ms) if grade_node is not None else 0,
                            expected_verdict=annotation.expected_verdict,
                        )

                        answer_sequence = len(operations) + 1
                        answer_node = next((item for item in reversed(hook.nodes) if item.node == "answer"), None)
                        self._save_answer_step(
                            result_id,
                            answer_sequence,
                            answer,
                            sources,
                            actual_verdict,
                            round(answer_node.elapsed_ms) if answer_node is not None else 0,
                        )
                        if annotation.expected_verdict == "abstain" and actual_verdict == "abstain":
                            answer_evaluation = AnswerEvaluationResult(
                                answerability_correct=True,
                                key_point_results=[],
                                claim_results=[],
                            )
                            judge_latency = 0
                            judge_skipped = True
                        else:
                            judge_started = perf_counter()
                            try:
                                answer_evaluation = await answer_evaluator.evaluate(
                                    query=str(case["query"]),
                                    annotation=annotation,
                                    actual_verdict=actual_verdict,
                                    generated_answer=answer,
                                    cited_evidence=self._load_cited_evidence(sources),
                                    gold_evidence=[
                                        {
                                            "id": str(item.id),
                                            "recording_id": str(item.recording_id),
                                            "quote": item.quote,
                                            "start_ms": item.start_ms,
                                            "end_ms": item.end_ms,
                                        }
                                        for item in evidence
                                    ],
                                )
                            except Exception as judge_error:
                                judge_latency = _elapsed_ms(judge_started)
                                logger.exception(
                                    "RAG 评测：Answer Judge 失败 run_id=%s case_id=%s",
                                    run_id,
                                    case_id,
                                )
                                self._save_answer_evaluation_failure_step(
                                    result_id,
                                    answer_sequence + 1,
                                    judge_latency,
                                    judge_model=settings.rag_answer_judge_model,
                                    prompt_version=settings.rag_answer_judge_prompt_version,
                                    error=judge_error,
                                )
                                answer_metrics_by_name["answer_evaluation_failure"].append(1.0)
                                latency = _elapsed_ms(started)
                                case_latencies.append(latency)
                                self._finish_case(
                                    result_id,
                                    latency,
                                    succeeded=True,
                                    details={
                                        "route_strategy": route_node.metadata.get("strategy_id") if route_node is not None else None,
                                        "route_error": route_error or None,
                                        "expected_verdict": annotation.expected_verdict,
                                        "actual_verdict": actual_verdict,
                                        "not_enough_evidence": not_enough_evidence,
                                        "grade_answerability_correct": grade_answerability_correct,
                                        "answer_evaluation_status": "failed",
                                        "answer_evaluation_error": str(judge_error)[-2_000:],
                                        "evidence_diagnosis": evidence_diagnosis,
                                    },
                                )
                                grade_answerability_results.append(float(grade_answerability_correct))
                                completed += 1
                                self._update_progress(run_id, completed, failed)
                                continue
                            judge_latency = _elapsed_ms(judge_started)
                            judge_skipped = False
                        case_answer_metrics = answer_case_metrics(annotation, actual_verdict, answer_evaluation)
                        self._save_answer_evaluation_step(
                            result_id,
                            answer_sequence + 1,
                            answer_evaluation,
                            case_answer_metrics,
                            judge_latency,
                            judge_model=settings.rag_answer_judge_model,
                            prompt_version=settings.rag_answer_judge_prompt_version,
                            judge_skipped=judge_skipped,
                        )
                        for metric_name, metric_value in case_answer_metrics.items():
                            answer_metrics_by_name[metric_name].append(metric_value)
                        answer_metrics_by_name["answer_evaluation_failure"].append(0.0)

                        latency = _elapsed_ms(started)
                        case_latencies.append(latency)
                        self._finish_case(
                            result_id,
                            latency,
                            succeeded=True,
                            details={
                                "route_strategy": route_node.metadata.get("strategy_id") if route_node is not None else None,
                                "route_error": route_error or None,
                                "expected_verdict": annotation.expected_verdict,
                                "actual_verdict": actual_verdict,
                                "not_enough_evidence": not_enough_evidence,
                                "grade_verdict": actual_verdict,
                                "grade_answerability_correct": grade_answerability_correct,
                                "answer_evaluation_status": "skipped" if judge_skipped else "succeeded",
                                "evidence_diagnosis": evidence_diagnosis,
                            },
                        )
                        grade_answerability_results.append(float(grade_answerability_correct))
                        completed += 1
                    except Exception as error:
                        logger.exception("RAG 评测：Case 失败 run_id=%s case_id=%s", run_id, case_id)
                        self._finish_case(result_id, _elapsed_ms(started), succeeded=False, error=str(error))
                        failed += 1
                    self._update_progress(run_id, completed, failed)
            finally:
                await asyncio.to_thread(retriever.release)

        self._save_run_metrics(
            run_id,
            metrics_by_operation,
            case_latencies,
            grade_answerability_results,
            answer_metrics_by_name,
        )
        with self._engine.begin() as connection:
            status = "failed" if failed and not completed else "succeeded"
            connection.execute(
                text(
                    """
                    update evaluation_runs
                    set status = :status, completed_case_count = :completed,
                        failed_case_count = :failed, finished_at = now(), updated_at = now(),
                        error_message = case when :status = 'failed' then 'All RAG evaluation cases failed' else null end
                    where id = :run_id
                    """
                ),
                {"run_id": run_id, "status": status, "completed": completed, "failed": failed},
            )

    def _model_worker_client(self, settings: Settings) -> _AsyncSyncWorkerClient:
        del settings
        return _AsyncSyncWorkerClient(self._worker_client)

    @staticmethod
    def _empty_terminal_operation(hook: EvaluationTraceHook, route_error: str | None) -> RagOperationCompleted:
        route_node = next((item for item in reversed(hook.nodes) if item.node == "route"), None)
        return RagOperationCompleted(
            node="route",
            operation="route.unresolved" if route_error else "retrieve.empty",
            output=[],
            elapsed_ms=route_node.elapsed_ms if route_node is not None else 0,
            status="degraded" if route_error else "succeeded",
            details={"route_error": route_error},
        )

    def _save_ranked_step(
        self,
        case_result_id: UUID,
        sequence: int,
        operation: str,
        items: Sequence[RankedItem],
        evidence: Sequence[EvidenceAnchor],
        snapshot_chunks: Mapping[UUID, UUID],
        latency_ms: int,
        *,
        status: str = "succeeded",
        details: Mapping[str, object] | None = None,
    ) -> RetrievalMetrics:
        matches = [match_ranked_item(item, evidence) for item in items]
        metrics = retrieval_metrics(matches, [item.relevance for item in evidence])
        with self._engine.begin() as connection:
            step_id = cast(
                UUID,
                connection.execute(
                    text(
                        """
                        insert into rag_evaluation_step_results (
                            case_result_id, operation, operation_version, sequence, attempt,
                            output_kind, status, latency_ms, output, details
                        ) values (
                            :case_result_id, :operation, '1', :sequence, 0,
                            'ranked_candidates', :status, :latency_ms,
                            cast(:output as jsonb), cast(:details as jsonb)
                        ) returning id
                        """
                    ),
                    {
                        "case_result_id": case_result_id,
                        "operation": operation,
                        "sequence": sequence,
                        "latency_ms": latency_ms,
                        "status": status,
                        "output": _json({"candidate_count": len(items)}),
                        "details": _json(details or {}),
                    },
                ).scalar_one(),
            )
            for rank, (item, match) in enumerate(zip(items[:MAX_SAVED_CANDIDATES], matches, strict=False), start=1):
                matched_evidences: list[dict[str, object]] = [
                    {
                        "evidence_id": covered.evidence_id,
                        "relevance": covered.relevance,
                        "match_kind": covered.kind,
                    }
                    for covered in match.all_matches()
                ]
                ranked_result_details: dict[str, object] = {
                    "text": item.text,
                    "start_ms": item.start_ms,
                    "end_ms": item.end_ms,
                    "matched_evidences": matched_evidences,
                }
                connection.execute(
                    text(
                        """
                        insert into rag_evaluation_ranked_results (
                            step_result_id, rank, corpus_snapshot_chunk_id, recording_id,
                            source_chunk_id, score, vector_score, lexical_score, rrf_score,
                            rerank_score, matched_evidence_id, matched_relevance, match_kind, details
                        ) values (
                            :step_id, :rank, :snapshot_chunk_id, :recording_id,
                            :source_chunk_id, :score, :vector_score, :lexical_score, :rrf_score,
                            :rerank_score, :evidence_id, :relevance, :match_kind, cast(:details as jsonb)
                        )
                        """
                    ),
                    {
                        "step_id": step_id,
                        "rank": rank,
                        "snapshot_chunk_id": snapshot_chunks.get(item.source_chunk_id) if item.source_chunk_id else None,
                        "recording_id": item.recording_id,
                        "source_chunk_id": item.source_chunk_id,
                        "score": item.score,
                        "vector_score": item.vector_score,
                        "lexical_score": item.lexical_score,
                        "rrf_score": item.rrf_score,
                        "rerank_score": item.rerank_score,
                        "evidence_id": match.evidence_id,
                        "relevance": match.relevance,
                        "match_kind": match.kind,
                        "details": _json(ranked_result_details),
                    },
                )
            for name, value in metrics.as_dict().items():
                connection.execute(
                    text(
                        """
                        insert into rag_evaluation_metric_values (
                            evaluation_run_id, evaluation_case_id, step_result_id, scope,
                            operation, metric_name, metric_version, value, sample_count
                        )
                        select results.evaluation_run_id, results.evaluation_case_id, :step_id,
                               'step', :operation, :metric_name, :metric_version, :value, 1
                        from rag_evaluation_case_results results where results.id = :case_result_id
                        """
                    ),
                    {
                        "step_id": step_id,
                        "case_result_id": case_result_id,
                        "operation": operation,
                        "metric_name": name,
                        "metric_version": RETRIEVAL_METRIC_VERSION,
                        "value": value,
                    },
                )
        return metrics

    def _save_run_metrics(
        self,
        run_id: UUID,
        metrics_by_operation: Mapping[str, Sequence[RetrievalMetrics]],
        case_latencies: Sequence[int],
        grade_answerability_results: Sequence[float],
        answer_metrics_by_name: Mapping[str, Sequence[float]],
    ) -> None:
        grade_metric_details: dict[str, object] = {"expected": "answerable_vs_abstain"}
        with self._engine.begin() as connection:
            for operation, values in metrics_by_operation.items():
                if operation == FINAL_METRICS_KEY:
                    continue
                for name, value in mean_metrics(values).items():
                    connection.execute(
                        text(
                            """
                            insert into rag_evaluation_metric_values (
                                evaluation_run_id, scope, operation, metric_name,
                                metric_version, value, sample_count
                            ) values (:run_id, 'operation', :operation, :name, :metric_version, :value, :sample_count)
                            """
                        ),
                        {
                            "run_id": run_id,
                            "operation": operation,
                            "name": name,
                            "metric_version": RETRIEVAL_METRIC_VERSION,
                            "value": value,
                            "sample_count": len(values),
                        },
                    )
            final_operation = "retrieval.final"
            final_values = metrics_by_operation.get(FINAL_METRICS_KEY, ())
            for name, value in mean_metrics(final_values).items():
                connection.execute(
                    text(
                        """
                        insert into rag_evaluation_metric_values (
                            evaluation_run_id, scope, scope_key, operation, metric_name,
                            metric_version, value, sample_count
                        ) values (:run_id, 'run', 'final', :operation, :name, :metric_version, :value, :sample_count)
                        """
                    ),
                    {
                        "run_id": run_id,
                        "operation": final_operation,
                        "name": name,
                        "metric_version": RETRIEVAL_METRIC_VERSION,
                        "value": value,
                        "sample_count": len(final_values),
                    },
                )
            connection.execute(
                text(
                    """
                    insert into rag_evaluation_metric_values (
                        evaluation_run_id, scope, scope_key, operation, metric_name,
                        metric_version, value, sample_count, details
                    ) values (
                        :run_id, 'run', 'final', 'grade.evidence', 'grade_answerability_accuracy',
                        '1', :value, :sample_count, cast(:details as jsonb)
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "value": (sum(grade_answerability_results) / len(grade_answerability_results) if grade_answerability_results else 0.0),
                    "sample_count": len(grade_answerability_results),
                    "details": _json(grade_metric_details),
                },
            )
            for name, value in {
                "latency_p50_ms": percentile(case_latencies, 0.5),
                "latency_p90_ms": percentile(case_latencies, 0.9),
            }.items():
                connection.execute(
                    text(
                        """
                        insert into rag_evaluation_metric_values (
                            evaluation_run_id, scope, scope_key, metric_name,
                            metric_version, value, sample_count
                        ) values (:run_id, 'run', 'final', :name, '1', :value, :sample_count)
                        """
                    ),
                    {"run_id": run_id, "name": name, "value": value, "sample_count": len(case_latencies)},
                )
            for name, values in answer_metrics_by_name.items():
                if not values:
                    continue
                connection.execute(
                    text(
                        """
                        insert into rag_evaluation_metric_values (
                            evaluation_run_id, scope, scope_key, operation, metric_name,
                            metric_version, value, sample_count
                        ) values (
                            :run_id, 'run', 'final', 'evaluate.answer', :name,
                            :metric_version, :value, :sample_count
                        )
                        """
                    ),
                    {
                        "run_id": run_id,
                        "name": name,
                        "metric_version": ANSWER_METRIC_VERSION,
                        "value": sum(values) / len(values),
                        "sample_count": len(values),
                    },
                )

    def _save_grade_step(
        self,
        case_result_id: UUID,
        sequence: int,
        grade: EvidenceGrade,
        answerability_correct: bool,
        latency_ms: int,
        *,
        expected_verdict: AnswerVerdict,
    ) -> None:
        operation = "grade.evidence"
        grade_step_details: dict[str, object] = {
            "expected": expected_verdict,
            "answerability_correct": answerability_correct,
        }
        grade_step_metric_details: dict[str, object] = {
            "expected": expected_verdict,
            "verdict": grade.verdict,
        }
        with self._engine.begin() as connection:
            step_id = cast(
                UUID,
                connection.execute(
                    text(
                        """
                        insert into rag_evaluation_step_results (
                            case_result_id, operation, operation_version, sequence, attempt,
                            output_kind, status, latency_ms, output, details
                        ) values (
                            :case_result_id, :operation, '1', :sequence, 0,
                            'grade_verdict', 'succeeded', :latency_ms,
                            cast(:output as jsonb), cast(:details as jsonb)
                        ) returning id
                        """
                    ),
                    {
                        "case_result_id": case_result_id,
                        "operation": operation,
                        "sequence": sequence,
                        "latency_ms": latency_ms,
                        "output": _json({"verdict": grade.verdict, "reason": grade.reason}),
                        "details": _json(grade_step_details),
                    },
                ).scalar_one(),
            )
            connection.execute(
                text(
                    """
                    insert into rag_evaluation_metric_values (
                        evaluation_run_id, evaluation_case_id, step_result_id, scope,
                        operation, metric_name, metric_version, value, sample_count, details
                    )
                    select results.evaluation_run_id, results.evaluation_case_id, :step_id,
                           'step', :operation, 'grade_answerability_correct', '1', :value, 1, cast(:details as jsonb)
                    from rag_evaluation_case_results results where results.id = :case_result_id
                    """
                ),
                {
                    "step_id": step_id,
                    "case_result_id": case_result_id,
                    "operation": operation,
                    "value": float(answerability_correct),
                    "details": _json(grade_step_metric_details),
                },
            )

    def _save_answer_step(
        self,
        case_result_id: UUID,
        sequence: int,
        answer: str,
        sources: Sequence[EvidenceSource],
        actual_verdict: AnswerVerdict,
        latency_ms: int,
    ) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    insert into rag_evaluation_step_results (
                        case_result_id, operation, operation_version, sequence, attempt,
                        output_kind, status, latency_ms, output, details
                    ) values (
                        :case_result_id, 'answer.generate', '1', :sequence, 0,
                        'answer', 'succeeded', :latency_ms,
                        cast(:output as jsonb), cast(:details as jsonb)
                    )
                    """
                ),
                {
                    "case_result_id": case_result_id,
                    "sequence": sequence,
                    "latency_ms": latency_ms,
                    "output": _json({"answer": answer, "sources": [dict(source) for source in sources]}),
                    "details": _json({"actual_verdict": actual_verdict}),
                },
            )

    def _save_answer_evaluation_step(
        self,
        case_result_id: UUID,
        sequence: int,
        evaluation: AnswerEvaluationResult,
        metrics: Mapping[str, float],
        latency_ms: int,
        *,
        judge_model: str,
        prompt_version: str,
        judge_skipped: bool,
    ) -> None:
        with self._engine.begin() as connection:
            step_id = cast(
                UUID,
                connection.execute(
                    text(
                        """
                        insert into rag_evaluation_step_results (
                            case_result_id, operation, operation_version, sequence, attempt,
                            output_kind, status, latency_ms, output, details
                        ) values (
                            :case_result_id, 'evaluate.answer', :operation_version, :sequence, 0,
                            'answer_evaluation', 'succeeded', :latency_ms,
                            cast(:output as jsonb), cast(:details as jsonb)
                        ) returning id
                        """
                    ),
                    {
                        "case_result_id": case_result_id,
                        "operation_version": prompt_version,
                        "sequence": sequence,
                        "latency_ms": latency_ms,
                        "output": _json(evaluation.model_dump(mode="json")),
                        "details": _json(
                            {
                                "judge_model": judge_model,
                                "judge_skipped": judge_skipped,
                            }
                        ),
                    },
                ).scalar_one(),
            )
            for name, value in metrics.items():
                connection.execute(
                    text(
                        """
                        insert into rag_evaluation_metric_values (
                            evaluation_run_id, evaluation_case_id, step_result_id, scope,
                            operation, metric_name, metric_version, value, sample_count
                        )
                        select results.evaluation_run_id, results.evaluation_case_id, :step_id,
                               'step', 'evaluate.answer', :metric_name, :metric_version, :value, 1
                        from rag_evaluation_case_results results where results.id = :case_result_id
                        """
                    ),
                    {
                        "step_id": step_id,
                        "case_result_id": case_result_id,
                        "metric_name": name,
                        "metric_version": ANSWER_METRIC_VERSION,
                        "value": value,
                    },
                )

    def _save_answer_evaluation_failure_step(
        self,
        case_result_id: UUID,
        sequence: int,
        latency_ms: int,
        *,
        judge_model: str,
        prompt_version: str,
        error: Exception,
    ) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    insert into rag_evaluation_step_results (
                        case_result_id, operation, operation_version, sequence, attempt,
                        output_kind, status, latency_ms, output, details, error_message
                    ) values (
                        :case_result_id, 'evaluate.answer', :operation_version, :sequence, 0,
                        'answer_evaluation', 'failed', :latency_ms, '{}'::jsonb,
                        cast(:details as jsonb), :error
                    )
                    """
                ),
                {
                    "case_result_id": case_result_id,
                    "operation_version": prompt_version,
                    "sequence": sequence,
                    "latency_ms": latency_ms,
                    "details": _json({"judge_model": judge_model, "judge_skipped": False}),
                    "error": str(error)[-2_000:],
                },
            )

    def _save_evidence_diagnosis(
        self,
        case_result_id: UUID,
        evaluation_case_id: UUID,
        journeys: Sequence[Mapping[str, object]],
    ) -> None:
        with self._engine.begin() as connection:
            for journey in journeys:
                connection.execute(
                    text(
                        """
                        insert into rag_evaluation_evidence_diagnostics (
                            evaluation_run_id, case_result_id, evaluation_case_id, evidence_id,
                            diagnostic_version, final_covered, last_visible_node,
                            first_loss_node, stage_journey
                        )
                        select results.evaluation_run_id, :case_result_id, :evaluation_case_id,
                               :evidence_id, :diagnostic_version, :final_covered,
                               :last_visible_node, :first_loss_node, cast(:stage_journey as jsonb)
                        from rag_evaluation_case_results results
                        where results.id = :case_result_id
                        on conflict (case_result_id, evidence_id, diagnostic_version)
                        do update set
                            final_covered = excluded.final_covered,
                            last_visible_node = excluded.last_visible_node,
                            first_loss_node = excluded.first_loss_node,
                            stage_journey = excluded.stage_journey,
                            created_at = now()
                        """
                    ),
                    {
                        "case_result_id": case_result_id,
                        "evaluation_case_id": evaluation_case_id,
                        "evidence_id": journey["evidence_id"],
                        "diagnostic_version": DIAGNOSTIC_VERSION,
                        "final_covered": bool(journey["final_covered"]),
                        "last_visible_node": journey.get("last_visible_node"),
                        "first_loss_node": journey.get("first_loss_node"),
                        "stage_journey": _json(journey["stages"]),
                    },
                )

    def _load_evidence(self, case_id: UUID) -> list[EvidenceAnchor]:
        with self._engine.connect() as connection:
            return [
                EvidenceAnchor(
                    id=cast(UUID, row["id"]),
                    recording_id=cast(UUID, row["source_recording_id"]),
                    source_chunk_id=cast(UUID | None, row["source_chunk_id"]),
                    quote=str(row["quote"]),
                    start_ms=int(row["start_ms"]),
                    end_ms=int(row["end_ms"]),
                    relevance=int(row["relevance"]),
                    content_checksum=str(row["content_checksum"]),
                )
                for row in connection.execute(
                    text("select * from rag_evaluation_evidence where evaluation_case_id = :case_id order by relevance desc, id"),
                    {"case_id": case_id},
                ).mappings()
            ]

    def _load_cited_evidence(self, sources: Sequence[EvidenceSource]) -> list[dict[str, object]]:
        if not sources:
            return []
        result: list[dict[str, object]] = []
        with self._engine.connect() as connection:
            for source in sources:
                recording = source["recording"]
                chunk = source["chunk"]
                rows = connection.execute(
                    text(
                        """
                        select coalesce(profiles.display_name, mappings.display_name, utterances.speaker_label) as speaker_label,
                               utterances.text, utterances.start_ms, utterances.end_ms
                        from utterance_segments utterances
                        left join recording_speaker_mappings mappings
                          on mappings.recording_id = utterances.recording_id
                         and mappings.speaker_cluster_id = utterances.speaker_cluster_id
                        left join speaker_profiles profiles on profiles.id = mappings.speaker_profile_id
                        where utterances.recording_id = :recording_id
                          and utterances.start_ms < :end_ms and utterances.end_ms > :start_ms
                        order by utterances.utterance_index
                        """
                    ),
                    {
                        "recording_id": UUID(recording["id"]),
                        "start_ms": chunk["startMs"],
                        "end_ms": chunk["endMs"],
                    },
                ).mappings()
                evidence_text = "\n".join(f"{row['speaker_label'] or 'Unknown Speaker'}: {row['text']}" for row in rows)
                result.append(
                    {
                        "index": source["index"],
                        "recording_id": recording["id"],
                        "chunk_id": chunk["id"],
                        "start_ms": chunk["startMs"],
                        "end_ms": chunk["endMs"],
                        "text": evidence_text,
                    }
                )
        return result

    def _start_case(self, run_id: UUID, case_id: UUID, query: str) -> UUID:
        with self._engine.begin() as connection:
            existing_id = connection.execute(
                text(
                    """
                    select id from rag_evaluation_case_results
                    where evaluation_run_id = :run_id and evaluation_case_id = :case_id
                    for update
                    """
                ),
                {"run_id": run_id, "case_id": case_id},
            ).scalar_one_or_none()
            if existing_id is not None:
                connection.execute(
                    text("delete from rag_evaluation_evidence_diagnostics where case_result_id = :result_id"),
                    {"result_id": existing_id},
                )
                connection.execute(
                    text("delete from rag_evaluation_step_results where case_result_id = :result_id"),
                    {"result_id": existing_id},
                )
                connection.execute(
                    text(
                        """
                        update rag_evaluation_case_results
                        set status = 'running', query_used = :query, latency_ms = null,
                            error_message = null, details = '{}'::jsonb, updated_at = now()
                        where id = :result_id
                        """
                    ),
                    {"result_id": existing_id, "query": query},
                )
                return cast(UUID, existing_id)
            return cast(
                UUID,
                connection.execute(
                    text(
                        """
                        insert into rag_evaluation_case_results (
                            evaluation_run_id, evaluation_case_id, status, query_used
                        ) values (:run_id, :case_id, 'running', :query)
                        returning id
                        """
                    ),
                    {"run_id": run_id, "case_id": case_id, "query": query},
                ).scalar_one(),
            )

    def _finish_case(
        self,
        result_id: UUID,
        latency_ms: int,
        *,
        succeeded: bool,
        error: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    update rag_evaluation_case_results
                    set status = :status, latency_ms = :latency_ms, error_message = :error,
                        details = cast(:details as jsonb),
                        updated_at = now()
                    where id = :result_id
                    """
                ),
                {
                    "result_id": result_id,
                    "status": "succeeded" if succeeded else "failed",
                    "latency_ms": latency_ms,
                    "error": None if succeeded else (error or "Unknown RAG evaluation error")[-2000:],
                    "details": _json(details or {}),
                },
            )

    def _update_progress(self, run_id: UUID, completed: int, failed: int) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    update evaluation_runs
                    set completed_case_count = :completed, failed_case_count = :failed, updated_at = now()
                    where id = :run_id
                    """
                ),
                {"run_id": run_id, "completed": completed, "failed": failed},
            )

    def _cancel_requested(self, run_id: UUID) -> bool:
        with self._engine.connect() as connection:
            return bool(
                connection.execute(
                    text("select cancel_requested from evaluation_runs where id = :run_id"),
                    {"run_id": run_id},
                ).scalar_one()
            )

    def _finish_cancelled(self, run_id: UUID) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    update evaluation_runs
                    set status = 'cancelled', finished_at = now(), updated_at = now()
                    where id = :run_id
                    """
                ),
                {"run_id": run_id},
            )

    def _settings_for_run(self, config: Mapping[str, object]) -> Settings:
        embedding = cast(Mapping[str, object], config.get("embedding") or {})
        rerank = cast(Mapping[str, object], config.get("rerank") or {})
        answer_judge = cast(Mapping[str, object], config.get("answer_judge") or {})
        legacy_vector_weight = _float_setting(config, "vector_weight", self._settings.rag_original_vector_weight)
        return self._settings.model_copy(
            update={
                "embedding_profile": str(embedding.get("profile", self._settings.embedding_profile)),
                "rag_hybrid_search_enabled": bool(config.get("hybrid_enabled", True)),
                "rag_online_default_model": str(config.get("online_default_model", self._settings.rag_online_default_model)),
                "rag_query_term_expansion_enabled": bool(config.get("query_term_expansion_enabled", self._settings.rag_query_term_expansion_enabled)),
                "rag_vector_candidate_limit": _int_setting(config, "vector_top_k", self._settings.rag_vector_candidate_limit),
                "rag_lexical_candidate_limit": _int_setting(config, "lexical_top_k", self._settings.rag_lexical_candidate_limit),
                "rag_fused_candidate_limit": _int_setting(config, "fused_top_k", self._settings.rag_fused_candidate_limit),
                "rag_rrf_k": _int_setting(config, "rrf_k", self._settings.rag_rrf_k),
                "rag_original_vector_weight": _float_setting(
                    config,
                    "original_vector_weight",
                    legacy_vector_weight,
                ),
                "rag_expanded_vector_weight": _float_setting(
                    config,
                    "expanded_vector_weight",
                    legacy_vector_weight,
                ),
                "rag_lexical_weight": _float_setting(config, "lexical_weight", self._settings.rag_lexical_weight),
                "rag_chunk_context_window_utterances": _int_setting(config, "context_window_utterances", self._settings.rag_chunk_context_window_utterances),
                "rag_rerank_enabled": bool(rerank.get("enabled", self._settings.rag_rerank_enabled)),
                "rag_rerank_model": str(rerank.get("model", self._settings.rag_rerank_model)),
                "rag_rerank_candidate_limit": _int_setting(rerank, "candidate_limit", self._settings.rag_rerank_candidate_limit),
                "rag_rerank_output_limit": _int_setting(rerank, "output_limit", self._settings.rag_rerank_output_limit),
                "rag_rerank_max_total_tokens": _int_setting(rerank, "max_total_tokens", self._settings.rag_rerank_max_total_tokens),
                "rag_answer_judge_model": str(answer_judge.get("model", self._settings.rag_answer_judge_model)),
                "rag_answer_judge_prompt_version": str(answer_judge.get("prompt_version", self._settings.rag_answer_judge_prompt_version)),
                "rag_answer_judge_max_output_tokens": _int_setting(
                    answer_judge,
                    "max_output_tokens",
                    self._settings.rag_answer_judge_max_output_tokens,
                ),
            }
        )

    @staticmethod
    def _scope_recording_ids(scope: Mapping[str, object], workspace_recording_ids: list[UUID]) -> list[UUID]:
        raw_ids = scope.get("recording_ids")
        requested = [UUID(str(item)) for item in cast(list[object], raw_ids)] if isinstance(raw_ids, list) else []
        allowed = set(workspace_recording_ids)
        return [item for item in requested if item in allowed] if requested else workspace_recording_ids


def _row_item(row: Mapping[str, object], *, score_kind: str) -> RankedItem:
    score = float(cast(float, row["score"]))
    return RankedItem(
        recording_id=cast(UUID, row["recording_id"]),
        source_chunk_id=cast(UUID, row["chunk_id"]),
        text=str(row["text"]),
        start_ms=int(cast(int, row["start_ms"])),
        end_ms=int(cast(int, row["end_ms"])),
        score=score,
        vector_score=score if score_kind == "vector" else None,
        lexical_score=score if score_kind == "lexical" else None,
        rrf_score=score if score_kind == "rrf" else None,
    )


def _evidence_item(item: Evidence, *, reranked: bool = False) -> RankedItem:
    return RankedItem(
        recording_id=item.recording.id,
        source_chunk_id=item.chunk.id,
        text=item.chunk.text,
        start_ms=item.chunk.start_ms,
        end_ms=item.chunk.end_ms,
        score=item.score,
        rerank_score=item.score if reranked else None,
    )


def _operation_items(event: RagOperationCompleted) -> list[RankedItem]:
    if event.operation == "retrieve.rrf" or event.operation.startswith(("retrieve.vector", "retrieve.lexical")):
        score_kind = event.operation.removeprefix("retrieve.").split(".", maxsplit=1)[0]
        rows = cast(Sequence[Mapping[str, object]], event.output)
        return [_row_item(row, score_kind=score_kind) for row in rows]
    evidence = cast(Sequence[Evidence], event.output)
    return [_evidence_item(item, reranked=event.operation == "retrieve.rerank") for item in evidence]


def _elapsed_ms(started: float) -> int:
    return max(0, round((perf_counter() - started) * 1000))


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _int_setting(values: Mapping[str, object], key: str, default: int) -> int:
    value = values.get(key)
    return int(value) if isinstance(value, (int, float, str)) else default


def _float_setting(values: Mapping[str, object], key: str, default: float) -> float:
    value = values.get(key)
    return float(value) if isinstance(value, (int, float, str)) else default


def _answer_verdict(value: object) -> AnswerVerdict:
    if value in ("direct_answer", "qualified_answer", "abstain"):
        return value
    raise ValueError(f"Unsupported answer verdict: {value!r}")
