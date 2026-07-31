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

from l1_foundation.llm import LlmProvider
from l1_foundation.observability import InstrumentedModelClient
from l1_foundation.settings import Settings
from l1_foundation.worker import ComputeCommand, ExecutionScope, SyncWorkerClient, WorkerClient, execution_scope
from l2_core.rag.contracts import Evidence, EvidenceGrade
from l2_core.rag.graph import RagGraph
from l2_core.rag.hooks import RagNodeCompleted, RagOperationCompleted
from l2_core.rag.retrieval import RagRetriever
from l2_core.rag_evaluation.contracts import EvidenceAnchor, RankedItem, RetrievalMetrics
from l2_core.rag_evaluation.evidence_matcher import match_ranked_item
from l2_core.rag_evaluation.metrics import mean_metrics, percentile, retrieval_metrics

logger = logging.getLogger("evaluation")
MAX_SAVED_CANDIDATES = 50
FINAL_METRICS_KEY = "__final__"


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
                        where status = 'queued' and evaluator_type = 'rag_retrieval'
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
                    where status = 'running' and evaluator_type = 'rag_retrieval'
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
                ).mappings().one()
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
        grade_passes: list[float] = []
        completed = failed = 0

        async with self._model_worker_client(settings) as model_worker:
            graph = RagGraph(
                retriever,
                InstrumentedModelClient(cast(WorkerClient, model_worker)),
                online_provider=LlmProvider(settings.rag_answer_provider),
                context_size=settings.rag_context_size,
                plan_local_input_tokens=settings.rag_plan_local_input_tokens,
                max_total_tokens=settings.rag_run_max_total_tokens,
                route_model_profile=settings.rag_route_model_profile,
                node_model_profile=settings.rag_node_model_profile,
                query_term_expansion_enabled=settings.rag_query_term_expansion_enabled,
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
                        scope_ids = self._scope_recording_ids(
                            cast(Mapping[str, object], case["scope"]), workspace_recording_ids
                        )
                        hook = EvaluationTraceHook()
                        state = await graph.run_retrieval(
                            query=str(case["query"]),
                            limit=settings.rag_fused_candidate_limit,
                            scope_recording_ids=scope_ids,
                            run_id=result_id,
                            hook=hook,
                        )
                        state = await graph.grade_retrieval(state, hook=hook)
                        grade = state.get("grade")
                        if grade is None:
                            raise RuntimeError("RAG evaluation completed without an evidence grade")
                        grade_passed = grade.verdict != "abstain"
                        operations = hook.operations or [self._empty_terminal_operation(hook, state.get("route_error"))]
                        final_metric: RetrievalMetrics | None = None
                        for sequence, operation in enumerate(operations):
                            items = _operation_items(operation)
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
                        grade_node = next((item for item in reversed(hook.nodes) if item.node == "grade"), None)
                        self._save_grade_step(
                            result_id,
                            len(operations),
                            grade,
                            grade_passed,
                            round(grade_node.elapsed_ms) if grade_node is not None else 0,
                        )

                        latency = _elapsed_ms(started)
                        case_latencies.append(latency)
                        route = state.get("route")
                        self._finish_case(
                            result_id,
                            latency,
                            succeeded=True,
                            details={
                                "route_strategy": route.strategy_id if route is not None else None,
                                "route_error": state.get("route_error"),
                                "grade_verdict": grade.verdict,
                                "grade_passed": grade_passed,
                            },
                        )
                        grade_passes.append(float(grade_passed))
                        completed += 1
                    except Exception as error:
                        logger.exception("RAG 评测：Case 失败 run_id=%s case_id=%s", run_id, case_id)
                        self._finish_case(result_id, _elapsed_ms(started), succeeded=False, error=str(error))
                        failed += 1
                    self._update_progress(run_id, completed, failed)
            finally:
                await asyncio.to_thread(retriever.release)

        self._save_run_metrics(run_id, metrics_by_operation, case_latencies, grade_passes)
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
    def _empty_terminal_operation(
        hook: EvaluationTraceHook, route_error: str | None
    ) -> RagOperationCompleted:
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
                        "details": _json({"text": item.text, "start_ms": item.start_ms, "end_ms": item.end_ms}),
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
                               'step', :operation, :metric_name, '1', :value, 1
                        from rag_evaluation_case_results results where results.id = :case_result_id
                        """
                    ),
                    {
                        "step_id": step_id,
                        "case_result_id": case_result_id,
                        "operation": operation,
                        "metric_name": name,
                        "value": value,
                    },
                )
        return metrics

    def _save_run_metrics(
        self,
        run_id: UUID,
        metrics_by_operation: Mapping[str, Sequence[RetrievalMetrics]],
        case_latencies: Sequence[int],
        grade_passes: Sequence[float],
    ) -> None:
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
                            ) values (:run_id, 'operation', :operation, :name, '1', :value, :sample_count)
                            """
                        ),
                        {
                            "run_id": run_id,
                            "operation": operation,
                            "name": name,
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
                        ) values (:run_id, 'run', 'final', :operation, :name, '1', :value, :sample_count)
                        """
                    ),
                    {
                        "run_id": run_id,
                        "operation": final_operation,
                        "name": name,
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
                        :run_id, 'run', 'final', 'grade.evidence', 'grade_pass_rate',
                        '1', :value, :sample_count, cast(:details as jsonb)
                    )
                    """
                ),
                {
                    "run_id": run_id,
                    "value": sum(grade_passes) / len(grade_passes) if grade_passes else 0.0,
                    "sample_count": len(grade_passes),
                    "details": _json({"expected": "non_abstain"}),
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

    def _save_grade_step(
        self,
        case_result_id: UUID,
        sequence: int,
        grade: EvidenceGrade,
        passed: bool,
        latency_ms: int,
    ) -> None:
        operation = "grade.evidence"
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
                        "details": _json({"expected": "non_abstain", "passed": passed}),
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
                           'step', :operation, 'grade_pass', '1', :value, 1, cast(:details as jsonb)
                    from rag_evaluation_case_results results where results.id = :case_result_id
                    """
                ),
                {
                    "step_id": step_id,
                    "case_result_id": case_result_id,
                    "operation": operation,
                    "value": float(passed),
                    "details": _json({"expected": "non_abstain", "verdict": grade.verdict}),
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
        return self._settings.model_copy(
            update={
                "embedding_model": str(embedding.get("model", self._settings.embedding_model)),
                "embedding_dimensions": _int_setting(embedding, "dimensions", self._settings.embedding_dimensions),
                "rag_hybrid_search_enabled": bool(config.get("hybrid_enabled", True)),
                "rag_query_term_expansion_enabled": bool(
                    config.get("query_term_expansion_enabled", self._settings.rag_query_term_expansion_enabled)
                ),
                "rag_vector_candidate_limit": _int_setting(config, "vector_top_k", self._settings.rag_vector_candidate_limit),
                "rag_lexical_candidate_limit": _int_setting(config, "lexical_top_k", self._settings.rag_lexical_candidate_limit),
                "rag_fused_candidate_limit": _int_setting(config, "fused_top_k", self._settings.rag_fused_candidate_limit),
                "rag_rrf_k": _int_setting(config, "rrf_k", self._settings.rag_rrf_k),
                "rag_vector_weight": _float_setting(config, "vector_weight", self._settings.rag_vector_weight),
                "rag_lexical_weight": _float_setting(config, "lexical_weight", self._settings.rag_lexical_weight),
                "rag_chunk_context_window_utterances": _int_setting(
                    config, "context_window_utterances", self._settings.rag_chunk_context_window_utterances
                ),
                "rag_rerank_enabled": bool(rerank.get("enabled", self._settings.rag_rerank_enabled)),
                "rag_rerank_model": str(rerank.get("model", self._settings.rag_rerank_model)),
                "rag_rerank_candidate_limit": _int_setting(rerank, "candidate_limit", self._settings.rag_rerank_candidate_limit),
                "rag_rerank_output_limit": _int_setting(rerank, "output_limit", self._settings.rag_rerank_output_limit),
                "rag_rerank_max_total_tokens": _int_setting(
                    rerank, "max_total_tokens", self._settings.rag_rerank_max_total_tokens
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
    if event.operation in {"retrieve.vector", "retrieve.lexical", "retrieve.rrf"}:
        score_kind = event.operation.removeprefix("retrieve.")
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
