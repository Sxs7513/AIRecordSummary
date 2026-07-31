from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from sqlalchemy import Engine

from l1_foundation.llm import LlmProvider
from l1_foundation.observability import InstrumentedModelClient, ObservabilityClient, ObservabilityScope, observation_scope
from l1_foundation.settings import Settings
from l1_foundation.streaming import SyncRedisStreamStore
from l1_foundation.worker import SyncWorkerClient, WorkerClient
from l2_core.access.recordings import RecordingAccessService
from l2_core.auth.contracts import CurrentUser
from l2_core.generation.contracts import CreateGenerationCommand, GenerationAccessScope, GenerationKind, GenerationPriority, GenerationSnapshot
from l2_core.generation.service import GenerationService
from l2_core.rag.checkpoint import RagCheckpointSession, RagCheckpointStore, rag_input_hash
from l2_core.rag.contracts import RagHistoryMessage
from l2_core.rag.execution_middleware import RagExecutionCancelled, rag_cancellation_scope, rag_checkpoint_scope, throttled_cancellation_check
from l2_core.rag.graph import RagGraph
from l2_core.rag.observability import elapsed_ms, log_event, started_at
from l2_core.rag.retrieval import RagRetriever
from l2_core.rag.token_budget import RagTokenBudgetExceeded

logger = logging.getLogger("rag")
RAG_TOKEN_BUDGET_EXCEEDED_MESSAGE = "问题较复杂，已超出本次处理上限，请缩小问题范围后重试。"


class RagService:
    """Application boundary for durable, LangGraph-orchestrated recording questions."""

    def __init__(
        self,
        engine: Engine,
        settings: Settings,
        worker_client: WorkerClient,
        sync_worker_client: SyncWorkerClient,
        observability_client: ObservabilityClient,
        redis_store: SyncRedisStreamStore,
    ) -> None:
        self._engine = engine
        self._settings = settings
        self._observability_client = observability_client
        self._retriever = RagRetriever(engine, settings, sync_worker_client)
        self._checkpoint_store = RagCheckpointStore(redis_store, settings.rag_checkpoint_ttl_seconds)
        self._access = RecordingAccessService(engine)
        self._graph = RagGraph(
            self._retriever,
            InstrumentedModelClient(worker_client),
            online_provider=LlmProvider(settings.rag_answer_provider),
            context_size=settings.rag_context_size,
            plan_local_input_tokens=settings.rag_plan_local_input_tokens,
            max_total_tokens=settings.rag_run_max_total_tokens,
            route_model_profile=settings.rag_route_model_profile,
            node_model_profile=settings.rag_node_model_profile,
            query_term_expansion_enabled=settings.rag_query_term_expansion_enabled,
        )

    def create_answer_generation(
        self, generation_service: GenerationService, user: CurrentUser, query: str, limit: int, idempotency_key: str
    ) -> GenerationSnapshot:
        return generation_service.create(
            CreateGenerationCommand(
                kind=GenerationKind.TEXT,
                priority=GenerationPriority.INTERACTIVE,
                idempotency_key=idempotency_key,
                parent_type="rag_query",
                parent_id=idempotency_key,
                access_scope=GenerationAccessScope(owner_user_id=user.id),
                input={"query": query, "limit": limit, "workspace_id": str(user.current_workspace_id)},
            )
        )

    def accessible_recording_ids(self, user: CurrentUser) -> list[UUID]:
        return self._access.accessible_recording_ids(user)

    async def execute_answer_generation(
        self,
        generation_service: GenerationService,
        run_id: UUID,
        workspace_id: UUID,
        query: str,
        limit: int,
        scope_recording_ids: list[UUID] | None = None,
        history: list[RagHistoryMessage] | None = None,
        resume_from_generation_id: UUID | None = None,
    ) -> None:
        workflow_started = started_at()
        log_event(
            "workflow_started",
            run_id,
            query_chars=len(query),
            limit=limit,
            scope_recording_count=len(scope_recording_ids or []),
            history_messages=len(history or []),
        )
        sink = generation_service.event_sink(run_id)
        existing_answer = None
        if resume_from_generation_id is not None:
            source = generation_service.get(resume_from_generation_id)
            existing_answer = "".join(block.value for block in source.blocks) or None
        try:
            sink.start()
            if sink.cancel_if_requested():
                log_event("generation_rag_cancelled", run_id, stage="before_graph")
                log_event("workflow_completed", run_id, status="cancelled", stage="before_graph", elapsed_ms=elapsed_ms(workflow_started))
                return
            with observation_scope(
                self._observability_client,
                ObservabilityScope(workspace_id=workspace_id, generation_run_id=run_id),
            ):
                cancellation_check = throttled_cancellation_check(lambda: generation_service.is_cancel_requested(run_id))
                checkpoint = RagCheckpointSession(
                    store=self._checkpoint_store,
                    generation_id=run_id,
                    source_generation_id=resume_from_generation_id,
                    input_hash=rag_input_hash(query, limit, scope_recording_ids or []),
                    hydrate_state=getattr(self._retriever, "hydrate_checkpoint_state", lambda state: state),
                )
                restored_state = await asyncio.to_thread(checkpoint.prepare)
                if resume_from_generation_id is not None:
                    await asyncio.to_thread(generation_service.delete_runtime_data, resume_from_generation_id)
                with rag_cancellation_scope(cancellation_check), rag_checkpoint_scope(checkpoint):
                    answer, sources, not_enough_evidence, message = await self._graph.run(
                        query=query,
                        limit=limit,
                        scope_recording_ids=scope_recording_ids or [],
                        on_phase=sink.phase,
                        on_delta=sink.text,
                        history=history,
                        run_id=run_id,
                        existing_answer=existing_answer,
                        restored_state=restored_state,
                    )
            if sink.cancel_if_requested():
                log_event("generation_rag_cancelled", run_id, stage="after_graph")
                log_event("workflow_completed", run_id, status="cancelled", stage="after_graph", elapsed_ms=elapsed_ms(workflow_started))
                return
            if not_enough_evidence:
                sink.text(answer)
            sink.succeed(
                {"notEnoughEvidence": not_enough_evidence, "message": message},
                sources,
                final_text=answer,
            )
            log_event(
                "workflow_completed",
                run_id,
                status="succeeded",
                not_enough_evidence=not_enough_evidence,
                source_count=len(sources),
                answer_chars=len(answer),
                elapsed_ms=elapsed_ms(workflow_started),
            )
        except RagExecutionCancelled:
            if sink.cancel_if_requested():
                log_event("generation_rag_cancelled", run_id, stage="graph_nodes", boundary="cooperative")
                log_event(
                    "workflow_completed",
                    run_id,
                    status="cancelled",
                    stage="graph_nodes",
                    elapsed_ms=elapsed_ms(workflow_started),
                )
                return
            raise
        except RagTokenBudgetExceeded as error:
            log_event(
                "workflow_completed",
                run_id,
                level=logging.WARNING,
                status="rejected",
                reason="token_budget_exceeded",
                elapsed_ms=elapsed_ms(workflow_started),
            )
            logger.info("rag token budget exceeded: run_id=%s details=%s", run_id, error)
            sink.fail("rag_token_budget_exceeded", RAG_TOKEN_BUDGET_EXCEEDED_MESSAGE)
        except Exception as error:
            if generation_service.is_cancel_requested(run_id):
                sink.cancel_if_requested()
                log_event(
                    "generation_rag_cancelled",
                    run_id,
                    stage="exception_boundary",
                    boundary="compute",
                    error_type=type(error).__name__,
                )
                log_event(
                    "workflow_completed",
                    run_id,
                    status="cancelled",
                    stage="exception_boundary",
                    error_type=type(error).__name__,
                    elapsed_ms=elapsed_ms(workflow_started),
                )
                return
            log_event(
                "workflow_completed",
                run_id,
                level=logging.ERROR,
                status="failed",
                error_type=type(error).__name__,
                elapsed_ms=elapsed_ms(workflow_started),
            )
            logger.exception("rag answer generation failed: run_id=%s", run_id)
            sink.fail("rag_answer_failed", str(error) or "录音问答执行失败")
            raise
        finally:
            await self._release_retriever()

    async def _release_retriever(self) -> None:
        def release() -> None:
            self._retriever.release()

        try:
            await asyncio.to_thread(release)
        except Exception:
            logger.exception("rag retriever cleanup failed")
