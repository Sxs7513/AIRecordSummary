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
from l2_core.generation.contracts import (
    AggreMessageBlock,
    CreateGenerationCommand,
    GenerationAccessScope,
    GenerationKind,
    GenerationPriority,
    GenerationSnapshot,
    TextBlock,
)
from l2_core.generation.service import GenerationService
from l2_core.rag.adjudication.contracts import ClaimConfirmationDecision
from l2_core.rag.adjudication.web_research import ChromeAiOverviewSearchClient, GeminiGroundedSearchClient, GroundedSearchClient
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
        grounded_search_client = self._grounded_search_client(settings)
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
            asr_adjudication_enabled=settings.rag_asr_adjudication_enabled,
            asr_adjudication_web_search_enabled=settings.rag_asr_adjudication_web_search_enabled,
            asr_adjudication_auto_resolve_confidence=settings.rag_asr_adjudication_auto_resolve_confidence,
            asr_adjudication_audit_prompt_variant=settings.rag_asr_adjudication_audit_prompt_variant,
            asr_adjudication_audit_model=settings.rag_asr_adjudication_audit_model,
            asr_adjudication_audit_min_request_interval_seconds=(
                settings.rag_asr_adjudication_audit_min_request_interval_seconds
            ),
            grounded_search_client=grounded_search_client,
        )

    @staticmethod
    def _grounded_search_client(settings: Settings) -> GroundedSearchClient | None:
        if not settings.rag_asr_adjudication_enabled or not settings.rag_asr_adjudication_web_search_enabled:
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
        adjudication_user_decision: ClaimConfirmationDecision | None = None,
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
        existing_original_answer = None
        if resume_from_generation_id is not None:
            source = generation_service.get(resume_from_generation_id)
            existing_answer = "".join(block.value for block in source.blocks if isinstance(block, TextBlock)) or None
            aggregate = next((block for block in source.blocks if isinstance(block, AggreMessageBlock)), None)
            if aggregate is not None:
                variants = {item.variant: "".join(block.value for block in item.blocks) for item in aggregate.sub_message.sub_message_list}
                existing_answer = variants.get("corrected") or existing_answer
                existing_original_answer = variants.get("original") or None
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
                    hydrate_state=self._retriever.hydrate_checkpoint_state,
                    rerun_nodes=(
                        {
                            "fact_lookup-apply_user_adjudication_decision",
                            "fact_lookup-finalize",
                            "root-strategy_fact_lookup",
                        }
                        if adjudication_user_decision is not None
                        else set()
                    ),
                    repeatable_nodes={
                        "fact_lookup-adjudication_agent_step",
                        "fact_lookup-adjudication_execute_operation",
                    },
                )
                restored_state = await asyncio.to_thread(checkpoint.prepare)
                if resume_from_generation_id is not None:
                    await asyncio.to_thread(generation_service.delete_runtime_data, resume_from_generation_id)
                with rag_cancellation_scope(cancellation_check), rag_checkpoint_scope(checkpoint):
                    answer, sources, not_enough_evidence, message, confirmation = await self._graph.run(
                        query=query,
                        limit=limit,
                        scope_recording_ids=scope_recording_ids or [],
                        on_phase=sink.phase,
                        on_delta=sink.text,
                        history=history,
                        run_id=run_id,
                        existing_answer=existing_answer,
                        existing_original_answer=existing_original_answer,
                        aggregate_stream=sink,
                        restored_state=restored_state,
                        adjudication_user_decision=adjudication_user_decision,
                    )
            if sink.cancel_if_requested():
                log_event("generation_rag_cancelled", run_id, stage="after_graph")
                log_event("workflow_completed", run_id, status="cancelled", stage="after_graph", elapsed_ms=elapsed_ms(workflow_started))
                return
            if confirmation is not None:
                sink.block(confirmation)
                sink.succeed(
                    {
                        "notEnoughEvidence": False,
                        "message": None,
                        "interaction": {
                            "type": confirmation.type,
                            "request_id": str(confirmation.request_id),
                            "status": "pending",
                        },
                    },
                    [dict(source) for source in sources],
                    preserve_checkpoints=True,
                )
                log_event(
                    "workflow_completed",
                    run_id,
                    status="succeeded",
                    interaction=confirmation.type,
                    confirmation_items=len(confirmation.items),
                    elapsed_ms=elapsed_ms(workflow_started),
                )
                return
            if not_enough_evidence:
                sink.text(answer)
            sink.succeed(
                {"notEnoughEvidence": not_enough_evidence, "message": message},
                [dict(source) for source in sources],
                final_text=None if sink.has_aggregate_message else answer,
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
