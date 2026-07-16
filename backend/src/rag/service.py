from __future__ import annotations

import asyncio
import logging
from asyncio import Task, create_task, gather
from collections.abc import Callable
from uuid import UUID

from sqlalchemy import Engine

from access.recordings import RecordingAccessService
from auth.contracts import CurrentUser
from generation.contracts import CreateGenerationCommand, GenerationAccessScope, GenerationKind, GenerationPriority, GenerationSnapshot
from generation.service import GenerationService
from rag.contracts import RagHistoryMessage
from rag.graph import RagGraph
from rag.model import LocalLlamaChatModel, RagLanguageModel
from rag.observability import elapsed_ms, log_event, started_at
from rag.retrieval import RagRetriever
from settings import Settings
from task_runtime.resources import ResourceQueue
from task_runtime.scheduler import ResourceScheduler

logger = logging.getLogger("rag")


class RagService:
    """Application boundary for durable, LangGraph-orchestrated recording questions."""

    def __init__(self, engine: Engine, settings: Settings, scheduler: ResourceScheduler | None = None, model: RagLanguageModel | None = None) -> None:
        self._engine = engine
        self._settings = settings
        self._scheduler = scheduler
        self._model = model or LocalLlamaChatModel(settings)
        self._retriever = RagRetriever(engine, settings)
        self._access = RecordingAccessService(engine)
        self._graph = RagGraph(self._retriever, self._model, scheduler)

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
                input={"query": query, "limit": limit},
            )
        )

    def accessible_recording_ids(self, user: CurrentUser) -> list[UUID]:
        return self._access.accessible_recording_ids(user)

    async def execute_answer_generation(
        self,
        generation_service: GenerationService,
        run_id: UUID,
        query: str,
        limit: int,
        scope_recording_ids: list[UUID] | None = None,
        history: list[RagHistoryMessage] | None = None,
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
        emitter = generation_service.emitter(run_id)
        try:
            emitter.start()
            if emitter.cancel_if_requested():
                log_event("workflow_completed", run_id, status="cancelled", stage="before_graph", elapsed_ms=elapsed_ms(workflow_started))
                return
            answer, sources, not_enough_evidence, message = await self._graph.run(
                query=query,
                limit=limit,
                scope_recording_ids=scope_recording_ids or [],
                on_phase=emitter.phase,
                on_delta=emitter.text,
                history=history,
                run_id=run_id,
            )
            if emitter.cancel_if_requested():
                log_event("workflow_completed", run_id, status="cancelled", stage="after_graph", elapsed_ms=elapsed_ms(workflow_started))
                return
            if not_enough_evidence:
                emitter.text(answer)
            emitter.succeed({"notEnoughEvidence": not_enough_evidence, "message": message}, sources)
            log_event(
                "workflow_completed",
                run_id,
                status="succeeded",
                not_enough_evidence=not_enough_evidence,
                source_count=len(sources),
                answer_chars=len(answer),
                elapsed_ms=elapsed_ms(workflow_started),
            )
        except Exception as error:
            log_event(
                "workflow_completed",
                run_id,
                level=logging.ERROR,
                status="failed",
                error_type=type(error).__name__,
                elapsed_ms=elapsed_ms(workflow_started),
            )
            logger.exception("rag answer generation failed: run_id=%s", run_id)
            emitter.fail("rag_answer_failed", str(error) or "录音问答执行失败")
            raise
        finally:
            await self._release_models()

    async def _release_models(self) -> None:
        def release() -> None:
            release_model = getattr(self._model, "release", None)
            if callable(release_model):
                release_model()
            self._retriever.release()

        try:
            if self._scheduler is None:
                await asyncio.to_thread(release)
            else:
                await self._scheduler.submit(ResourceQueue.GPU_HIGH, release)
        except Exception:
            logger.exception("rag model cleanup failed")


class RagWorkflowRunner:
    """RAG-owned in-memory workflow runner; its LangGraph nodes submit resource work themselves."""

    def __init__(self, service: RagService, generation_service: GenerationService) -> None:
        self._service = service
        self._generation_service = generation_service
        self._tasks: set[Task[None]] = set()

    def submit(
        self,
        run_id: UUID,
        query: str,
        limit: int,
        scope_recording_ids: list[UUID] | None = None,
        history: list[RagHistoryMessage] | None = None,
        on_started: Callable[[UUID], None] | None = None,
        on_finished: Callable[[UUID], None] | None = None,
    ) -> None:
        log_event(
            "workflow_submitted",
            run_id,
            query_chars=len(query),
            limit=limit,
            scope_recording_count=len(scope_recording_ids or []),
            history_messages=len(history or []),
        )
        if on_started is not None:
            on_started(run_id)
        task = create_task(
            self._run(run_id, query, limit, scope_recording_ids, history, on_finished),
            name=f"rag-workflow-{run_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(
        self,
        run_id: UUID,
        query: str,
        limit: int,
        scope_recording_ids: list[UUID] | None,
        history: list[RagHistoryMessage] | None,
        on_finished: Callable[[UUID], None] | None,
    ) -> None:
        try:
            await self._service.execute_answer_generation(self._generation_service, run_id, query, limit, scope_recording_ids, history)
        finally:
            if on_finished is not None:
                on_finished(run_id)

    async def shutdown(self) -> None:
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await gather(*self._tasks, return_exceptions=True)
