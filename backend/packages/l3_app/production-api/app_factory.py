from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Engine

from l1_foundation.infrastructure.db.session import create_database_engine
from l1_foundation.infrastructure.storage.local import LocalStorage
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l1_foundation.pipeline.runtime.coordinator import PipelineCoordinator, run_pipeline_coordinator
from l1_foundation.pipeline.runtime.executor import PipelineExecutor
from l1_foundation.pipeline.runtime.repository import PipelineRepository
from l1_foundation.settings import Settings, get_settings
from l1_foundation.task_runtime.scheduler import ResourceScheduler
from l2_core.application.storage_cleanup import StorageCleanupService
from l2_core.audio_processing.definition import build_recording_processing
from l2_core.audio_processing.hooks import RecordingProcessingHooks
from l2_core.audio_processing.registry import build_recording_stage_registry, build_recording_summary_stage
from l2_core.audio_processing.stages.summary.regeneration import RecordingSummaryRegenerationService
from l2_core.audio_processing.stages.summary.stage import GenerateSummaryStage
from l2_core.generation.hub import GenerationStreamHub
from l2_core.generation.service import GenerationService
from l2_core.rag.service import RagService, RagWorkflowRunner
from router import api_router

logger = logging.getLogger(__name__)
WORKER_SHUTDOWN_GRACE_SECONDS = 5.0


def _configure_rag_logger() -> None:
    """Expose diagnostic RAG milestones without enabling noisy global request logs."""
    rag_logger = logging.getLogger("rag")
    rag_logger.setLevel(logging.INFO)
    rag_logger.propagate = False
    if rag_logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    rag_logger.addHandler(handler)


def _configure_audio_processing_logger() -> None:
    """Expose audio pipeline milestones without enabling noisy global request logs."""
    audio_logger = logging.getLogger("audio_processing")
    audio_logger.setLevel(logging.INFO)
    audio_logger.propagate = False
    if audio_logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    audio_logger.addHandler(handler)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Initialize HTTP infrastructure and its queue workers as one application service."""
    storage = LocalStorage(app.state.settings.resolved_local_storage_root)
    storage.initialize()
    app.state.storage = storage
    app.state.database_engine = create_database_engine(app.state.settings)
    app.state.generation_hub = GenerationStreamHub()
    app.state.generation_service = GenerationService(app.state.database_engine, app.state.generation_hub)
    artifact_store = ArtifactStore(app.state.settings.resolved_local_storage_root)
    summary_stage = build_recording_summary_stage(app.state.settings, artifact_store, app.state.generation_service)
    scheduler = ResourceScheduler()
    scheduler.start()
    app.state.resource_scheduler = scheduler
    app.state.rag_service = RagService(app.state.database_engine, app.state.settings, scheduler)
    app.state.rag_workflow_runner = RagWorkflowRunner(app.state.rag_service, app.state.generation_service)
    app.state.recording_summary_regeneration_service = RecordingSummaryRegenerationService(
        app.state.database_engine, scheduler, app.state.generation_service, summary_stage
    )
    try:
        cleanup_result = StorageCleanupService(app.state.database_engine, storage).remove_inactive_pipeline_intermediates()
        logger.info(
            "startup storage cleanup completed: orphan_pipeline_runs_removed=%d orphan_stage_runs_removed=%d "
            "recording_directories_removed=%d normalized_files_removed=%d artifact_runs_removed=%d bytes_reclaimed=%d",
            cleanup_result.orphan_pipeline_runs_removed,
            cleanup_result.orphan_stage_runs_removed,
            cleanup_result.recording_directories_removed,
            cleanup_result.normalized_files_removed,
            cleanup_result.artifact_runs_removed,
            cleanup_result.bytes_reclaimed,
        )
    except Exception:
        logger.exception("startup storage cleanup failed; leaving existing intermediate files untouched")
    coordinator: PipelineCoordinator | None = None
    coordinator_stop: asyncio.Event | None = None
    coordinator_task: asyncio.Task[None] | None = None
    if app.state.start_pipeline_worker:
        coordinator = _build_pipeline_coordinator(
            app.state.settings, app.state.database_engine, app.state.generation_service, scheduler, artifact_store, summary_stage
        )
        coordinator_stop = asyncio.Event()
        coordinator_task = asyncio.create_task(run_pipeline_coordinator(coordinator, coordinator_stop), name="pipeline-coordinator")
    try:
        yield
    finally:
        if coordinator is not None and coordinator_stop is not None and coordinator_task is not None:
            coordinator_stop.set()
            await coordinator.shutdown()
            await coordinator_task
        await app.state.recording_summary_regeneration_service.shutdown()
        await app.state.rag_workflow_runner.shutdown()
        scheduler.stop(WORKER_SHUTDOWN_GRACE_SECONDS)
        app.state.database_engine.dispose()


def create_app(
    settings: Settings | None = None,
    *,
    router: APIRouter | None = None,
    start_pipeline_worker: bool = True,
) -> FastAPI:
    """Build the HTTP application with no import-time side effects."""
    _configure_rag_logger()
    _configure_audio_processing_logger()
    configured_settings = settings or get_settings()
    app = FastAPI(title=configured_settings.app_name, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Accept", "Authorization", "Content-Type", "Last-Event-ID"],
    )
    app.state.settings = configured_settings
    app.state.start_pipeline_worker = start_pipeline_worker
    app.state.recording_processing_definition = build_recording_processing(configured_settings.asr_provider)
    app.include_router(router or api_router, prefix=configured_settings.api_prefix)
    return app


def _build_pipeline_coordinator(
    settings: Settings,
    engine: Engine,
    generation_service: GenerationService,
    scheduler: ResourceScheduler,
    artifact_store: ArtifactStore,
    summary_stage: GenerateSummaryStage,
) -> PipelineCoordinator:
    repository = PipelineRepository(engine)
    executor = PipelineExecutor(
        repository,
        build_recording_stage_registry(settings, artifact_store, generation_service, summary_stage),
        artifact_store,
    )
    return PipelineCoordinator(repository, scheduler, executor, RecordingProcessingHooks(engine))
