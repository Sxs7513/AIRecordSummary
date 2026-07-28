from __future__ import annotations

import asyncio
import contextlib
import signal

from l1_foundation.infrastructure.db.session import create_database_engine
from l1_foundation.infrastructure.storage.local import LocalStorage
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l1_foundation.pipeline.runtime.coordinator import PipelineCoordinator, run_pipeline_coordinator
from l1_foundation.pipeline.runtime.executor import PipelineExecutor
from l1_foundation.pipeline.runtime.repository import PipelineRepository
from l1_foundation.settings import get_settings
from l1_foundation.task_runtime.scheduler import ResourceScheduler
from l2_core.audio_processing.hooks import RecordingProcessingHooks
from l2_core.audio_processing.registry import build_recording_stage_registry, build_recording_summary_stage
from l2_core.generation.hub import GenerationStreamHub
from l2_core.generation.service import GenerationService


async def run() -> None:
    settings = get_settings()
    storage = LocalStorage(settings.resolved_local_storage_root)
    storage.initialize()
    engine = create_database_engine(settings)
    scheduler = ResourceScheduler()
    scheduler.start()
    generation_service = GenerationService(engine, GenerationStreamHub())
    artifact_store = ArtifactStore(settings.resolved_local_storage_root)
    summary_stage = build_recording_summary_stage(settings, artifact_store, generation_service)
    repository = PipelineRepository(engine)
    executor = PipelineExecutor(
        repository,
        build_recording_stage_registry(settings, artifact_store, generation_service, summary_stage),
        artifact_store,
    )
    coordinator = PipelineCoordinator(repository, scheduler, executor, RecordingProcessingHooks(engine))
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(signal_name, stop.set)
    try:
        await run_pipeline_coordinator(coordinator, stop)
    finally:
        stop.set()
        await coordinator.shutdown()
        scheduler.stop(5)
        engine.dispose()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
