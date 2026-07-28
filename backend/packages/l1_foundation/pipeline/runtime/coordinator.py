from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from uuid import UUID

from l1_foundation.pipeline.contracts import ResourceQueue, StageRunId
from l1_foundation.pipeline.runtime.executor import PipelineExecutor
from l1_foundation.pipeline.runtime.hooks import PipelineLifecycleHooks
from l1_foundation.pipeline.runtime.repository import PipelineRepository
from l1_foundation.task_runtime.scheduler import ResourceScheduler

logger = logging.getLogger(__name__)


class PipelineCoordinator:
    """Advance ready stages for one assembled pipeline; scheduling remains resource-only."""

    def __init__(self, repository: PipelineRepository, scheduler: ResourceScheduler, executor: PipelineExecutor, hooks: PipelineLifecycleHooks) -> None:
        self._repository = repository
        self._scheduler = scheduler
        self._executor = executor
        self._hooks = hooks
        self._inflight: set[asyncio.Task[None]] = set()

    async def run_once(self) -> bool:
        dispatched = False
        for stage in self._repository.ready_stages():
            stage_run_id = StageRunId(stage["id"])
            queue = ResourceQueue(str(stage["resource_queue"]))
            attempt_count = self._repository.mark_stage_running(stage_run_id)
            if attempt_count is None:
                continue
            task = asyncio.create_task(self._run_stage(stage_run_id, queue, attempt_count, stage), name=f"pipeline-stage-{stage_run_id}")
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)
            dispatched = True
        return dispatched

    @property
    def is_idle(self) -> bool:
        return not self._inflight

    async def _run_stage(self, stage_run_id: StageRunId, queue: ResourceQueue, attempt_count: int, stage_row: dict[str, object]) -> None:
        try:
            result = await self._scheduler.submit(queue, lambda: self._executor.execute(stage_run_id, attempt_count))
            subject_id = UUID(str(stage_row["subject_id"]))
            self._hooks.stage_succeeded(subject_id, str(stage_row["stage_name"]), result.output)
            self._repository.mark_stage_succeeded(stage_run_id, result.stage_result, result.artifacts)
            run = self._repository.get_run(UUID(str(stage_row["pipeline_run_id"])))
            self._hooks.run_state_changed(subject_id, str(run["status"]), run["error_message"])
        except Exception as error:
            retry_policy = self._executor.retry_policy(stage_run_id)
            delay = retry_policy.retry_delay_seconds(attempt_count)
            self._repository.mark_stage_retry(stage_run_id, str(error), delay)
            subject_id = UUID(str(stage_row["subject_id"]))
            run = self._repository.get_run(UUID(str(stage_row["pipeline_run_id"])))
            self._hooks.run_state_changed(subject_id, str(run["status"]), run["error_message"])
            logger.exception("pipeline stage failed and will retry: stage_run_id=%s", stage_run_id)

    async def shutdown(self) -> None:
        for task in self._inflight:
            task.cancel()
        if self._inflight:
            await asyncio.gather(*self._inflight, return_exceptions=True)


async def run_pipeline_coordinator(coordinator: PipelineCoordinator, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        if not await coordinator.run_once():
            with suppress(TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=0.2)
