from __future__ import annotations

import logging
from asyncio import Task, create_task, gather
from collections.abc import Callable
from uuid import UUID

from sqlalchemy import Engine, text

from access.recordings import RecordingAccessService
from audio_processing.contracts import RecordingId
from audio_processing.projections import RecordingProjectionService
from audio_processing.stages.recording_models import Utterance
from audio_processing.stages.summary.generation import create_manual_summary_generation
from audio_processing.stages.summary.stage import GenerateSummaryStage
from auth.contracts import CurrentUser
from generation.emitter import StreamEmitter
from generation.service import GenerationService
from task_runtime.resources import ResourceQueue
from task_runtime.scheduler import ResourceScheduler

logger = logging.getLogger(__name__)


class RecordingSummaryNotReadyError(ValueError):
    """Raised when a recording has no corrected utterances to summarize yet."""


class RecordingSummaryRegenerationService:
    """Regenerate one recording summary without creating a new recording pipeline run."""

    def __init__(self, engine: Engine, scheduler: ResourceScheduler, generation_service: GenerationService, summary_stage: GenerateSummaryStage) -> None:
        self._engine = engine
        self._scheduler = scheduler
        self._generation_service = generation_service
        self._access = RecordingAccessService(engine)
        self._projections = RecordingProjectionService(engine)
        self._stage = summary_stage
        self._tasks: set[Task[None]] = set()

    async def regenerate(self, user: CurrentUser, recording_id: UUID) -> UUID:
        """Authorize, snapshot current utterances, and enqueue an interactive summary generation."""
        self._access.require_edit(recording_id, user)
        utterances = self.load_utterances(recording_id)
        if not utterances:
            raise RecordingSummaryNotReadyError("录音尚未生成可用于总结的润色文本")
        generation = create_manual_summary_generation(self._generation_service, recording_id)
        logger.info("summary：收到手动重新生成请求，recording_id=%s utterances=%d generation_id=%s", recording_id, len(utterances), generation.id)
        task = create_task(
            self._execute(generation.id, RecordingId(recording_id), utterances),
            name=f"recording-summary-regeneration-{generation.id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return generation.id

    async def shutdown(self) -> None:
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await gather(*self._tasks, return_exceptions=True)

    async def _execute(self, generation_id: UUID, recording_id: RecordingId, utterances: list[Utterance]) -> None:
        emitter = self._generation_service.emitter(generation_id)
        try:
            emitter.start()
            if emitter.cancel_if_requested():
                return
            output = await self._scheduler.submit(
                ResourceQueue.GPU_NORMAL,
                lambda: self._stage.generate(utterances, self._generation_progress(emitter), emitter),
            )
            if emitter.cancel_if_requested():
                return
            self._projections.project(recording_id, "generate_summary", output)
            emitter.succeed(output.model_dump(mode="json"))
            logger.info("summary：手动重新生成完成，recording_id=%s generation_id=%s", recording_id, generation_id)
        except Exception as error:
            logger.exception("summary：手动重新生成失败，recording_id=%s generation_id=%s", recording_id, generation_id)
            emitter.fail("summary_regeneration_failed", str(error) or "录音总结重新生成失败", retryable=True)

    def load_utterances(self, recording_id: UUID) -> list[Utterance]:
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    select utterance_index, start_ms, end_ms, text, speaker_cluster_id, speaker_label
                    from utterance_segments
                    where recording_id = :recording_id
                    order by utterance_index
                    """
                ),
                {"recording_id": str(recording_id)},
            ).mappings()
            return [
                Utterance.model_validate(
                    {
                        **dict(row),
                        "speaker_cluster_id": row["speaker_cluster_id"] or "unknown",
                        "speaker_label": row["speaker_label"] or "Unknown Speaker",
                        "source_diarization_segment_ids": [],
                    }
                )
                for row in rows
            ]

    @staticmethod
    def _generation_progress(emitter: StreamEmitter) -> Callable[[int, str], None]:
        return lambda percent, message: emitter.phase("generating", message, percent)
