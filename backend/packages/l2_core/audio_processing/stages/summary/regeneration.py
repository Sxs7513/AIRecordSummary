from __future__ import annotations

import logging
from asyncio import Task, create_task, gather, to_thread
from collections.abc import Callable
from uuid import UUID

from sqlalchemy import Engine, text

from l2_core.access.recordings import RecordingAccessService
from l2_core.audio_processing.contracts import RecordingId
from l2_core.audio_processing.projections import RecordingProjectionService
from l2_core.audio_processing.stages.recording_models import Utterance
from l2_core.audio_processing.stages.summary.generation import create_manual_summary_generation
from l2_core.audio_processing.stages.summary.stage import GenerateSummaryStage
from l2_core.audio_processing.stages.summary_embedding_indexing import SummaryEmbeddingIndexer
from l2_core.auth.contracts import CurrentUser
from l2_core.generation.event_sink import GenerationEventSink
from l2_core.generation.service import GenerationService
from l2_core.rag.queue import GenerationCommandPublisher, SummaryGenerationWorkItem

logger = logging.getLogger("audio_processing")


class RecordingSummaryNotReadyError(ValueError):
    """Raised when a recording has no corrected utterances to summarize yet."""


class RecordingSummaryQueueUnavailableError(RuntimeError):
    """Raised after compensating a summary generation that Kafka did not accept."""


class RecordingSummaryRegenerationService:
    """Regenerate one recording summary without creating a new recording pipeline run."""

    def __init__(
        self,
        engine: Engine,
        generation_service: GenerationService,
        summary_stage: GenerateSummaryStage,
        publisher: GenerationCommandPublisher | None = None,
        summary_embedding_indexer: SummaryEmbeddingIndexer | None = None,
    ) -> None:
        self._engine = engine
        self._generation_service = generation_service
        self._access = RecordingAccessService(engine)
        self._projections = RecordingProjectionService(engine)
        self._stage = summary_stage
        self._summary_embedding_indexer = summary_embedding_indexer
        self._publisher = publisher
        self._tasks: set[Task[None]] = set()

    async def regenerate(self, user: CurrentUser, recording_id: UUID) -> UUID:
        """Authorize, snapshot current utterances, and enqueue an interactive summary generation."""
        self._access.require_edit(recording_id, user)
        utterances = self.load_utterances(recording_id)
        if not utterances:
            raise RecordingSummaryNotReadyError("录音尚未生成可用于总结的润色文本")
        generation = create_manual_summary_generation(self._generation_service, recording_id)
        logger.info("summary：收到手动重新生成请求，recording_id=%s utterances=%d generation_id=%s", recording_id, len(utterances), generation.id)
        if self._publisher is not None:
            try:
                await self._publisher.submit_summary(
                    SummaryGenerationWorkItem(
                        run_id=generation.id,
                        recording_id=recording_id,
                        generation=self._generation_service.command(generation.id),
                    )
                )
            except Exception as error:
                self._generation_service.event_sink(generation.id).fail(
                    "kafka_unavailable",
                    str(error) or type(error).__name__,
                    retryable=True,
                )
                raise RecordingSummaryQueueUnavailableError("Generation queue unavailable") from error
            return generation.id
        task = create_task(
            self.execute(generation.id, RecordingId(recording_id), utterances),
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

    async def execute(self, generation_id: UUID, recording_id: RecordingId, utterances: list[Utterance] | None = None) -> None:
        resolved_utterances = utterances if utterances is not None else self.load_utterances(recording_id)
        sink = self._generation_service.event_sink(generation_id)
        try:
            sink.start()
            if sink.cancel_if_requested():
                return
            output = await to_thread(
                self._stage.generate,
                resolved_utterances,
                self._generation_progress(sink),
                sink,
            )
            if sink.cancel_if_requested():
                return
            self._projections.project(recording_id, "generate_summary", output)
            if self._summary_embedding_indexer is not None:
                try:
                    embedding = await to_thread(
                        self._summary_embedding_indexer.encode,
                        output.summary_text,
                        self._recording_title(recording_id),
                    )
                    self._projections.project(recording_id, "summary_embedding_indexing", embedding)
                except Exception:
                    logger.warning("summary：重新生成后的向量化失败 recording_id=%s", recording_id, exc_info=True)
            sink.succeed(output.model_dump(mode="json"))
            logger.info("summary：手动重新生成完成，recording_id=%s generation_id=%s", recording_id, generation_id)
        except Exception as error:
            if sink.cancel_if_requested():
                logger.info(
                    "summary：手动重新生成已取消，recording_id=%s generation_id=%s",
                    recording_id,
                    generation_id,
                )
                return
            logger.exception("summary：手动重新生成失败，recording_id=%s generation_id=%s", recording_id, generation_id)
            sink.fail("summary_regeneration_failed", str(error) or "录音总结重新生成失败", retryable=True)

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

    def _recording_title(self, recording_id: UUID) -> str:
        with self._engine.connect() as connection:
            return str(
                connection.execute(
                    text("select title from recordings where id = :recording_id"),
                    {"recording_id": recording_id},
                ).scalar_one()
            )

    @staticmethod
    def _generation_progress(sink: GenerationEventSink) -> Callable[[int, str], None]:
        return lambda percent, message: sink.phase("generating", message, percent)
