from __future__ import annotations

from typing import Protocol

from audio_processing.contracts import RecordingId
from audio_processing.definition import recording_processing
from pipeline.contracts import ArtifactRef, PipelineRunId
from pipeline.definitions.graph import PipelineDefinition


class PipelineRunCreator(Protocol):
    def create_run(
        self,
        subject_type: str,
        subject_id: RecordingId,
        definition: PipelineDefinition,
        initial_artifacts: tuple[ArtifactRef, ...] = (),
    ) -> PipelineRunId: ...


class StartRecordingProcessing:
    """Application use case that starts the recording pipeline assembled by audio_processing."""

    def __init__(self, pipeline_repository: PipelineRunCreator, definition: PipelineDefinition = recording_processing) -> None:
        self._pipeline_repository = pipeline_repository
        self._definition = definition

    def execute(self, recording_id: RecordingId, source_audio: ArtifactRef) -> PipelineRunId:
        return self._pipeline_repository.create_run("recording", recording_id, self._definition, (source_audio,))
