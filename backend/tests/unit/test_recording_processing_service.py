from uuid import uuid4

from l1_foundation.pipeline.contracts import ArtifactRef, PipelineRunId
from l1_foundation.pipeline.definitions.graph import PipelineDefinition
from l2_core.application.recording_processing import StartRecordingProcessing
from l2_core.audio_processing.contracts import RecordingId
from l2_core.audio_processing.definition import recording_processing


class FakePipelineRepository:
    def __init__(self) -> None:
        self.recording_id: RecordingId | None = None
        self.definition_name: str | None = None

    def create_run(
        self, subject_type: str, subject_id: RecordingId, definition: PipelineDefinition, initial_artifacts: tuple[ArtifactRef, ...] = ()
    ) -> PipelineRunId:
        assert subject_type == "recording"
        assert initial_artifacts
        self.recording_id = subject_id
        self.definition_name = definition.name
        return PipelineRunId(uuid4())


def test_start_recording_processing_creates_the_declared_pipeline_run() -> None:
    repository = FakePipelineRepository()
    recording_id = RecordingId(uuid4())

    source_audio = ArtifactRef(artifact_type="audio.source", artifact_version="1", uri="recordings/test.mp3")
    run_id = StartRecordingProcessing(repository).execute(recording_id, source_audio)

    assert run_id
    assert repository.recording_id == recording_id
    assert repository.definition_name == recording_processing.name
