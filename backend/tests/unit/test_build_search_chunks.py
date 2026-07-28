import asyncio
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from l1_foundation.pipeline.contracts import ArtifactPayload, PipelineRunId, PipelineSubjectId, StageContext, StageRunId
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l2_core.audio_processing.stages.build_search_chunks import BuildSearchChunksStage
from l2_core.audio_processing.stages.build_search_chunks.contracts import TopicSection
from l2_core.audio_processing.stages.recording_models import BuildSearchChunksInput, Utterance, UtterancesOutput


def test_build_search_chunks_keeps_source_utterance_provenance(tmp_path: Path) -> None:
    storage = ArtifactStore(tmp_path)
    context = StageContext(PipelineSubjectId(uuid4()), PipelineRunId(uuid4()), StageRunId(uuid4()), 1)
    utterances = UtterancesOutput(
        segments=[
            Utterance(
                utterance_index=0,
                start_ms=0,
                end_ms=1_000,
                text="第一句",
                speaker_cluster_id="A",
                speaker_label="Speaker A",
                source_diarization_segment_ids=["A:0:1000"],
            ),
            Utterance(
                utterance_index=1,
                start_ms=1_100,
                end_ms=2_000,
                text="第二句",
                speaker_cluster_id="B",
                speaker_label="Speaker B",
                source_diarization_segment_ids=["B:1100:2000"],
            ),
        ]
    )
    artifact = storage.write_json(
        context.subject_id,
        context.pipeline_run_id,
        context.stage_run_id,
        "build_utterances",
        ArtifactPayload(artifact_type="utterances.final", data=utterances.model_dump(mode="json")),
    )

    result = asyncio.run(BuildSearchChunksStage(storage).run(context, BuildSearchChunksInput(utterances=artifact)))

    assert result.output.chunks[0].source_utterance_indexes == [0, 1]
    assert result.output.chunks[0].source_diarization_segment_ids == ["A:0:1000", "B:1100:2000"]
    assert result.output.build_method == "deterministic_fallback"


def test_build_search_chunks_uses_continuous_topic_sections(tmp_path: Path) -> None:
    class FakeDetector:
        def detect(self, utterances: list[Utterance]) -> list[TopicSection]:
            assert len(utterances) == 2
            return [
                TopicSection(start_utterance_index=0, end_utterance_index=0, topic="话题一"),
                TopicSection(start_utterance_index=1, end_utterance_index=1, topic="话题二"),
            ]

        def release(self) -> None:
            return None

    storage = ArtifactStore(tmp_path)
    context = StageContext(PipelineSubjectId(uuid4()), PipelineRunId(uuid4()), StageRunId(uuid4()), 1)
    utterances = UtterancesOutput(
        segments=[
            Utterance(
                utterance_index=index,
                start_ms=index * 1000,
                end_ms=(index + 1) * 1000,
                text=f"第{index + 1}句",
                speaker_cluster_id="A",
                speaker_label="Speaker A",
                source_diarization_segment_ids=[f"A:{index * 1000}:{(index + 1) * 1000}"],
            )
            for index in range(2)
        ]
    )
    artifact = storage.write_json(
        context.subject_id,
        context.pipeline_run_id,
        context.stage_run_id,
        "build_utterances",
        ArtifactPayload(artifact_type="utterances.final", data=utterances.model_dump(mode="json")),
    )
    stage = BuildSearchChunksStage(storage, topic_detection_enabled=True, topic_model_path=tmp_path / "model.gguf")
    stage._detector = cast(Any, FakeDetector())

    result = asyncio.run(stage.run(context, BuildSearchChunksInput(utterances=artifact)))

    assert result.output.build_method == "topic_boundary"
    assert [chunk.topic for chunk in result.output.chunks] == ["话题一", "话题二"]
    assert all(chunk.build_method == "topic_boundary" for chunk in result.output.chunks)
