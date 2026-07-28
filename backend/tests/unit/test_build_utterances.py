import asyncio
from pathlib import Path
from uuid import uuid4

from l1_foundation.pipeline.contracts import ArtifactPayload, PipelineRunId, PipelineSubjectId, StageContext, StageRunId
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l2_core.audio_processing.stages.build_utterances import BuildUtterancesStage
from l2_core.audio_processing.stages.recording_models import BuildUtterancesInput, TranscriptOutput, TranscriptSegment


def _segment(speaker: str, start_ms: int, end_ms: int, text: str) -> TranscriptSegment:
    return TranscriptSegment(
        source_diarization_segment_id=f"{speaker}:{start_ms}:{end_ms}",
        start_ms=start_ms,
        end_ms=end_ms,
        text=text,
        speaker_cluster_id=speaker,
        speaker_label=f"Speaker {speaker}",
    )


def test_build_utterances_projects_upstream_segments_without_merging_or_reordering(tmp_path: Path) -> None:
    storage = ArtifactStore(tmp_path)
    context = StageContext(PipelineSubjectId(uuid4()), PipelineRunId(uuid4()), StageRunId(uuid4()), 1)
    transcript = TranscriptOutput(
        provider="qwen_asr",
        model_name="test-asr",
        language="Chinese",
        segments=[
            _segment("A", 1_200, 2_000, "world"),
            _segment("A", 0, 1_000, "hello"),
            _segment("B", 2_100, 3_000, "另一位"),
        ],
    )
    artifact = storage.write_json(
        context.subject_id,
        context.pipeline_run_id,
        context.stage_run_id,
        "align_transcript",
        ArtifactPayload(artifact_type="transcript.aligned", data=transcript.model_dump(mode="json")),
    )

    result = asyncio.run(BuildUtterancesStage(storage).run(context, BuildUtterancesInput(transcript=artifact)))

    assert [item.text for item in result.output.segments] == ["world", "hello", "另一位"]
    assert [item.utterance_index for item in result.output.segments] == [0, 1, 2]
    assert [item.source_segment_indexes for item in result.output.segments] == [[0], [1], [2]]
    assert [item.source_diarization_segment_ids for item in result.output.segments] == [
        ["A:1200:2000"],
        ["A:0:1000"],
        ["B:2100:3000"],
    ]
