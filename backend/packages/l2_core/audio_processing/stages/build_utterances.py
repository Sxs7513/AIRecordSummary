from __future__ import annotations

from l1_foundation.pipeline.contracts import ArtifactPayload, ResourceQueue, RetryPolicy, StageContext, StageResult
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l2_core.audio_processing.stages.recording_models import BuildUtterancesInput, TranscriptOutput, Utterance, UtterancesOutput


class BuildUtterancesStage:
    """Project aligned transcript segments one-to-one into business utterances."""

    name = "build_utterances"
    version = "4"
    resource_queue = ResourceQueue.CPU
    retry_policy = RetryPolicy(initial_backoff_seconds=10)
    input_model = BuildUtterancesInput

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self._artifact_store = artifact_store

    async def run(self, context: StageContext, input_payload: BuildUtterancesInput) -> StageResult[UtterancesOutput]:
        context.report_progress(10, "读取对齐转写")
        transcript = TranscriptOutput.model_validate(self._artifact_store.read_json(input_payload.transcript))
        utterances: list[Utterance] = []
        total = max(1, len(transcript.segments))
        for source_index, segment in enumerate(transcript.segments):
            utterances.append(
                Utterance(
                    utterance_index=source_index,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    text=segment.text,
                    speaker_cluster_id=segment.speaker_cluster_id,
                    speaker_label=segment.speaker_label,
                    source_segment_indexes=[source_index],
                    source_diarization_segment_ids=[segment.source_diarization_segment_id],
                )
            )
            context.report_progress(
                20 + round(65 * (source_index + 1) / total),
                f"生成最终转写段落 {source_index + 1}/{len(transcript.segments)}",
            )
        context.report_progress(95, "校验最终转写段落来源")
        output = UtterancesOutput(segments=utterances)
        return StageResult(output=output, artifacts=(ArtifactPayload(artifact_type="utterances.final", data=output.model_dump(mode="json")),))
