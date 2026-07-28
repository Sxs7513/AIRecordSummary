from __future__ import annotations

import asyncio
from pathlib import Path

from l1_foundation.pipeline.contracts import ArtifactPayload, ResourceQueue, RetryPolicy, StageContext, StageResult
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l2_core.audio_processing.stages.recording_models import AsrWindowTranscript, AsrWindowTranscriptOutput, DiarizationOutput, TranscribeAsrInput
from l2_core.audio_processing.stages.transcribe_funasr_nano.engine import FunAsrNanoConfig, FunAsrNanoEngine


class FunAsrNanoTranscribeStage:
    name = "transcribe_funasr_nano"
    version = "3"
    resource_queue = ResourceQueue.GPU_HIGH
    retry_policy = RetryPolicy(initial_backoff_seconds=30)
    input_model = TranscribeAsrInput

    def __init__(
        self,
        storage_root: Path,
        artifact_store: ArtifactStore,
        model_name: str,
        language: str,
        model_cache_root: Path,
        _context: str = "",
        max_inference_batch_size: int = 4,
        speech_window_target_duration_ms: int = 30_000,
        speech_window_max_duration_ms: int = 80_000,
        speech_window_overlap_ms: int = 500,
        hotwords: list[str] | None = None,
    ) -> None:
        self._storage_root = storage_root.resolve()
        self._artifact_store = artifact_store
        self._engine = FunAsrNanoEngine(
            FunAsrNanoConfig(
                model_name,
                language,
                model_cache_root,
                hotwords or [],
                max_inference_batch_size,
                speech_window_target_duration_ms,
                speech_window_max_duration_ms,
                speech_window_overlap_ms,
            )
        )

    async def run(self, context: StageContext, input_payload: TranscribeAsrInput) -> StageResult[AsrWindowTranscriptOutput]:
        descriptor = self._artifact_store.read_json(input_payload.audio)
        storage_path = descriptor.get("storage_path")
        if not isinstance(storage_path, str):
            raise ValueError("ASR audio artifact is missing storage_path")
        diarization = DiarizationOutput.model_validate(self._artifact_store.read_json(input_payload.diarization))
        path = (self._storage_root / storage_path).resolve()
        try:
            result = await asyncio.to_thread(self._engine.transcribe_continuous_windows, path, diarization.segments, context.report_progress)
            output = AsrWindowTranscriptOutput(
                provider="funasr_nano",
                model_name=self._engine.model_name,
                language=result.language,
                windows=[
                    AsrWindowTranscript(
                        window_index=window.window_index,
                        input_start_ms=window.input_start_ms,
                        input_end_ms=window.input_end_ms,
                        core_start_ms=window.core_start_ms,
                        core_end_ms=window.core_end_ms,
                        language=result.language,
                        text=text,
                        core_diarization_segment_ids=window.diarization_segment_ids,
                    )
                    for window, text in result.windows
                    if text
                ],
            )
            return StageResult(
                output=output,
                artifacts=(ArtifactPayload(artifact_type="transcript.asr_windows", data=output.model_dump(mode="json")),),
            )
        finally:
            self._engine.release()


__all__ = ["FunAsrNanoTranscribeStage"]
