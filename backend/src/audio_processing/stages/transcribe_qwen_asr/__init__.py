from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from audio_processing.stages.recording_models import (
    AsrWindowTranscript,
    AsrWindowTranscriptOutput,
    DiarizationOutput,
    TranscribeQwenAsrInput,
)
from audio_processing.stages.transcribe_qwen_asr.engine import QwenAsrConfig, QwenAsrEngine
from pipeline.contracts import ArtifactPayload, ResourceQueue, RetryPolicy, StageContext, StageResult
from pipeline.runtime.artifact_store import ArtifactStore


class QwenAsrTranscribeStage:
    """Transcribe standardized pyannote speaker segments with an in-process Qwen model."""

    name = "transcribe_qwen_asr"
    version = "5"
    resource_queue = ResourceQueue.GPU_HIGH
    retry_policy = RetryPolicy(initial_backoff_seconds=30)
    input_model = TranscribeQwenAsrInput

    @property
    def provider(self) -> Literal["qwen_asr"]:
        return "qwen_asr"

    def __init__(
        self,
        storage_root: Path,
        artifact_store: ArtifactStore,
        model_name: str,
        language: str,
        model_cache_root: Path,
        context: str = "",
        max_inference_batch_size: int = 4,
        speech_window_target_duration_ms: int = 30_000,
        speech_window_max_duration_ms: int = 80_000,
        speech_window_overlap_ms: int = 500,
    ) -> None:
        self._storage_root = storage_root.resolve()
        self._artifact_store = artifact_store
        config = QwenAsrConfig(
            model_name=model_name,
            language=language,
            model_cache_root=model_cache_root,
            context=context,
            max_inference_batch_size=max_inference_batch_size,
            speech_window_target_duration_ms=speech_window_target_duration_ms,
            speech_window_max_duration_ms=speech_window_max_duration_ms,
            speech_window_overlap_ms=speech_window_overlap_ms,
        )
        self._engine = self._build_engine(config)

    def _build_engine(self, config: QwenAsrConfig) -> QwenAsrEngine:
        return QwenAsrEngine(config)

    async def run(self, context: StageContext, input_payload: TranscribeQwenAsrInput) -> StageResult[AsrWindowTranscriptOutput]:
        audio_descriptor = self._artifact_store.read_json(input_payload.audio)
        audio_storage_path = audio_descriptor.get("storage_path")
        if not isinstance(audio_storage_path, str):
            raise ValueError("Normalized audio artifact is missing storage_path")
        diarization = DiarizationOutput.model_validate(self._artifact_store.read_json(input_payload.diarization))
        try:
            result = await asyncio.to_thread(
                self._engine.transcribe_continuous_windows,
                self._resolve_storage_path(audio_storage_path),
                diarization.segments,
                context.report_progress,
            )
            output = AsrWindowTranscriptOutput(
                provider="qwen_asr",
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
                        text=segment,
                        core_diarization_segment_ids=window.diarization_segment_ids,
                    )
                    for window, segment in result.windows
                    if segment
                ],
            )
            return StageResult(
                output=output,
                artifacts=(ArtifactPayload(artifact_type="transcript.asr_windows", data=output.model_dump(mode="json")),),
            )
        finally:
            self._engine.release()

    def _resolve_storage_path(self, uri: str) -> Path:
        path = (self._storage_root / uri).resolve()
        if self._storage_root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Normalized audio does not exist: {uri}")
        return path


__all__: Sequence[str] = ("QwenAsrTranscribeStage",)
