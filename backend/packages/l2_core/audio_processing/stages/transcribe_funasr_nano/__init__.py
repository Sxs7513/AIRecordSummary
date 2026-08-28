from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

from l1_foundation.files import FileStore
from l1_foundation.pipeline.contracts import ArtifactPayload, RetryPolicy, StageContext, StageResult
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l1_foundation.worker import WorkerClient
from l2_core.audio_processing.stages.recording_models import AsrWindowTranscript, AsrWindowTranscriptOutput, DiarizationOutput, TranscribeAsrInput
from l2_core.audio_processing.stages.transcribe_funasr_nano.engine import FunAsrNanoConfig, FunAsrNanoEngine
from l2_core.audio_processing.worker_tasks import AsrInferenceBatchResult, asr_inference_batch_command


class FunAsrNanoTranscribeStage:
    name = "transcribe_funasr_nano"
    version = "3"
    retry_policy = RetryPolicy(initial_backoff_seconds=30)
    input_model = TranscribeAsrInput

    def __init__(
        self,
        file_store: FileStore,
        artifact_store: ArtifactStore,
        model_name: str,
        language: str,
        model_cache_root: Path,
        _context: str = "",
        max_inference_batch_size: int = 1,
        speech_window_target_duration_ms: int = 30_000,
        speech_window_max_duration_ms: int = 80_000,
        speech_window_overlap_ms: int = 500,
        hotwords: list[str] | None = None,
        worker_client: WorkerClient | None = None,
    ) -> None:
        self._file_store = file_store
        self._artifact_store = artifact_store
        self._worker_client = worker_client
        self._engine = FunAsrNanoEngine(
            FunAsrNanoConfig(
                model_name=model_name,
                language=language,
                model_cache_root=model_cache_root,
                hotwords=hotwords or [],
                max_inference_batch_size=1,
                speech_window_target_duration_ms=speech_window_target_duration_ms,
                speech_window_max_duration_ms=speech_window_max_duration_ms,
                speech_window_overlap_ms=speech_window_overlap_ms,
            )
        )

    async def try_restore(self, context: StageContext, _input_payload: TranscribeAsrInput) -> StageResult[AsrWindowTranscriptOutput] | None:
        return self._artifact_store.try_restore_json(
            context.pipeline_run_id, context.stage_run_id, self.name, self.version, "transcript.asr_windows", AsrWindowTranscriptOutput
        )

    async def run(self, context: StageContext, input_payload: TranscribeAsrInput) -> StageResult[AsrWindowTranscriptOutput]:
        descriptor = self._artifact_store.read_json(input_payload.audio)
        storage_path = descriptor.get("storage_path")
        if not isinstance(storage_path, str):
            raise ValueError("ASR audio artifact is missing storage_path")
        diarization = DiarizationOutput.model_validate(self._artifact_store.read_json(input_payload.diarization))
        if self._worker_client is None:
            raise RuntimeError("FunAsrNanoTranscribeStage requires WorkerClient")
        audio_path = self._resolve_storage_path(storage_path)
        input_keys: list[str] = []
        with (
            tempfile.TemporaryDirectory(prefix="funasr-input-") as directory,
            self._engine.prepare_inference_batch(audio_path, diarization.segments, Path(directory)) as (windows, paths),
        ):
            if paths:
                batch_id = uuid4().hex
                input_keys = [self._file_store.put_file(path, key=f"compute-inputs/{batch_id}/{path.name}") for path in paths]
                try:
                    inference = await self._worker_client.execute(
                        asr_inference_batch_command("funasr_nano", input_keys),
                        result_type=AsrInferenceBatchResult,
                        on_progress=lambda progress, message: context.report_progress(round(progress * 100), message or "FunASR"),
                    )
                finally:
                    for key in input_keys:
                        self._file_store.delete_file(key)
            else:
                inference = AsrInferenceBatchResult(provider="funasr_nano", model_name=self._engine.model_name, items=[])
        if len(inference.items) != len(windows):
            raise RuntimeError(f"FunASR inference result count mismatch: expected {len(windows)}, got {len(inference.items)}")
        output = AsrWindowTranscriptOutput(
            provider="funasr_nano",
            model_name=inference.model_name,
            language=next((item.language for item in inference.items if item.language), None),
            windows=[
                AsrWindowTranscript(
                    window_index=window.window_index,
                    input_start_ms=window.input_start_ms,
                    input_end_ms=window.input_end_ms,
                    core_start_ms=window.core_start_ms,
                    core_end_ms=window.core_end_ms,
                    language=item.language,
                    text=item.text,
                    core_diarization_segment_ids=window.diarization_segment_ids,
                )
                for window, item in zip(windows, inference.items, strict=True)
                if item.text
            ],
        )
        return StageResult(
            output=output,
            artifacts=(ArtifactPayload(artifact_type="transcript.asr_windows", data=output.model_dump(mode="json")),),
        )

    def _resolve_storage_path(self, uri: str) -> Path:
        return self._file_store.get_file_by_key(uri)


__all__ = ["FunAsrNanoTranscribeStage"]
