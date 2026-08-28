from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from l1_foundation.files import FileStore
from l1_foundation.pipeline.contracts import ArtifactPayload, RetryPolicy, StageContext, StageResult
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l2_core.audio_processing.stages.recording_models import NormalizedAudioOutput, PreprocessAsrAudioInput

ASR_AUDIO_OUTPUT_ARGS = ("-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le")


class PreprocessAsrAudioStage:
    """Create a time-preserving ASR waveform without altering speech characteristics."""

    name = "preprocess_asr_audio"
    version = "5"
    retry_policy = RetryPolicy(initial_backoff_seconds=10)
    input_model = PreprocessAsrAudioInput

    def __init__(self, file_store: FileStore, artifact_store: ArtifactStore, enabled: bool, ffmpeg_binary: str = "ffmpeg") -> None:
        self._file_store = file_store
        self._artifact_store = artifact_store
        self._enabled = enabled
        self._ffmpeg_binary = ffmpeg_binary

    async def try_restore(self, context: StageContext, _input_payload: PreprocessAsrAudioInput) -> StageResult[NormalizedAudioOutput] | None:
        restored = self._artifact_store.try_restore_json(
            context.pipeline_run_id,
            context.stage_run_id,
            self.name,
            self.version,
            "audio.asr_preprocessed",
            NormalizedAudioOutput,
        )
        if restored is None:
            return None
        try:
            self._resolve(restored.output.storage_path)
        except FileNotFoundError:
            return None
        return restored

    async def run(self, context: StageContext, input_payload: PreprocessAsrAudioInput) -> StageResult[NormalizedAudioOutput]:
        descriptor = self._artifact_store.read_json(input_payload.audio)
        storage_path = descriptor.get("storage_path")
        if not isinstance(storage_path, str):
            raise ValueError("Normalized audio artifact is missing storage_path")
        source = self._resolve(storage_path)
        if not self._enabled:
            output = NormalizedAudioOutput(storage_path=storage_path, sample_rate_hz=16000, channels=1, format="wav")
            return StageResult(
                output=output,
                artifacts=(ArtifactPayload(artifact_type="audio.asr_preprocessed", data=output.model_dump(mode="json")),),
            )
        target_key = f"asr-preprocessed/{context.subject_id}/{context.pipeline_run_id}.wav"
        context.report_progress(5, "执行 ASR 音频前置处理")
        with TemporaryDirectory(prefix="preprocess-asr-") as temporary_directory:
            target = Path(temporary_directory) / "preprocessed.wav"
            process = await asyncio.create_subprocess_exec(
                self._ffmpeg_binary,
                "-y",
                "-i",
                str(source),
                "-vn",
                *ASR_AUDIO_OUTPUT_ARGS,
                str(target),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await process.communicate()
            except asyncio.CancelledError:
                if process.returncode is None:
                    process.terminate()
                    await process.wait()
                raise
            if process.returncode != 0:
                raise RuntimeError(f"ffmpeg ASR preprocessing failed: {stderr.decode('utf-8', errors='replace')[-2000:]}")
            self._file_store.put_file(target, key=target_key)
        output = NormalizedAudioOutput(storage_path=target_key, sample_rate_hz=16000, channels=1, format="wav")
        context.report_progress(100, "ASR 音频前置处理完成")
        return StageResult(
            output=output,
            artifacts=(ArtifactPayload(artifact_type="audio.asr_preprocessed", data=output.model_dump(mode="json")),),
        )

    def _resolve(self, uri: str) -> Path:
        return self._file_store.get_file_by_key(uri)
