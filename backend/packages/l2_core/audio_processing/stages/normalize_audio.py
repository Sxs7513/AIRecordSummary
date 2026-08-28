from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from l1_foundation.files import FileStore
from l1_foundation.pipeline.contracts import ArtifactPayload, RetryPolicy, StageContext, StageResult
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l2_core.audio_processing.stages.recording_models import NormalizeAudioInput, NormalizedAudioOutput


class NormalizeAudioStage:
    """Convert arbitrary uploaded audio into the one format used by all models."""

    name = "normalize_audio"
    version = "1"
    retry_policy = RetryPolicy(initial_backoff_seconds=10)
    input_model = NormalizeAudioInput

    def __init__(self, file_store: FileStore, artifact_store: ArtifactStore, ffmpeg_binary: str = "ffmpeg") -> None:
        self._file_store = file_store
        self._artifact_store = artifact_store
        self._ffmpeg_binary = ffmpeg_binary

    async def try_restore(self, context: StageContext, _input_payload: NormalizeAudioInput) -> StageResult[NormalizedAudioOutput] | None:
        restored = self._artifact_store.try_restore_json(
            context.pipeline_run_id,
            context.stage_run_id,
            self.name,
            self.version,
            "audio.normalized",
            NormalizedAudioOutput,
        )
        if restored is None:
            return None
        try:
            self._resolve_storage_path(restored.output.storage_path)
        except FileNotFoundError:
            return None
        return restored

    async def run(self, context: StageContext, input_payload: NormalizeAudioInput) -> StageResult[NormalizedAudioOutput]:
        source_path = self._resolve_storage_path(input_payload.source_audio.uri)
        target_key = f"normalized/{context.subject_id}/{context.pipeline_run_id}.wav"
        with TemporaryDirectory(prefix="normalize-audio-") as temporary_directory:
            target_path = Path(temporary_directory) / "normalized.wav"
            process = await asyncio.create_subprocess_exec(
                self._ffmpeg_binary,
                "-y",
                "-i",
                str(source_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(target_path),
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
                raise RuntimeError(f"ffmpeg normalization failed: {stderr.decode('utf-8', errors='replace')[-2000:]}")
            self._file_store.put_file(target_path, key=target_key)
        output = NormalizedAudioOutput(storage_path=target_key, sample_rate_hz=16000, channels=1, format="wav")
        return StageResult(
            output=output,
            artifacts=(ArtifactPayload(artifact_type="audio.normalized", data=output.model_dump(mode="json")),),
        )

    def _resolve_storage_path(self, uri: str) -> Path:
        return self._file_store.get_file_by_key(uri)
