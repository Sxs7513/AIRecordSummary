from __future__ import annotations

import asyncio
from pathlib import Path

from audio_processing.stages.recording_models import NormalizeAudioInput, NormalizedAudioOutput
from pipeline.contracts import ArtifactPayload, ResourceQueue, RetryPolicy, StageContext, StageResult


class NormalizeAudioStage:
    """Convert arbitrary uploaded audio into the one format used by all models."""

    name = "normalize_audio"
    version = "1"
    resource_queue = ResourceQueue.CPU
    retry_policy = RetryPolicy(initial_backoff_seconds=10)
    input_model = NormalizeAudioInput

    def __init__(self, storage_root: Path, ffmpeg_binary: str = "ffmpeg") -> None:
        self._storage_root = storage_root.resolve()
        self._ffmpeg_binary = ffmpeg_binary

    async def run(self, context: StageContext, input_payload: NormalizeAudioInput) -> StageResult[NormalizedAudioOutput]:
        source_path = self._resolve_storage_path(input_payload.source_audio.uri)
        relative_target = Path("normalized") / str(context.subject_id) / f"{context.pipeline_run_id}.wav"
        target_path = self._storage_root / relative_target
        target_path.parent.mkdir(parents=True, exist_ok=True)
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
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg normalization failed: {stderr.decode('utf-8', errors='replace')[-2000:]}")
        output = NormalizedAudioOutput(storage_path=relative_target.as_posix(), sample_rate_hz=16000, channels=1, format="wav")
        return StageResult(
            output=output,
            artifacts=(ArtifactPayload(artifact_type="audio.normalized", data=output.model_dump(mode="json")),),
        )

    def _resolve_storage_path(self, uri: str) -> Path:
        path = (self._storage_root / uri).resolve()
        if self._storage_root not in path.parents:
            raise ValueError("Source audio URI escapes storage root")
        if not path.is_file():
            raise FileNotFoundError(f"Source audio does not exist: {uri}")
        return path
