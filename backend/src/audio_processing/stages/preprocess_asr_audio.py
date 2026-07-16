from __future__ import annotations

import asyncio
from pathlib import Path

from audio_processing.stages.recording_models import NormalizedAudioOutput, PreprocessAsrAudioInput
from pipeline.contracts import ArtifactPayload, ResourceQueue, RetryPolicy, StageContext, StageResult
from pipeline.runtime.artifact_store import ArtifactStore


class PreprocessAsrAudioStage:
    """Create a time-preserving, lightly conditioned ASR input waveform."""

    name = "preprocess_asr_audio"
    version = "1"
    resource_queue = ResourceQueue.CPU
    retry_policy = RetryPolicy(initial_backoff_seconds=10)
    input_model = PreprocessAsrAudioInput

    def __init__(self, storage_root: Path, artifact_store: ArtifactStore, enabled: bool, ffmpeg_binary: str = "ffmpeg") -> None:
        self._storage_root = storage_root.resolve()
        self._artifact_store = artifact_store
        self._enabled = enabled
        self._ffmpeg_binary = ffmpeg_binary

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

        target_relative = Path("asr-preprocessed") / str(context.subject_id) / f"{context.pipeline_run_id}.wav"
        target = self._storage_root / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        context.report_progress(5, "执行 ASR 音频前置处理")
        process = await asyncio.create_subprocess_exec(
            self._ffmpeg_binary,
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-af",
            "highpass=f=80,acompressor=threshold=-30dB:ratio=2.5:attack=20:release=250:makeup=4,loudnorm=I=-19:TP=-1.5:LRA=11,alimiter=limit=0.95",
            "-c:a",
            "pcm_s16le",
            str(target),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg ASR preprocessing failed: {stderr.decode('utf-8', errors='replace')[-2000:]}")
        output = NormalizedAudioOutput(storage_path=target_relative.as_posix(), sample_rate_hz=16000, channels=1, format="wav")
        context.report_progress(100, "ASR 音频前置处理完成")
        return StageResult(
            output=output,
            artifacts=(ArtifactPayload(artifact_type="audio.asr_preprocessed", data=output.model_dump(mode="json")),),
        )

    def _resolve(self, uri: str) -> Path:
        path = (self._storage_root / uri).resolve()
        if self._storage_root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Normalized audio does not exist: {uri}")
        return path
