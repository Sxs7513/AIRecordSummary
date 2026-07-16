from __future__ import annotations

import asyncio
import contextlib
import gc
import subprocess
import tempfile
from collections.abc import Callable, Generator, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from audio_processing.stages.recording_models import (
    AlignedTranscriptToken,
    AlignTranscriptInput,
    CorrectedAsrWindowTranscriptOutput,
    DiarizationOutput,
    DiarizationSegment,
    TranscriptOutput,
    TranscriptSegment,
)
from infrastructure.huggingface import resolve_local_snapshot
from pipeline.contracts import ArtifactPayload, ArtifactRef, ResourceQueue, RetryPolicy, StageContext, StageResult
from pipeline.runtime.artifact_store import ArtifactStore


class ForcedAlignItem(Protocol):
    text: str
    start_time: float
    end_time: float


class AlignTranscriptStage:
    """Align final window text once, then attribute aligned tokens to pyannote turns."""

    name = "align_transcript"
    version = "1"
    resource_queue = ResourceQueue.GPU_HIGH
    retry_policy = RetryPolicy(initial_backoff_seconds=30)
    input_model = AlignTranscriptInput

    def __init__(self, storage_root: Path, artifact_store: ArtifactStore, model_name: str, model_cache_root: Path) -> None:
        self._storage_root = storage_root.resolve()
        self._artifact_store = artifact_store
        self._model_name = model_name
        self._model_cache_root = model_cache_root
        self._aligner: Any | None = None

    async def run(self, context: StageContext, input_payload: AlignTranscriptInput) -> StageResult[TranscriptOutput]:
        audio = self._audio_path(input_payload.audio)
        diarization = DiarizationOutput.model_validate(self._artifact_store.read_json(input_payload.diarization))
        corrected = CorrectedAsrWindowTranscriptOutput.model_validate(self._artifact_store.read_json(input_payload.transcript))
        try:
            output = await asyncio.to_thread(self._align, audio, diarization.segments, corrected, context.report_progress)
            return StageResult(
                output=output,
                artifacts=(ArtifactPayload(artifact_type="transcript.aligned", data=output.model_dump(mode="json")),),
            )
        finally:
            self._release()

    def _align(
        self,
        audio: Path,
        diarization: Sequence[DiarizationSegment],
        corrected: CorrectedAsrWindowTranscriptOutput,
        progress: Callable[[int, str], None],
    ) -> TranscriptOutput:
        aligner = self._load(progress)
        tokens: list[AlignedTranscriptToken] = []
        total = max(1, len(corrected.windows))
        for index, window in enumerate(corrected.windows, start=1):
            progress(15 + round(75 * (index - 1) / total), f"对齐窗口 {index}/{len(corrected.windows)}")
            if not window.text.strip():
                continue
            with self._cropped_wav(audio, window.input_start_ms, window.input_end_ms) as clip:
                results = cast(
                    list[list[ForcedAlignItem]],
                    aligner.align(audio=str(clip), text=window.text, language=window.language or "Chinese"),
                )
            if len(results) != 1:
                raise RuntimeError(f"ForcedAligner result count mismatch for window {window.window_index}")
            aligned_items = results[0]
            display_texts = self._restore_unaligned_text(window.text, [item.text for item in aligned_items])
            for item, display_text in zip(aligned_items, display_texts, strict=True):
                start_ms = window.input_start_ms + round(float(item.start_time) * 1000)
                end_ms = window.input_start_ms + round(float(item.end_time) * 1000)
                midpoint = (start_ms + end_ms) // 2
                if not (window.core_start_ms <= midpoint < window.core_end_ms):
                    continue
                speaker, status = self._speaker_for(start_ms, end_ms, diarization)
                tokens.append(
                    AlignedTranscriptToken(
                        token_index=len(tokens),
                        text=display_text,
                        start_ms=start_ms,
                        end_ms=max(start_ms, end_ms),
                        speaker_cluster_id=speaker.speaker_cluster_id if speaker else None,
                        speaker_label=speaker.speaker_label if speaker else None,
                        attribution_status=status,
                        source_window_index=window.window_index,
                        source_diarization_segment_id=speaker.id if speaker else None,
                    )
                )
        segments = self._segments_from_tokens(tokens, diarization)
        progress(95, "整理说话人转写结果")
        return TranscriptOutput(
            provider=corrected.asr_provider,
            model_name=corrected.asr_model_name,
            language=corrected.language,
            segments=segments,
            alignment_tokens=tokens,
            alignment_model_name=self._model_name,
        )

    @staticmethod
    def _speaker_for(
        start_ms: int, end_ms: int, segments: Sequence[DiarizationSegment]
    ) -> tuple[DiarizationSegment | None, Literal["matched", "ambiguous", "unmatched"]]:
        duration = max(1, end_ms - start_ms)
        ranked = sorted(((max(0, min(end_ms, item.end_ms) - max(start_ms, item.start_ms)), item) for item in segments), reverse=True, key=lambda item: item[0])
        if not ranked or ranked[0][0] == 0:
            return None, "unmatched"
        best_overlap, best = ranked[0]
        second_overlap = ranked[1][0] if len(ranked) > 1 else 0
        if best_overlap / duration < 0.6 or (second_overlap > 0 and best_overlap - second_overlap < 100):
            return None, "ambiguous"
        return best, "matched"

    @staticmethod
    def _restore_unaligned_text(original: str, aligned_texts: Sequence[str]) -> list[str]:
        """Attach punctuation and whitespace omitted by ForcedAligner to nearby timed tokens."""
        if not aligned_texts:
            return []

        positions: list[tuple[int, int]] = []
        cursor = 0
        lowered = original.lower()
        for aligned_text in aligned_texts:
            position = lowered.find(aligned_text.lower(), cursor)
            if position < 0:
                return list(aligned_texts)
            end = position + len(aligned_text)
            positions.append((position, end))
            cursor = end

        restored = [original[start:end] for start, end in positions]
        cursor = 0
        for index, (start, end) in enumerate(positions):
            skipped = original[cursor:start]
            if index == 0:
                restored[index] = skipped + restored[index]
            else:
                restored[index - 1] += skipped
            cursor = end
        if cursor < len(original):
            restored[-1] += original[cursor:]
        return restored

    @staticmethod
    def _segments_from_tokens(tokens: Sequence[AlignedTranscriptToken], diarization: Sequence[DiarizationSegment]) -> list[TranscriptSegment]:
        tokens_by_segment_id: dict[str, list[AlignedTranscriptToken]] = {}
        for token in tokens:
            if token.attribution_status != "matched" or token.source_diarization_segment_id is None:
                continue
            tokens_by_segment_id.setdefault(token.source_diarization_segment_id, []).append(token)

        output: list[TranscriptSegment] = []
        for segment in sorted(diarization, key=lambda item: (item.start_ms, item.end_ms)):
            segment_tokens = tokens_by_segment_id.get(segment.id)
            if not segment_tokens:
                continue
            output.append(
                TranscriptSegment(
                    source_diarization_segment_id=segment.id,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    text="".join(token.text for token in segment_tokens),
                    speaker_cluster_id=segment.speaker_cluster_id,
                    speaker_label=segment.speaker_label,
                )
            )
        return output

    def _load(self, progress: Callable[[int, str], None]) -> Any:
        if self._aligner is not None:
            return self._aligner
        progress(5, f"加载 ForcedAligner {self._model_name}")
        torch = cast(Any, import_module("torch"))
        qwen_asr = cast(Any, import_module("qwen_asr"))
        if bool(torch.cuda.is_available()):
            device_map, dtype = "cuda:0", torch.bfloat16
        elif bool(torch.backends.mps.is_available()):
            device_map, dtype = "mps", torch.float16
        else:
            device_map, dtype = "cpu", torch.float32
        model_path = resolve_local_snapshot(self._model_name, self._model_cache_root)
        self._aligner = qwen_asr.Qwen3ForcedAligner.from_pretrained(str(model_path), dtype=dtype, device_map=device_map, local_files_only=True)
        progress(15, "ForcedAligner 加载完成")
        return self._aligner

    def _release(self) -> None:
        self._aligner = None
        gc.collect()
        try:
            torch = import_module("torch")
            if bool(torch.cuda.is_available()):
                torch.cuda.empty_cache()
            if bool(torch.backends.mps.is_available()):
                torch.mps.empty_cache()
        except ImportError, RuntimeError, AttributeError:
            pass

    def _audio_path(self, artifact: ArtifactRef) -> Path:
        descriptor = self._artifact_store.read_json(artifact)
        path = descriptor.get("storage_path")
        if not isinstance(path, str):
            raise ValueError("ASR preprocessed audio artifact is missing storage_path")
        resolved = (self._storage_root / path).resolve()
        if self._storage_root not in resolved.parents or not resolved.is_file():
            raise FileNotFoundError(f"ASR preprocessed audio does not exist: {path}")
        return resolved

    @contextlib.contextmanager
    def _cropped_wav(self, source: Path, start_ms: int, end_ms: int) -> Generator[Path]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
            output = Path(temporary.name)
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(source),
                    "-ss",
                    str(start_ms / 1000),
                    "-to",
                    str(end_ms / 1000),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(output),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
            )
            yield output
        finally:
            output.unlink(missing_ok=True)
