from __future__ import annotations

import contextlib
import gc
import subprocess
import tempfile
from collections.abc import Callable, Generator, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

from l1_foundation.files import FileStore
from l1_foundation.infrastructure.huggingface import resolve_local_snapshot
from l1_foundation.pipeline.contracts import ArtifactPayload, ArtifactRef, RetryPolicy, StageContext, StageResult
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l1_foundation.worker import WorkerClient
from l2_core.audio_processing.stages.recording_models import (
    AlignedTranscriptToken,
    AlignTranscriptInput,
    CorrectedAsrWindowTranscript,
    CorrectedAsrWindowTranscriptOutput,
    DiarizationOutput,
    DiarizationSegment,
    TranscriptOutput,
    TranscriptSegment,
)
from l2_core.audio_processing.worker_tasks import (
    AlignmentInferenceBatchResult,
    AlignmentInferenceItem,
    AlignmentInferenceItemResult,
    AlignmentTokenResult,
    alignment_inference_batch_command,
)


class ForcedAlignItem(Protocol):
    text: str
    start_time: float
    end_time: float


class AlignTranscriptStage:
    """Align final window text once, then attribute aligned tokens to pyannote turns."""

    name = "align_transcript"
    version = "2"
    retry_policy = RetryPolicy(initial_backoff_seconds=30)
    input_model = AlignTranscriptInput

    def __init__(
        self,
        file_store: FileStore,
        artifact_store: ArtifactStore,
        model_name: str,
        model_cache_root: Path,
        worker_client: WorkerClient | None = None,
    ) -> None:
        self._file_store = file_store
        self._artifact_store = artifact_store
        self._model_name = model_name
        self._model_cache_root = model_cache_root
        self._aligner: Any | None = None
        self._worker_client = worker_client

    async def try_restore(self, context: StageContext, _input_payload: AlignTranscriptInput) -> StageResult[TranscriptOutput] | None:
        return self._artifact_store.try_restore_json(
            context.pipeline_run_id,
            context.stage_run_id,
            self.name,
            self.version,
            "transcript.aligned",
            TranscriptOutput,
        )

    async def run(self, context: StageContext, input_payload: AlignTranscriptInput) -> StageResult[TranscriptOutput]:
        descriptor = self._artifact_store.read_json(input_payload.audio)
        audio_storage_path = descriptor.get("storage_path")
        if not isinstance(audio_storage_path, str):
            raise ValueError("ASR preprocessed audio artifact is missing storage_path")
        diarization = DiarizationOutput.model_validate(self._artifact_store.read_json(input_payload.diarization))
        corrected = CorrectedAsrWindowTranscriptOutput.model_validate(self._artifact_store.read_json(input_payload.transcript))
        if self._worker_client is None:
            raise RuntimeError("AlignTranscriptStage requires WorkerClient")
        audio_path = self._resolve_storage_path(audio_storage_path)
        source_windows = [window for window in corrected.windows if window.text.strip()]
        if source_windows:
            with tempfile.TemporaryDirectory(prefix="alignment-input-") as directory, contextlib.ExitStack() as stack:
                clips = [
                    stack.enter_context(self._cropped_wav(audio_path, window.input_start_ms, window.input_end_ms, Path(directory))) for window in source_windows
                ]
                batch_id = uuid4().hex
                input_keys = [self._file_store.put_file(clip, key=f"compute-inputs/{batch_id}/{index}.wav") for index, clip in enumerate(clips)]
                try:
                    inference = await self._worker_client.execute(
                        alignment_inference_batch_command(
                            [
                                AlignmentInferenceItem(
                                    item_id=str(window.window_index),
                                    audio_storage_path=key,
                                    text=window.text,
                                    language=window.language or "Chinese",
                                )
                                for window, key in zip(source_windows, input_keys, strict=True)
                            ]
                        ),
                        result_type=AlignmentInferenceBatchResult,
                        on_progress=lambda progress, message: context.report_progress(round(progress * 100), message or "转写对齐"),
                    )
                finally:
                    for key in input_keys:
                        self._file_store.delete_file(key)
        else:
            inference = AlignmentInferenceBatchResult(model_name=self._model_name, items=[])
        output = self._build_output(diarization.segments, corrected, source_windows, inference)
        return StageResult(
            output=output,
            artifacts=(ArtifactPayload(artifact_type="transcript.aligned", data=output.model_dump(mode="json")),),
        )

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

    def compute(
        self,
        audio: Path,
        diarization: Sequence[DiarizationSegment],
        corrected: CorrectedAsrWindowTranscriptOutput,
        progress: Callable[[int, str], None],
    ) -> TranscriptOutput:
        return self._align(audio, diarization, corrected, progress)

    def release(self) -> None:
        self._release()

    def infer_batch(
        self,
        items: Sequence[tuple[AlignmentInferenceItem, Path]],
        progress: Callable[[int, str], None],
        check_cancelled: Callable[[], None] | None = None,
    ) -> AlignmentInferenceBatchResult:
        """Run only ForcedAligner inference; orchestration and attribution stay in the Stage."""
        aligner = self._load(progress)
        results: list[AlignmentInferenceItemResult] = []
        total = len(items)
        for index, (item, path) in enumerate(items):
            if check_cancelled is not None:
                check_cancelled()
            progress(15 + round(80 * index / max(1, total)), f"ForcedAligner 推理 {index + 1}/{total}")
            aligned = cast(list[list[ForcedAlignItem]], aligner.align(audio=str(path), text=item.text, language=item.language))
            if len(aligned) != 1:
                raise RuntimeError(f"ForcedAligner result count mismatch for item {item.item_id}")
            results.append(
                AlignmentInferenceItemResult(
                    item_id=item.item_id,
                    tokens=[AlignmentTokenResult(text=token.text, start_time=token.start_time, end_time=token.end_time) for token in aligned[0]],
                )
            )
        progress(100, f"ForcedAligner 推理 {total}/{total}")
        return AlignmentInferenceBatchResult(model_name=self._model_name, items=results)

    def _build_output(
        self,
        diarization: Sequence[DiarizationSegment],
        corrected: CorrectedAsrWindowTranscriptOutput,
        windows: Sequence[CorrectedAsrWindowTranscript],
        inference: AlignmentInferenceBatchResult,
    ) -> TranscriptOutput:
        window_by_id = {str(window.window_index): window for window in corrected.windows}
        if len(inference.items) != len(windows):
            raise RuntimeError(f"Alignment inference result count mismatch: expected {len(windows)}, got {len(inference.items)}")
        tokens: list[AlignedTranscriptToken] = []
        for item in inference.items:
            window = window_by_id.get(item.item_id)
            if window is None:
                raise RuntimeError(f"Alignment inference returned unknown item {item.item_id}")
            display_texts = self._restore_unaligned_text(window.text, [token.text for token in item.tokens])
            for aligned, display_text in zip(item.tokens, display_texts, strict=True):
                start_ms = window.input_start_ms + round(aligned.start_time * 1000)
                end_ms = window.input_start_ms + round(aligned.end_time * 1000)
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
        return TranscriptOutput(
            provider=corrected.asr_provider,
            model_name=corrected.asr_model_name,
            language=corrected.language,
            segments=self._segments_from_tokens(tokens, diarization),
            alignment_tokens=tokens,
            alignment_model_name=inference.model_name,
        )

    @staticmethod
    def _speaker_for(
        start_ms: int, end_ms: int, segments: Sequence[DiarizationSegment]
    ) -> tuple[DiarizationSegment | None, Literal["matched", "ambiguous", "unmatched"]]:
        if not segments:
            return None, "unmatched"
        if end_ms <= start_ms:
            containing = [item for item in segments if item.start_ms <= start_ms < item.end_ms]
            if containing:
                best = min(containing, key=lambda item: (item.end_ms - item.start_ms, item.start_ms))
                return best, "matched" if len(containing) == 1 else "ambiguous"
            return AlignTranscriptStage._nearest_speaker(start_ms, segments), "unmatched"

        duration = max(1, end_ms - start_ms)
        ranked = sorted(((max(0, min(end_ms, item.end_ms) - max(start_ms, item.start_ms)), item) for item in segments), reverse=True, key=lambda item: item[0])
        if ranked[0][0] == 0:
            return AlignTranscriptStage._nearest_speaker((start_ms + end_ms) // 2, segments), "unmatched"
        best_overlap, best = ranked[0]
        second_overlap = ranked[1][0] if len(ranked) > 1 else 0
        if best_overlap / duration < 0.6 or (second_overlap > 0 and best_overlap - second_overlap < 100):
            return best, "ambiguous"
        return best, "matched"

    @staticmethod
    def _nearest_speaker(point_ms: int, segments: Sequence[DiarizationSegment]) -> DiarizationSegment:
        return min(
            segments,
            key=lambda item: (
                0 if item.start_ms <= point_ms < item.end_ms else min(abs(point_ms - item.start_ms), abs(point_ms - item.end_ms)),
                item.start_ms,
                item.end_ms,
            ),
        )

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
            if token.source_diarization_segment_id is None:
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
        return self._file_store.get_file_by_key(path)

    def _resolve_storage_path(self, uri: str) -> Path:
        return self._file_store.get_file_by_key(uri)

    @contextlib.contextmanager
    def _cropped_wav(self, source: Path, start_ms: int, end_ms: int, output_dir: Path | None = None) -> Generator[Path]:
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=output_dir, delete=False) as temporary:
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
