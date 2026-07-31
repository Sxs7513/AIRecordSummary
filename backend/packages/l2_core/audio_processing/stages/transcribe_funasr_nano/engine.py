from __future__ import annotations

import contextlib
import gc
import subprocess
import tempfile
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from l1_foundation.infrastructure.huggingface import resolve_local_snapshot
from l2_core.audio_processing.stages.recording_models import DiarizationSegment


@dataclass(frozen=True, slots=True)
class FunAsrNanoConfig:
    model_name: str
    language: str
    model_cache_root: Path
    hotwords: list[str]
    max_inference_batch_size: int = 1
    speech_window_target_duration_ms: int = 30_000
    speech_window_max_duration_ms: int = 80_000
    speech_window_overlap_ms: int = 500


@dataclass(frozen=True, slots=True)
class SpeechWindow:
    window_index: int
    input_start_ms: int
    input_end_ms: int
    core_start_ms: int
    core_end_ms: int
    diarization_segment_ids: list[str]


@dataclass(frozen=True, slots=True)
class FunAsrNanoResult:
    language: str | None
    windows: list[tuple[SpeechWindow, str]]


@dataclass(frozen=True, slots=True)
class FunAsrNanoInferenceResult:
    language: str | None
    texts: list[str]


def build_continuous_speech_windows(
    segments: Sequence[DiarizationSegment],
    target_duration_ms: int,
    max_duration_ms: int,
    overlap_ms: int,
) -> list[SpeechWindow]:
    ordered = sorted((item for item in segments if item.end_ms > item.start_ms), key=lambda item: (item.start_ms, item.end_ms))
    if not ordered:
        return []
    target = max(1, target_duration_ms)
    maximum = max(target, max_duration_ms)
    overlap = max(0, overlap_ms)
    windows: list[SpeechWindow] = []
    core_start = ordered[0].start_ms
    last_end = core_start
    included: list[DiarizationSegment] = []

    def emit(core_end: int) -> None:
        nonlocal included, core_start
        windows.append(
            SpeechWindow(
                window_index=len(windows),
                input_start_ms=max(0, core_start - overlap),
                input_end_ms=core_end + overlap,
                core_start_ms=core_start,
                core_end_ms=core_end,
                diarization_segment_ids=[item.id for item in included if item.end_ms > core_start and item.start_ms < core_end],
            )
        )
        core_start = core_end
        included = [item for item in included if item.end_ms > core_start]

    for segment in ordered:
        while segment.end_ms - core_start > maximum:
            emit(core_start + target)
        included.append(segment)
        last_end = max(last_end, segment.end_ms)
        if last_end - core_start >= target:
            emit(last_end)
    if last_end > core_start:
        emit(last_end)
    return windows


class FunAsrNanoEngine:
    """Standalone compatibility engine for Fun-ASR-Nano."""

    def __init__(self, config: FunAsrNanoConfig) -> None:
        self._config = config
        self._runtime: Any | None = None

    @property
    def model_name(self) -> str:
        return self._config.model_name

    def transcribe_continuous_windows(
        self,
        audio_path: Path,
        segments: Sequence[DiarizationSegment],
        progress: Callable[[int, str], None],
    ) -> FunAsrNanoResult:
        with tempfile.TemporaryDirectory() as directory, self.prepare_inference_batch(audio_path, segments, Path(directory)) as (windows, paths):
            result = self.infer_batch(paths, progress)
        return FunAsrNanoResult(result.language, list(zip(windows, result.texts, strict=True)))

    @contextlib.contextmanager
    def prepare_inference_batch(
        self,
        audio_path: Path,
        segments: Sequence[DiarizationSegment],
        work_dir: Path,
    ) -> Generator[tuple[list[SpeechWindow], list[Path]]]:
        windows = build_continuous_speech_windows(
            segments,
            self._config.speech_window_target_duration_ms,
            self._config.speech_window_max_duration_ms,
            self._config.speech_window_overlap_ms,
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        with contextlib.ExitStack() as stack:
            paths = [stack.enter_context(self._cropped_wav(audio_path, item.input_start_ms, item.input_end_ms, work_dir)) for item in windows]
            yield windows, paths

    def infer_batch(
        self,
        audio_paths: Sequence[Path],
        progress: Callable[[int, str], None],
        check_cancelled: Callable[[], None] | None = None,
        on_item_completed: Callable[[int, str], None] | None = None,
    ) -> FunAsrNanoInferenceResult:
        language = self._language_argument()
        if not audio_paths:
            return FunAsrNanoInferenceResult(language, [])
        runtime = self._load_model(progress)
        texts: list[str] = []
        total = len(audio_paths)
        for index, path in enumerate(audio_paths):
            if check_cancelled is not None:
                check_cancelled()
            progress(15 + round(80 * index / total), f"FunASR 推理 {index + 1}/{total}")
            result = cast(
                object,
                runtime.generate(
                    input=str(path),
                    cache={},
                    batch_size=1,
                    language=language,
                    hotwords=self._config.hotwords,
                    sentence_timestamp=False,
                    disable_pbar=True,
                ),
            )
            items = list(cast(Sequence[object], result)) if isinstance(result, list | tuple) else [result]
            if len(items) != 1:
                raise RuntimeError(f"FunASR item result count mismatch: expected 1, got {len(items)}")
            text = self._text(items[0])
            texts.append(text)
            if on_item_completed is not None:
                on_item_completed(index, text)
        progress(100, f"FunASR 推理 {total}/{total}")
        return FunAsrNanoInferenceResult(language, texts)

    def release(self) -> None:
        self._runtime = None
        gc.collect()
        torch = cast(Any, import_module("torch"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _load_model(self, progress: Callable[[int, str], None]) -> Any:
        if self._runtime is not None:
            return self._runtime
        progress(5, f"加载 Fun-ASR-Nano 模型 {self._config.model_name}")
        funasr = cast(Any, import_module("funasr"))
        torch = cast(Any, import_module("torch"))
        device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model_path = resolve_local_snapshot(self._config.model_name, self._config.model_cache_root)
        self._runtime = funasr.AutoModel(model=str(model_path), trust_remote_code=True, hub="hf", device=device, disable_update=True, disable_pbar=True)
        progress(15, f"Fun-ASR-Nano 模型加载完成，运行设备 {device}")
        return self._runtime

    def _language_argument(self) -> str | None:
        normalized = self._config.language.strip().lower()
        if not normalized or normalized == "auto":
            return None
        return {"zh": "中文", "zh-cn": "中文", "chinese": "中文", "en": "英文", "english": "英文"}.get(normalized, self._config.language)

    @staticmethod
    def _text(item: object) -> str:
        if isinstance(item, Mapping):
            value = cast(Mapping[str, object], item).get("text")
            return value.strip() if isinstance(value, str) else ""
        value = getattr(item, "text", "")
        return value.strip() if isinstance(value, str) else ""

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
