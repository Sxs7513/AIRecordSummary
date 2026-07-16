from __future__ import annotations

import contextlib
import gc
import logging
import math
import subprocess
import tempfile
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from audio_processing.stages.recording_models import DiarizationSegment
from infrastructure.huggingface import resolve_local_snapshot

logger = logging.getLogger(__name__)


class TorchCuda(Protocol):
    def is_available(self) -> bool: ...

    def empty_cache(self) -> None: ...


class TorchMps(Protocol):
    def is_available(self) -> bool: ...


class TorchMpsMemory(Protocol):
    def empty_cache(self) -> None: ...


class TorchBackends(Protocol):
    mps: TorchMps


class TorchModule(Protocol):
    cuda: TorchCuda
    mps: TorchMpsMemory
    backends: TorchBackends
    bfloat16: object
    float16: object
    float32: object


class QwenAsrModel(Protocol):
    def transcribe(self, *, audio: str | list[str], context: str, language: str | None, return_time_stamps: bool) -> object: ...


class QwenAsrModelFactory(Protocol):
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs: object) -> QwenAsrModel: ...


class QwenAsrModule(Protocol):
    Qwen3ASRModel: QwenAsrModelFactory


@dataclass(frozen=True, slots=True)
class QwenAsrConfig:
    model_name: str
    language: str
    model_cache_root: Path
    max_new_tokens: int = 4096
    max_inference_batch_size: int = 4
    context: str = ""
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
class QwenAsrWindowResult:
    language: str | None
    windows: list[tuple[SpeechWindow, str]]


def build_continuous_speech_windows(
    segments: Sequence[DiarizationSegment], target_duration_ms: int, max_duration_ms: int, overlap_ms: int
) -> list[SpeechWindow]:
    """Build ASR windows from the speech timeline, never from speaker ownership."""
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
        ids = [item.id for item in included if item.end_ms > core_start and item.start_ms < core_end]
        windows.append(
            SpeechWindow(
                window_index=len(windows),
                input_start_ms=max(0, core_start - overlap),
                input_end_ms=core_end + overlap,
                core_start_ms=core_start,
                core_end_ms=core_end,
                diarization_segment_ids=ids,
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
            # A diarization boundary is only a preferred cut point after the target is reached.
            emit(last_end)
    if last_end > core_start:
        emit(last_end)
    return windows


class QwenAsrEngine:
    """Owns one lazily-loaded Qwen model inside the GPU worker process."""

    def __init__(self, config: QwenAsrConfig) -> None:
        self._config = config
        self._model: QwenAsrModel | None = None

    @property
    def display_name(self) -> str:
        return "Qwen ASR"

    @property
    def model_name(self) -> str:
        return self._config.model_name

    def release(self) -> None:
        """Release the ASR model after every diarized segment has been transcribed."""
        had_model = self._model is not None
        self._model = None
        gc.collect()
        try:
            torch = cast(TorchModule, import_module("torch"))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, RuntimeError, AttributeError) as error:
            logger.warning("%s：模型已释放，但设备缓存清理失败：%s", self.display_name, error)
        else:
            if not had_model:
                return
            logger.info("%s：模型和设备缓存已释放", self.display_name)

    def transcribe_continuous_windows(
        self, audio_path: Path, diarization_segments: Sequence[DiarizationSegment], progress: Callable[[int, str], None]
    ) -> QwenAsrWindowResult:
        """Transcribe continuous speech windows without cutting at speaker turns."""
        windows = build_continuous_speech_windows(
            diarization_segments,
            self._config.speech_window_target_duration_ms,
            self._config.speech_window_max_duration_ms,
            self._config.speech_window_overlap_ms,
        )
        language = self._language_argument()
        if not windows:
            logger.info("%s：没有可转写的连续语音窗口", self.display_name)
            return QwenAsrWindowResult(language=language, windows=[])
        model = self._load_model(progress, 5, 20)
        batch_size = max(1, self._config.max_inference_batch_size)
        output: list[tuple[SpeechWindow, str]] = []
        batch_count = math.ceil(len(windows) / batch_size)
        logger.info(
            "%s：开始连续语音窗口转写，窗口=%d，batch_size=%d，批次=%d，目标=%dms，最大=%dms，overlap=%dms",
            self.display_name,
            len(windows),
            batch_size,
            batch_count,
            self._config.speech_window_target_duration_ms,
            self._config.speech_window_max_duration_ms,
            self._config.speech_window_overlap_ms,
        )
        progress(20, f"开始转写 {len(windows)} 个连续语音窗口")
        for batch_index, offset in enumerate(range(0, len(windows), batch_size), start=1):
            batch = windows[offset : offset + batch_size]
            progress(20 + round(70 * offset / len(windows)), f"转写连续语音窗口 {offset + 1}-{offset + len(batch)}/{len(windows)}")
            logger.info(
                "%s：连续窗口转写批次 %d/%d，窗口 %d-%d/%d，输入范围=%s",
                self.display_name,
                batch_index,
                batch_count,
                offset + 1,
                offset + len(batch),
                len(windows),
                ", ".join(f"{item.input_start_ms}-{item.input_end_ms}ms" for item in batch),
            )
            with contextlib.ExitStack() as stack:
                paths = [stack.enter_context(self._cropped_wav(audio_path, item.input_start_ms, item.input_end_ms)) for item in batch]
                items = self._result_items(
                    model.transcribe(audio=[str(path) for path in paths], context=self._config.context, language=language, return_time_stamps=False)
                )
            if len(items) != len(batch):
                raise RuntimeError(f"Qwen ASR batch result count mismatch: expected {len(batch)}, got {len(items)}")
            texts = [self._extract_text(item) for item in items]
            logger.info(
                "%s：连续窗口转写批次 %d/%d 完成，有效=%d/%d，字符数=%d",
                self.display_name,
                batch_index,
                batch_count,
                sum(bool(text) for text in texts),
                len(texts),
                sum(len(text) for text in texts),
            )
            output.extend(zip(batch, texts, strict=True))
        progress(95, "整理连续语音窗口转写结果")
        logger.info("%s：连续语音窗口转写完成，有效窗口=%d/%d", self.display_name, sum(bool(text) for _, text in output), len(windows))
        return QwenAsrWindowResult(language=language, windows=output)

    def _load_model(self, progress: Callable[[int, str], None], progress_start: int = 5, progress_end: int = 15) -> QwenAsrModel:
        if self._model is not None:
            return self._model
        logger.info("Qwen ASR：加载模型 %s", self._config.model_name)
        progress(progress_start, f"加载 Qwen ASR 模型 {self._config.model_name}")
        model_path = resolve_local_snapshot(self._config.model_name, self._config.model_cache_root)
        try:
            torch = cast(TorchModule, import_module("torch"))
            qwen_module = cast(QwenAsrModule, import_module("qwen_asr"))
            factory = qwen_module.Qwen3ASRModel
        except (ImportError, AttributeError) as error:
            raise RuntimeError("Qwen ASR dependencies are missing; start the GPU worker with backend/.venv") from error
        device_map, dtype = self._device_options(torch)
        self._model = factory.from_pretrained(
            str(model_path),
            dtype=dtype,
            device_map=device_map,
            local_files_only=True,
            max_inference_batch_size=self._config.max_inference_batch_size,
            max_new_tokens=self._config.max_new_tokens,
        )
        logger.info("Qwen ASR：模型加载完成，运行设备 %s", device_map)
        progress(progress_end, f"Qwen ASR 模型加载完成，运行设备 {device_map}")
        return self._model

    @contextlib.contextmanager
    def _cropped_wav(self, source: Path, start_ms: int, end_ms: int) -> Generator[Path]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
            output_path = Path(temporary.name)
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
                    str(output_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
            )
            yield output_path
        finally:
            output_path.unlink(missing_ok=True)

    def _language_argument(self) -> str | None:
        normalized = self._config.language.strip().lower()
        if not normalized or normalized == "auto":
            return None
        return {"zh": "Chinese", "zh-cn": "Chinese", "chinese": "Chinese", "en": "English", "en-us": "English"}.get(normalized, self._config.language)

    @staticmethod
    def _device_options(torch: TorchModule) -> tuple[str, object]:
        if torch.cuda.is_available():
            return "cuda:0", torch.bfloat16
        if torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    @staticmethod
    def _extract_text(result: object) -> str:
        for item in QwenAsrEngine._result_items(result):
            text = QwenAsrEngine._get_attr(item, "text")
            if isinstance(text, str) and text.strip():
                return text.strip()
        return ""

    @staticmethod
    def _result_items(result: object) -> Sequence[object]:
        if isinstance(result, list | tuple):
            return cast(Sequence[object], result)
        return () if result is None else (result,)

    @staticmethod
    def _get_attr(value: object, key: str) -> object:
        if isinstance(value, Mapping):
            return cast(Mapping[str, object], value).get(key)
        return getattr(value, key, None)
