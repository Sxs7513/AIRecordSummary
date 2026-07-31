from __future__ import annotations

import contextlib
import gc
import logging
import math
import subprocess
import sys
import tempfile
import wave
from array import array
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from l1_foundation.infrastructure.huggingface import resolve_local_snapshot
from l2_core.audio_processing.stages.recording_models import DiarizationSegment

logger = logging.getLogger("audio_processing")


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
    max_inference_batch_size: int = 1
    num_beams: int = 2
    context: str = ""
    speech_window_target_duration_ms: int = 30_000
    speech_window_max_duration_ms: int = 80_000
    speech_window_overlap_ms: int = 500
    tempo: float = 1.0
    enhance_low_volume_segments: bool = True
    low_volume_rms_threshold: float = 0.01
    low_volume_peak_threshold: float = 0.08
    low_volume_max_gain_db: float = 9.0


@dataclass(frozen=True, slots=True)
class SpeechWindow:
    window_index: int
    input_start_ms: int
    input_end_ms: int
    core_start_ms: int
    core_end_ms: int
    diarization_segment_ids: list[str]


@dataclass(frozen=True, slots=True)
class LowVolumeAdjustment:
    start_ms: int
    end_ms: int
    gain_db: float


@dataclass(frozen=True, slots=True)
class QwenAsrWindowResult:
    language: str | None
    windows: list[tuple[SpeechWindow, str]]


@dataclass(frozen=True, slots=True)
class QwenAsrInferenceResult:
    language: str | None
    texts: list[str]


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

    _LOW_VOLUME_ANALYSIS_WINDOW_MS = 2_000
    _LOW_VOLUME_FADE_MS = 30
    _LOW_VOLUME_OUTPUT_PEAK_LIMIT = 0.95

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
        """Release the Worker-owned ASR model after inference or during shutdown."""
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
        with (
            tempfile.TemporaryDirectory() as directory,
            self.prepare_inference_batch(audio_path, diarization_segments, Path(directory)) as (windows, paths),
        ):
            result = self.infer_batch(paths, progress)
        return QwenAsrWindowResult(language=result.language, windows=list(zip(windows, result.texts, strict=True)))

    @contextlib.contextmanager
    def prepare_inference_batch(
        self,
        audio_path: Path,
        diarization_segments: Sequence[DiarizationSegment],
        work_dir: Path,
    ) -> Generator[tuple[list[SpeechWindow], list[Path]]]:
        """Prepare model-ready windows locally; no model is loaded here."""
        windows = build_continuous_speech_windows(
            diarization_segments,
            self._config.speech_window_target_duration_ms,
            self._config.speech_window_max_duration_ms,
            self._config.speech_window_overlap_ms,
        )
        if not windows:
            yield [], []
            return
        work_dir.mkdir(parents=True, exist_ok=True)
        low_volume_adjustments = self._build_low_volume_adjustments(audio_path, diarization_segments)
        with contextlib.ExitStack() as stack:
            paths = [
                stack.enter_context(
                    self._cropped_wav(
                        audio_path,
                        item.input_start_ms,
                        item.input_end_ms,
                        [
                            adjustment
                            for adjustment in low_volume_adjustments
                            if adjustment.start_ms < item.input_end_ms and adjustment.end_ms > item.input_start_ms
                        ],
                        work_dir,
                    )
                )
                for item in windows
            ]
            yield windows, paths

    def infer_batch(
        self,
        audio_paths: Sequence[Path],
        progress: Callable[[int, str], None],
        check_cancelled: Callable[[], None] | None = None,
        on_item_completed: Callable[[int, str], None] | None = None,
    ) -> QwenAsrInferenceResult:
        """Run a serialized request batch with an actual model batch size of one."""
        language = self._language_argument()
        if not audio_paths:
            return QwenAsrInferenceResult(language, [])
        model = self._load_model(progress, 5, 15)
        texts: list[str] = []
        total = len(audio_paths)
        for index, path in enumerate(audio_paths):
            if check_cancelled is not None:
                check_cancelled()
            progress(15 + round(80 * index / total), f"Qwen ASR 推理 {index + 1}/{total}")
            result = model.transcribe(
                audio=str(path),
                context=self._config.context,
                language=language,
                return_time_stamps=False,
            )
            text = self._extract_text(result)
            texts.append(text)
            if on_item_completed is not None:
                on_item_completed(index, text)
        progress(100, f"Qwen ASR 推理 {total}/{total}")
        return QwenAsrInferenceResult(language, texts)

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
        model = factory.from_pretrained(
            str(model_path),
            dtype=dtype,
            device_map=device_map,
            local_files_only=True,
            max_inference_batch_size=1,
            max_new_tokens=self._config.max_new_tokens,
        )
        self._configure_generation(model)
        self._model = model
        logger.info("Qwen ASR：模型加载完成，运行设备 %s", device_map)
        progress(progress_end, f"Qwen ASR 模型加载完成，运行设备 {device_map}")
        return self._model

    def _configure_generation(self, model: QwenAsrModel) -> None:
        backend_model = getattr(model, "model", None)
        thinker = getattr(backend_model, "thinker", None)
        generation_config = getattr(thinker, "generation_config", None)
        if generation_config is None:
            raise RuntimeError("Qwen ASR backend does not expose thinker.generation_config; num_beams cannot be applied")
        generation_config.num_beams = self._config.num_beams
        generation_config.do_sample = False
        generation_config.num_return_sequences = 1
        logger.info(
            "Qwen ASR：解码配置 num_beams=%d, do_sample=false, num_return_sequences=1",
            self._config.num_beams,
        )

    @contextlib.contextmanager
    def _cropped_wav(
        self,
        source: Path,
        start_ms: int,
        end_ms: int,
        low_volume_adjustments: Sequence[LowVolumeAdjustment] = (),
        output_dir: Path | None = None,
    ) -> Generator[Path]:
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=output_dir, delete=False) as temporary:
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
                    "-af",
                    f"atempo={self._config.tempo:.6g}",
                    "-c:a",
                    "pcm_s16le",
                    str(output_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
            )
            self._apply_low_volume_adjustments(output_path, start_ms, end_ms, low_volume_adjustments)
            yield output_path
        finally:
            output_path.unlink(missing_ok=True)

    def _build_low_volume_adjustments(self, source: Path, segments: Sequence[DiarizationSegment]) -> list[LowVolumeAdjustment]:
        if not self._config.enhance_low_volume_segments:
            logger.info("%s：低音量增强已关闭", self.display_name)
            return []
        ordered = sorted(
            (item for item in segments if item.end_ms > item.start_ms),
            key=lambda item: (item.start_ms, item.end_ms),
        )
        overlapping_ids: set[str] = set()
        for index, item in enumerate(ordered):
            for other in ordered[index + 1 :]:
                if other.start_ms >= item.end_ms:
                    break
                if other.end_ms > item.start_ms:
                    overlapping_ids.update((item.id, other.id))
        adjustments: list[LowVolumeAdjustment] = []
        measured_count = 0
        rms_below_count = 0
        peak_below_count = 0
        with wave.open(str(source), "rb") as reader:
            if reader.getnchannels() != 1 or reader.getsampwidth() != 2:
                logger.warning("%s：低音量增强跳过，输入并非单声道 PCM16 WAV", self.display_name)
                return []
            frame_rate = reader.getframerate()
            frame_count = reader.getnframes()
            for segment in ordered:
                if segment.id in overlapping_ids:
                    logger.debug("%s：低音量检测跳过重叠片段 %s", self.display_name, segment.id)
                    continue
                analysis_start_ms = segment.start_ms
                while analysis_start_ms < segment.end_ms:
                    analysis_end_ms = min(segment.end_ms, analysis_start_ms + self._LOW_VOLUME_ANALYSIS_WINDOW_MS)
                    start_frame = min(frame_count, max(0, round(analysis_start_ms * frame_rate / 1000)))
                    end_frame = min(frame_count, max(start_frame, round(analysis_end_ms * frame_rate / 1000)))
                    reader.setpos(start_frame)
                    samples = array("h")
                    samples.frombytes(reader.readframes(end_frame - start_frame))
                    if sys.byteorder == "big":
                        samples.byteswap()
                    if not samples:
                        logger.debug(
                            "%s：低音量检测跳过空区间 %s %d-%dms",
                            self.display_name,
                            segment.id,
                            analysis_start_ms,
                            analysis_end_ms,
                        )
                        analysis_start_ms = analysis_end_ms
                        continue
                    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples)) / 32768
                    peak = max(abs(sample) for sample in samples) / 32768
                    measured_count += 1
                    rms_below = rms < self._config.low_volume_rms_threshold
                    peak_below = peak < self._config.low_volume_peak_threshold
                    rms_below_count += rms_below
                    peak_below_count += peak_below
                    if rms <= 0 or not rms_below:
                        logger.debug(
                            "%s：低音量检测区间 %s %d-%dms，rms=%.6f，peak=%.6f，不增强（RMS未低于阈值）",
                            self.display_name,
                            segment.id,
                            analysis_start_ms,
                            analysis_end_ms,
                            rms,
                            peak,
                        )
                        analysis_start_ms = analysis_end_ms
                        continue
                    rms_gain_db = 20 * math.log10(self._config.low_volume_rms_threshold / rms)
                    peak_gain_db = 20 * math.log10(self._config.low_volume_peak_threshold / peak) if 0 < peak < self._config.low_volume_peak_threshold else 0.0
                    headroom_db = 20 * math.log10(self._LOW_VOLUME_OUTPUT_PEAK_LIMIT / peak) if peak > 0 else 0.0
                    gain_db = min(
                        self._config.low_volume_max_gain_db,
                        max(rms_gain_db, peak_gain_db),
                        max(0.0, headroom_db),
                    )
                    if gain_db > 0:
                        adjustments.append(
                            LowVolumeAdjustment(
                                start_ms=analysis_start_ms,
                                end_ms=analysis_end_ms,
                                gain_db=gain_db,
                            )
                        )
                        logger.debug(
                            "%s：低音量检测区间 %s %d-%dms，rms=%.6f，peak=%.6f，计划增强 %.2fdB",
                            self.display_name,
                            segment.id,
                            analysis_start_ms,
                            analysis_end_ms,
                            rms,
                            peak,
                            gain_db,
                        )
                    analysis_start_ms = analysis_end_ms
        logger.info(
            "%s：低音量增强扫描完成，片段=%d，%dms分析区间=%d，重叠片段跳过=%d，RMS低于阈值=%d，"
            "Peak低于阈值=%d，命中区间=%d，阈值(rms=%.4f, peak=%.4f)，最大增益=%.1fdB",
            self.display_name,
            len(ordered),
            self._LOW_VOLUME_ANALYSIS_WINDOW_MS,
            measured_count,
            len(overlapping_ids),
            rms_below_count,
            peak_below_count,
            len(adjustments),
            self._config.low_volume_rms_threshold,
            self._config.low_volume_peak_threshold,
            self._config.low_volume_max_gain_db,
        )
        return adjustments

    def _apply_low_volume_adjustments(
        self,
        output_path: Path,
        window_start_ms: int,
        window_end_ms: int,
        adjustments: Sequence[LowVolumeAdjustment],
    ) -> None:
        if not adjustments:
            return
        with wave.open(str(output_path), "rb") as reader:
            params = reader.getparams()
            if params.nchannels != 1 or params.sampwidth != 2:
                logger.warning("%s：低音量增强跳过，ASR 临时窗口并非单声道 PCM16 WAV", self.display_name)
                return
            samples = array("h")
            samples.frombytes(reader.readframes(params.nframes))
        if sys.byteorder == "big":
            samples.byteswap()
        frame_rate = params.framerate
        fade_frames = max(1, round(self._LOW_VOLUME_FADE_MS * frame_rate / 1000 / self._config.tempo))
        ordered_adjustments = sorted(adjustments, key=lambda item: (item.start_ms, item.end_ms))
        applied_adjustments: list[LowVolumeAdjustment] = []
        for adjustment_index, adjustment in enumerate(ordered_adjustments):
            clipped_start_ms = max(window_start_ms, adjustment.start_ms)
            clipped_end_ms = min(window_end_ms, adjustment.end_ms)
            start_frame = max(0, round((clipped_start_ms - window_start_ms) * frame_rate / 1000 / self._config.tempo))
            end_frame = min(len(samples), round((clipped_end_ms - window_start_ms) * frame_rate / 1000 / self._config.tempo))
            if end_frame <= start_frame:
                continue
            applied_adjustments.append(adjustment)
            gain = 10 ** (adjustment.gain_db / 20)
            previous = ordered_adjustments[adjustment_index - 1] if adjustment_index > 0 else None
            following = ordered_adjustments[adjustment_index + 1] if adjustment_index + 1 < len(ordered_adjustments) else None
            previous_gain = 10 ** (previous.gain_db / 20) if previous is not None and previous.end_ms == adjustment.start_ms else 1.0
            fade_in = adjustment.start_ms >= window_start_ms
            fade_out = adjustment.end_ms <= window_end_ms and (following is None or following.start_ms != adjustment.end_ms)
            for frame_index in range(start_frame, end_frame):
                factor = gain
                if fade_in and frame_index - start_frame < fade_frames:
                    fade_progress = (frame_index - start_frame + 1) / fade_frames
                    factor = previous_gain + (gain - previous_gain) * fade_progress
                if fade_out and end_frame - frame_index <= fade_frames:
                    fade_progress = (end_frame - frame_index) / fade_frames
                    factor = 1 + (gain - 1) * fade_progress
                samples[frame_index] = max(-32768, min(32767, round(samples[frame_index] * factor)))
        if sys.byteorder == "big":
            samples.byteswap()
        with wave.open(str(output_path), "wb") as writer:
            writer.setparams(params)
            writer.writeframes(samples.tobytes())
        if applied_adjustments:
            logger.info(
                "%s：ASR 临时窗口 %d-%dms 已实际应用低音量增强，区间=%d，增益=%.2f-%.2fdB",
                self.display_name,
                window_start_ms,
                window_end_ms,
                len(applied_adjustments),
                min(item.gain_db for item in applied_adjustments),
                max(item.gain_db for item in applied_adjustments),
            )

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
