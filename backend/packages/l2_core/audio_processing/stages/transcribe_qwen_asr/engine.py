from __future__ import annotations

import contextlib
import logging
import math
import subprocess
import sys
import tempfile
import wave
from array import array
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path

from l2_core.audio_processing.stages.recording_models import DiarizationSegment

logger = logging.getLogger("audio_processing")


@dataclass(frozen=True, slots=True)
class QwenAsrWindowConfig:
    """Recording-domain preparation options, not model inference options."""

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


class QwenAsrWindowPreparer:
    """Builds model-ready audio windows from recording diarization output."""

    _LOW_VOLUME_ANALYSIS_WINDOW_MS = 2_000
    _LOW_VOLUME_FADE_MS = 30
    _LOW_VOLUME_OUTPUT_PEAK_LIMIT = 0.95

    def __init__(self, config: QwenAsrWindowConfig) -> None:
        self._config = config

    @property
    def display_name(self) -> str:
        return "Qwen ASR"

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
