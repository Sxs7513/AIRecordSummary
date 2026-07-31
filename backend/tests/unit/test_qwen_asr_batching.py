from __future__ import annotations

import contextlib
import logging
import subprocess
import wave
from array import array
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from l2_core.audio_processing.stages.recording_models import DiarizationSegment
from l2_core.audio_processing.stages.transcribe_qwen_asr.engine import (
    LowVolumeAdjustment,
    QwenAsrConfig,
    QwenAsrEngine,
    QwenAsrModel,
)


class FakeBatchModel:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def transcribe(self, *, audio: str | list[str], context: str, language: str | None, return_time_stamps: bool) -> object:
        assert isinstance(audio, str)
        self.batches.append([audio])
        return SimpleNamespace(text=f"text-{Path(audio).stem}")


def test_qwen_asr_configures_beam_search_on_inner_thinker(tmp_path: Path) -> None:
    engine = QwenAsrEngine(
        QwenAsrConfig(
            model_name="test-model",
            language="auto",
            model_cache_root=tmp_path,
            num_beams=2,
        )
    )
    generation_config = SimpleNamespace(num_beams=1, do_sample=True, num_return_sequences=3)
    model = SimpleNamespace(model=SimpleNamespace(thinker=SimpleNamespace(generation_config=generation_config)))

    engine._configure_generation(cast(QwenAsrModel, model))

    assert generation_config.num_beams == 2
    assert generation_config.do_sample is False
    assert generation_config.num_return_sequences == 1


def test_qwen_asr_accepts_a_request_batch_but_infers_one_item_at_a_time(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    engine = QwenAsrEngine(
        QwenAsrConfig(
            model_name="test-model",
            language="Chinese",
            model_cache_root=tmp_path,
            max_inference_batch_size=2,
            speech_window_target_duration_ms=1_000,
            speech_window_max_duration_ms=1_000,
            speech_window_overlap_ms=0,
            enhance_low_volume_segments=False,
        )
    )
    model = FakeBatchModel()

    @contextlib.contextmanager
    def cropped_audio(_audio_path: Path, start_ms: int, end_ms: int, _adjustments: object, _output_dir: Path) -> Generator[Path]:
        yield tmp_path / f"{start_ms}-{end_ms}.wav"

    monkeypatch.setattr(engine, "_load_model", lambda *_args: cast(QwenAsrModel, model))
    monkeypatch.setattr(engine, "_cropped_wav", cropped_audio)
    segments = [_segment("A", 0, 1_000), _segment("B", 1_000, 2_000), _segment("C", 2_000, 3_000)]

    result = engine.transcribe_continuous_windows(tmp_path / "source.wav", segments, cast(Any, lambda _percent, _message: None))

    assert [len(batch) for batch in model.batches] == [1, 1, 1]
    assert [window.window_index for window, _text in result.windows] == [0, 1, 2]
    assert [text for _window, text in result.windows] == ["text-0-1000", "text-1000-2000", "text-2000-3000"]


def test_qwen_asr_applies_pitch_preserving_tempo_to_temporary_window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    engine = QwenAsrEngine(
        QwenAsrConfig(
            model_name="test-model",
            language="auto",
            model_cache_root=tmp_path,
            tempo=0.9,
        )
    )
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)

    with engine._cropped_wav(tmp_path / "source.wav", 1_000, 2_000):
        pass

    assert len(commands) == 1
    assert commands[0][commands[0].index("-af") + 1] == "atempo=0.9"


def test_qwen_asr_enhances_only_non_overlapping_low_volume_segments(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_pcm16_wav(source, [100, -100] * 16_000 + [4_000, -4_000] * 16_000)
    engine = QwenAsrEngine(
        QwenAsrConfig(
            model_name="test-model",
            language="auto",
            model_cache_root=tmp_path,
            low_volume_rms_threshold=0.004,
            low_volume_peak_threshold=0.025,
        )
    )

    adjustments = engine._build_low_volume_adjustments(
        source,
        [_segment("A", 0, 4_000)],
    )

    assert len(adjustments) == 1
    assert adjustments[0].start_ms == 0
    assert adjustments[0].end_ms == 2_000
    assert adjustments[0].gain_db == 9


def test_qwen_asr_skips_low_volume_segments_during_overlapping_speech(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    source = tmp_path / "source.wav"
    _write_pcm16_wav(source, [100, -100] * 16_000)
    engine = QwenAsrEngine(
        QwenAsrConfig(model_name="test-model", language="auto", model_cache_root=tmp_path)
    )

    with caplog.at_level(logging.INFO):
        adjustments = engine._build_low_volume_adjustments(
            source,
            [_segment("A", 0, 1_000), _segment("B", 500, 1_500)],
        )

    assert adjustments == []
    assert "重叠片段跳过=2" in caplog.text
    assert "命中区间=0" in caplog.text


def test_qwen_asr_transient_peak_does_not_veto_a_quiet_analysis_window(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    samples = [100, -100] * 16_000
    samples[8_000] = 10_000
    _write_pcm16_wav(source, samples)
    engine = QwenAsrEngine(
        QwenAsrConfig(
            model_name="test-model",
            language="auto",
            model_cache_root=tmp_path,
            low_volume_rms_threshold=0.004,
            low_volume_peak_threshold=0.025,
        )
    )

    adjustments = engine._build_low_volume_adjustments(source, [_segment("A", 0, 2_000)])

    assert len(adjustments) == 1
    assert 0 < adjustments[0].gain_db <= 9


def test_qwen_asr_applies_low_volume_gain_without_changing_other_audio(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    output = tmp_path / "window.wav"
    original = [100, -100] * 8_000 + [4_000, -4_000] * 8_000
    _write_pcm16_wav(output, original)
    engine = QwenAsrEngine(
        QwenAsrConfig(model_name="test-model", language="auto", model_cache_root=tmp_path)
    )
    adjustment = engine._build_low_volume_adjustments(output, [_segment("A", 0, 1_000)])[0]

    with caplog.at_level(logging.INFO):
        engine._apply_low_volume_adjustments(output, 0, 2_000, [adjustment])

    actual = _read_pcm16_wav(output)
    assert max(abs(sample) for sample in actual[480:15_520]) > 100
    assert actual[16_000:] == original[16_000:]
    assert "ASR 临时窗口 0-2000ms 已实际应用低音量增强，区间=1" in caplog.text


def test_qwen_asr_smoothly_transitions_between_adjacent_gain_intervals(tmp_path: Path) -> None:
    output = tmp_path / "window.wav"
    _write_pcm16_wav(output, [100, -100] * 32_000)
    engine = QwenAsrEngine(
        QwenAsrConfig(model_name="test-model", language="auto", model_cache_root=tmp_path)
    )

    engine._apply_low_volume_adjustments(
        output,
        0,
        4_000,
        [
            LowVolumeAdjustment(start_ms=0, end_ms=2_000, gain_db=6),
            LowVolumeAdjustment(start_ms=2_000, end_ms=4_000, gain_db=3),
        ],
    )

    actual = _read_pcm16_wav(output)
    assert abs(actual[31_999]) > 150
    assert abs(actual[32_000]) > 150


def _segment(cluster: str, start_ms: int, end_ms: int) -> DiarizationSegment:
    return DiarizationSegment(
        id=f"{cluster}-{start_ms}",
        start_ms=start_ms,
        end_ms=end_ms,
        speaker_cluster_id=cluster,
        speaker_label=f"Speaker {cluster}",
    )


def _write_pcm16_wav(path: Path, samples: list[int]) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(array("h", samples).tobytes())


def _read_pcm16_wav(path: Path) -> list[int]:
    with wave.open(str(path), "rb") as reader:
        samples = array("h")
        samples.frombytes(reader.readframes(reader.getnframes()))
    return samples.tolist()
