from __future__ import annotations

import contextlib
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from audio_processing.stages.recording_models import DiarizationSegment
from audio_processing.stages.transcribe_qwen_asr.engine import QwenAsrConfig, QwenAsrEngine, QwenAsrModel


class FakeBatchModel:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def transcribe(self, *, audio: str | list[str], context: str, language: str | None, return_time_stamps: bool) -> object:
        assert isinstance(audio, list)
        self.batches.append(audio)
        return [SimpleNamespace(text=f"text-{Path(path).stem}") for path in audio]


def test_qwen_asr_uses_native_batches_and_preserves_segment_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    engine = QwenAsrEngine(
        QwenAsrConfig(
            model_name="test-model",
            language="Chinese",
            model_cache_root=tmp_path,
            max_inference_batch_size=2,
            speech_window_target_duration_ms=1_000,
            speech_window_max_duration_ms=1_000,
            speech_window_overlap_ms=0,
        )
    )
    model = FakeBatchModel()

    @contextlib.contextmanager
    def cropped_audio(_audio_path: Path, start_ms: int, end_ms: int) -> Generator[Path]:
        yield tmp_path / f"{start_ms}-{end_ms}.wav"

    monkeypatch.setattr(engine, "_load_model", lambda *_args: cast(QwenAsrModel, model))
    monkeypatch.setattr(engine, "_cropped_wav", cropped_audio)
    segments = [_segment("A", 0, 1_000), _segment("B", 1_000, 2_000), _segment("C", 2_000, 3_000)]

    result = engine.transcribe_continuous_windows(tmp_path / "source.wav", segments, cast(Any, lambda _percent, _message: None))

    assert [len(batch) for batch in model.batches] == [2, 1]
    assert [window.window_index for window, _text in result.windows] == [0, 1, 2]
    assert [text for _window, text in result.windows] == ["text-0-1000", "text-1000-2000", "text-2000-3000"]


def _segment(cluster: str, start_ms: int, end_ms: int) -> DiarizationSegment:
    return DiarizationSegment(
        id=f"{cluster}-{start_ms}",
        start_ms=start_ms,
        end_ms=end_ms,
        speaker_cluster_id=cluster,
        speaker_label=f"Speaker {cluster}",
    )
