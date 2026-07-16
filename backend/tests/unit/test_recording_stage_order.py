from __future__ import annotations

from application.recordings import RecordingService
from audio_processing.definition import build_recording_processing


def test_recording_processing_stages_follow_the_declared_graph_order() -> None:
    stages = [
        {"node_name": "generate_summary"},
        {"node_name": "normalize_audio"},
        {"node_name": "correct_asr_windows"},
        {"node_name": "align_transcript"},
        {"node_name": "transcribe_qwen_asr"},
    ]

    ordered = RecordingService.order_stage_rows("recording_processing", stages)

    assert [stage["node_name"] for stage in ordered] == [
        "normalize_audio",
        "transcribe_qwen_asr",
        "correct_asr_windows",
        "align_transcript",
        "generate_summary",
    ]


def test_historical_qwen_stage_keeps_asr_position_after_switching_to_funasr() -> None:
    stages = [
        {"node_name": "correct_asr_windows"},
        {"node_name": "transcribe_qwen_asr"},
        {"node_name": "normalize_audio"},
    ]

    ordered = RecordingService.order_stage_rows(
        "recording_processing",
        stages,
        build_recording_processing("funasr_nano"),
    )

    assert [stage["node_name"] for stage in ordered] == ["normalize_audio", "transcribe_qwen_asr", "correct_asr_windows"]
