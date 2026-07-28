from l2_core.audio_processing.stages.recording_models import DiarizationSegment
from l2_core.audio_processing.stages.transcribe_qwen_asr.engine import build_continuous_speech_windows


def _segment(cluster: str, start_ms: int, end_ms: int) -> DiarizationSegment:
    return DiarizationSegment(
        id=f"{cluster}:{start_ms}:{end_ms}",
        start_ms=start_ms,
        end_ms=end_ms,
        speaker_cluster_id=cluster,
        speaker_label=f"Speaker {cluster}",
    )


def test_continuous_windows_do_not_split_on_speaker_changes() -> None:
    source = [_segment("A", 0, 15_000), _segment("B", 15_000, 31_000), _segment("A", 31_000, 45_000)]

    result = build_continuous_speech_windows(source, target_duration_ms=30_000, max_duration_ms=80_000, overlap_ms=500)

    assert [(item.core_start_ms, item.core_end_ms) for item in result] == [(0, 31_000), (31_000, 45_000)]
    assert result[0].diarization_segment_ids == ["A:0:15000", "B:15000:31000"]
    assert (result[1].input_start_ms, result[1].input_end_ms) == (30_500, 45_500)
