from l2_core.audio_processing.stages.diarize_pyannote import PyannoteDiarizeStage
from l2_core.audio_processing.stages.recording_models import DiarizationSegment


def _segment(speaker: str, start_ms: int, end_ms: int) -> DiarizationSegment:
    return DiarizationSegment(
        id=f"{speaker}:{start_ms}:{end_ms}",
        start_ms=start_ms,
        end_ms=end_ms,
        speaker_cluster_id=speaker,
        speaker_label=f"Speaker {speaker}",
    )


def test_absorbs_short_segment_between_matching_speakers() -> None:
    segments = [
        _segment("A", 0, 10_000),
        _segment("B", 10_000, 11_500),
        _segment("A", 11_500, 20_000),
    ]

    merged, absorbed_count = PyannoteDiarizeStage._smooth_segments(segments, 2_000, 3_000, 80_000)

    assert absorbed_count == 1
    assert [(item.speaker_cluster_id, item.start_ms, item.end_ms) for item in merged] == [("A", 0, 20_000)]


def test_absorbs_alternating_short_segments_until_stable() -> None:
    segments = [
        _segment("A", 0, 10_000),
        _segment("B", 10_000, 10_500),
        _segment("A", 10_500, 11_000),
        _segment("B", 11_000, 11_500),
        _segment("A", 11_500, 20_000),
    ]

    merged, absorbed_count = PyannoteDiarizeStage._smooth_segments(segments, 2_000, 3_000, 80_000)

    assert absorbed_count == 2
    assert [(item.speaker_cluster_id, item.start_ms, item.end_ms) for item in merged] == [("A", 0, 20_000)]


def test_merges_short_same_speaker_run_before_sandwich_absorption() -> None:
    segments = [
        _segment("A", 0, 10_000),
        _segment("B", 10_000, 10_700),
        _segment("B", 10_700, 11_500),
        _segment("A", 11_500, 20_000),
    ]

    merged, absorbed_count = PyannoteDiarizeStage._smooth_segments(segments, 2_000, 3_000, 80_000)

    assert absorbed_count == 1
    assert [(item.speaker_cluster_id, item.start_ms, item.end_ms) for item in merged] == [("A", 0, 20_000)]


def test_preserves_sandwiched_segment_longer_than_threshold() -> None:
    segments = [
        _segment("A", 0, 10_000),
        _segment("B", 10_000, 12_001),
        _segment("A", 12_001, 20_000),
    ]

    smoothed, absorbed_count = PyannoteDiarizeStage._absorb_short_sandwiched_segments(segments, 2_000, 3_000)

    assert absorbed_count == 0
    assert [item.speaker_cluster_id for item in smoothed] == ["A", "B", "A"]


def test_preserves_short_segment_when_neighboring_speakers_differ() -> None:
    segments = [
        _segment("A", 0, 10_000),
        _segment("B", 10_000, 11_000),
        _segment("C", 11_000, 20_000),
    ]

    smoothed, absorbed_count = PyannoteDiarizeStage._absorb_short_sandwiched_segments(segments, 2_000, 3_000)

    assert absorbed_count == 0
    assert [item.speaker_cluster_id for item in smoothed] == ["A", "B", "C"]


def test_preserves_short_sandwiched_segment_when_absorb_gap_exceeds_limit() -> None:
    segments = [
        _segment("A", 0, 10_000),
        _segment("B", 12_001, 13_001),
        _segment("A", 13_001, 20_000),
    ]

    smoothed, absorbed_count = PyannoteDiarizeStage._smooth_segments(
        segments,
        short_segment_max_duration_ms=2_000,
        merge_max_gap_ms=86_400_000,
        merge_max_duration_ms=80_000,
        short_segment_max_gap_ms=2_000,
    )

    assert absorbed_count == 0
    assert [item.speaker_cluster_id for item in smoothed] == ["A", "B", "A"]


def test_absorbs_short_sandwiched_segment_within_independent_gap_limit() -> None:
    segments = [
        _segment("A", 0, 10_000),
        _segment("B", 12_000, 13_000),
        _segment("A", 15_000, 20_000),
    ]

    smoothed, absorbed_count = PyannoteDiarizeStage._smooth_segments(
        segments,
        short_segment_max_duration_ms=2_000,
        merge_max_gap_ms=86_400_000,
        merge_max_duration_ms=80_000,
        short_segment_max_gap_ms=2_000,
    )

    assert absorbed_count == 1
    assert [(item.speaker_cluster_id, item.start_ms, item.end_ms) for item in smoothed] == [("A", 0, 20_000)]
