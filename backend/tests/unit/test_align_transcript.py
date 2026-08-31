from typing import Literal

from l2_core.audio_processing.stages.align_transcript import AlignTranscriptStage
from l2_core.audio_processing.stages.recording_models import AlignedTranscriptToken, CorrectedAsrWindowTranscript, DiarizationSegment


def _corrected_window(window_index: int, original_text: str) -> CorrectedAsrWindowTranscript:
    return CorrectedAsrWindowTranscript(
        window_index=window_index,
        input_start_ms=window_index * 1000,
        input_end_ms=(window_index + 1) * 1000,
        core_start_ms=window_index * 1000,
        core_end_ms=(window_index + 1) * 1000,
        language="Chinese",
        text=f"润色：{original_text}",
        original_text=original_text,
    )


def test_original_full_text_preserves_raw_windows_in_index_order() -> None:
    text = AlignTranscriptStage._original_full_text(  # pyright: ignore[reportPrivateUsage]
        [_corrected_window(1, " 第二段 "), _corrected_window(0, "第一段"), _corrected_window(2, "  ")]
    )

    assert text == "第一段\n第二段"


def test_restore_unaligned_text_attaches_chinese_punctuation_to_timed_tokens() -> None:
    restored = AlignTranscriptStage._restore_unaligned_text(  # pyright: ignore[reportPrivateUsage]
        "你好，世界！", ["你", "好", "世", "界"]
    )

    assert restored == ["你", "好，", "世", "界！"]
    assert "".join(restored) == "你好，世界！"


def test_restore_unaligned_text_preserves_spaces_and_latin_case() -> None:
    restored = AlignTranscriptStage._restore_unaligned_text(  # pyright: ignore[reportPrivateUsage]
        " Hello, Qwen ASR.", ["Hello", "Qwen", "ASR"]
    )

    assert "".join(restored) == " Hello, Qwen ASR."


def test_restore_unaligned_text_recovers_punctuation_when_aligner_merges_tokens() -> None:
    restored = AlignTranscriptStage._restore_unaligned_text(  # pyright: ignore[reportPrivateUsage]
        "你好，世界！",
        ["你好世界"],
    )

    assert restored == ["你好，世界！"]


def test_restore_unaligned_text_falls_back_when_aligned_text_does_not_match() -> None:
    restored = AlignTranscriptStage._restore_unaligned_text(  # pyright: ignore[reportPrivateUsage]
        "原始文本。", ["归一化文本"]
    )

    assert restored == ["归一化文本"]


def test_zero_duration_token_is_attributed_by_its_time_point() -> None:
    segment = DiarizationSegment(
        id="Speaker A:1000:2000",
        start_ms=1000,
        end_ms=2000,
        speaker_cluster_id="Speaker A",
        speaker_label="Speaker A",
    )

    speaker, status = AlignTranscriptStage._speaker_for(1500, 1500, [segment])  # pyright: ignore[reportPrivateUsage]

    assert speaker == segment
    assert status == "matched"


def test_unmatched_tokens_with_inferred_speaker_are_not_dropped_from_segments() -> None:
    segment = DiarizationSegment(
        id="Speaker A:1000:2000",
        start_ms=1000,
        end_ms=2000,
        speaker_cluster_id="Speaker A",
        speaker_label="Speaker A",
    )
    texts_and_statuses: list[tuple[str, Literal["matched", "ambiguous", "unmatched"]]] = [
        ("一", "unmatched"),
        ("次", "matched"),
        ("投", "matched"),
        ("片", "matched"),
        ("额", "ambiguous"),
        ("度", "matched"),
    ]
    tokens = [
        AlignedTranscriptToken(
            token_index=index,
            text=text,
            start_ms=1500 + index,
            end_ms=1500 + index,
            speaker_cluster_id=segment.speaker_cluster_id,
            speaker_label=segment.speaker_label,
            attribution_status=status,
            source_window_index=0,
            source_diarization_segment_id=segment.id,
        )
        for index, (text, status) in enumerate(texts_and_statuses)
    ]

    output = AlignTranscriptStage._segments_from_tokens(tokens, [segment])  # pyright: ignore[reportPrivateUsage]

    assert len(output) == 1
    assert output[0].text == "一次投片额度"


def test_segments_keep_only_original_tokens_attributed_to_the_same_speaker_turn() -> None:
    segment = DiarizationSegment(
        id="Speaker A:1000:2000",
        start_ms=1000,
        end_ms=2000,
        speaker_cluster_id="Speaker A",
        speaker_label="Speaker A",
    )
    tokens = [
        AlignedTranscriptToken(
            token_index=index,
            text=text,
            start_ms=1200 + index,
            end_ms=1200 + index,
            speaker_cluster_id=segment.speaker_cluster_id,
            speaker_label=segment.speaker_label,
            attribution_status="matched",
            source_window_index=window_index,
            source_diarization_segment_id=segment.id,
        )
        for index, (text, window_index) in enumerate([("润", 0), ("色", 0), ("文", 1)])
    ]

    original_tokens = [
        AlignedTranscriptToken(
            token_index=index,
            text=text,
            start_ms=1200 + index,
            end_ms=1200 + index,
            speaker_cluster_id=segment.speaker_cluster_id,
            speaker_label=segment.speaker_label,
            attribution_status="matched",
            source_window_index=0,
            source_diarization_segment_id=segment.id,
        )
        for index, text in enumerate("原始文本")
    ]

    output = AlignTranscriptStage._segments_from_tokens(tokens, [segment], original_tokens)  # pyright: ignore[reportPrivateUsage]

    assert output[0].text == "润色文"
    assert output[0].original_text == "原始文本"


def test_original_tokens_are_split_between_adjacent_speaker_turns() -> None:
    first = DiarizationSegment(
        id="Speaker A:0:1000", start_ms=0, end_ms=1000, speaker_cluster_id="Speaker A", speaker_label="Speaker A"
    )
    second = DiarizationSegment(
        id="Speaker A:1000:2000", start_ms=1000, end_ms=2000, speaker_cluster_id="Speaker A", speaker_label="Speaker A"
    )
    tokens = [
        AlignedTranscriptToken(
            token_index=index,
            text=text,
            start_ms=start_ms,
            end_ms=start_ms + 1,
            speaker_cluster_id="Speaker A",
            speaker_label="Speaker A",
            attribution_status="matched",
            source_window_index=window_index,
            source_diarization_segment_id=segment_id,
        )
        for index, (text, start_ms, window_index, segment_id) in enumerate(
            [("甲", 100, 0, first.id), ("乙", 1100, 0, second.id), ("丙", 1200, 1, second.id)]
        )
    ]

    original_tokens = [
        AlignedTranscriptToken(
            token_index=index,
            text=text,
            start_ms=start_ms,
            end_ms=start_ms + 1,
            speaker_cluster_id="Speaker A",
            speaker_label="Speaker A",
            attribution_status="matched",
            source_window_index=0,
            source_diarization_segment_id=segment_id,
        )
        for index, (text, start_ms, segment_id) in enumerate([("甲", 100, first.id), ("乙", 1100, second.id)])
    ]

    output = AlignTranscriptStage._segments_from_tokens(tokens, [first, second], original_tokens)  # pyright: ignore[reportPrivateUsage]

    assert [segment.original_text for segment in output] == ["甲", "乙"]
