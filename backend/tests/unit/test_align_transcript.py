from typing import Literal

from l2_core.audio_processing.stages.align_transcript import AlignTranscriptStage
from l2_core.audio_processing.stages.recording_models import AlignedTranscriptToken, DiarizationSegment


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
