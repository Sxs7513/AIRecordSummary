from audio_processing.stages.align_transcript import AlignTranscriptStage


def test_restore_unaligned_text_attaches_chinese_punctuation_to_timed_tokens() -> None:
    restored = AlignTranscriptStage._restore_unaligned_text("你好，世界！", ["你", "好", "世", "界"])

    assert restored == ["你", "好，", "世", "界！"]
    assert "".join(restored) == "你好，世界！"


def test_restore_unaligned_text_preserves_spaces_and_latin_case() -> None:
    restored = AlignTranscriptStage._restore_unaligned_text(" Hello, Qwen ASR.", ["Hello", "Qwen", "ASR"])

    assert "".join(restored) == " Hello, Qwen ASR."


def test_restore_unaligned_text_falls_back_when_aligned_text_does_not_match() -> None:
    restored = AlignTranscriptStage._restore_unaligned_text("原始文本。", ["归一化文本"])

    assert restored == ["归一化文本"]
