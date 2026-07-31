from l2_core.audio_processing.stages.preprocess_asr_audio import ASR_AUDIO_OUTPUT_ARGS, PreprocessAsrAudioStage


def test_asr_audio_preprocessing_only_standardizes_output_format() -> None:
    assert PreprocessAsrAudioStage.version == "5"
    assert ASR_AUDIO_OUTPUT_ARGS == ("-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le")
    assert "-af" not in ASR_AUDIO_OUTPUT_ARGS
