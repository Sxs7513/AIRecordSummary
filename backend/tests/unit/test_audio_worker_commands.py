from l2_core.audio_processing.worker_tasks import (
    AlignmentInferenceItem,
    alignment_inference_batch_command,
    asr_inference_batch_command,
    audio_diarize_command,
    embedding_encode_command,
)


def test_non_streaming_audio_commands_do_not_wait_for_sse_subscriber() -> None:
    commands = (
        audio_diarize_command("recordings/audio.wav"),
        asr_inference_batch_command("qwen_asr", ["compute-inputs/window.wav"]),
        alignment_inference_batch_command(
            [
                AlignmentInferenceItem(
                    item_id="0",
                    audio_storage_path="compute-inputs/window.wav",
                    text="测试",
                    language="Chinese",
                )
            ]
        ),
        embedding_encode_command(["测试文本"]),
    )

    assert all(command.wait_for_subscriber is False for command in commands)
