"""The declared production graph for one uploaded recording."""

from typing import Literal

from l1_foundation.pipeline.contracts import ResourceQueue, RetryPolicy
from l1_foundation.pipeline.definitions.graph import ArtifactBinding, PipelineDefinition, PipelineNode

CPU_RETRY = RetryPolicy(initial_backoff_seconds=10)
GPU_RETRY = RetryPolicy(initial_backoff_seconds=30)

type AsrProvider = Literal["qwen_asr", "funasr_nano"]


def build_recording_processing(asr_provider: AsrProvider = "qwen_asr") -> PipelineDefinition:
    """Assemble the business pipeline with the configured ASR stage plugin."""
    asr_stage_name, asr_stage_version = {
        "qwen_asr": ("transcribe_qwen_asr", "5"),
        "funasr_nano": ("transcribe_funasr_nano", "3"),
    }[asr_provider]
    return PipelineDefinition(
        name="recording_processing",
        version="10",
        nodes=(
            PipelineNode(
                "normalize_audio",
                "normalize_audio",
                "1",
                ResourceQueue.CPU,
                CPU_RETRY,
                input_artifacts=(ArtifactBinding("source_audio", "audio.source"),),
                output_artifacts=("audio.normalized",),
            ),
            PipelineNode(
                "diarize_pyannote",
                "diarize_pyannote",
                "1",
                ResourceQueue.GPU_HIGH,
                GPU_RETRY,
                depends_on=("normalize_audio",),
                input_artifacts=(ArtifactBinding("audio", "audio.normalized", "normalize_audio"),),
                output_artifacts=("diarization.pyannote",),
            ),
            PipelineNode(
                "preprocess_asr_audio",
                "preprocess_asr_audio",
                "1",
                ResourceQueue.CPU,
                CPU_RETRY,
                depends_on=("normalize_audio",),
                input_artifacts=(ArtifactBinding("audio", "audio.normalized", "normalize_audio"),),
                output_artifacts=("audio.asr_preprocessed",),
            ),
            PipelineNode(
                asr_stage_name,
                asr_stage_name,
                asr_stage_version,
                ResourceQueue.GPU_HIGH,
                GPU_RETRY,
                depends_on=("preprocess_asr_audio", "diarize_pyannote"),
                input_artifacts=(
                    ArtifactBinding("audio", "audio.asr_preprocessed", "preprocess_asr_audio"),
                    ArtifactBinding("diarization", "diarization.pyannote", "diarize_pyannote"),
                ),
                output_artifacts=("transcript.asr_windows",),
            ),
            PipelineNode(
                "correct_asr_windows",
                "correct_asr_windows",
                "1",
                ResourceQueue.GPU_NORMAL,
                GPU_RETRY,
                depends_on=(asr_stage_name,),
                input_artifacts=(ArtifactBinding("transcript", "transcript.asr_windows", asr_stage_name),),
                output_artifacts=("transcript.corrected_windows",),
            ),
            PipelineNode(
                "align_transcript",
                "align_transcript",
                "1",
                ResourceQueue.GPU_HIGH,
                GPU_RETRY,
                depends_on=("correct_asr_windows", "preprocess_asr_audio", "diarize_pyannote"),
                input_artifacts=(
                    ArtifactBinding("transcript", "transcript.corrected_windows", "correct_asr_windows"),
                    ArtifactBinding("audio", "audio.asr_preprocessed", "preprocess_asr_audio"),
                    ArtifactBinding("diarization", "diarization.pyannote", "diarize_pyannote"),
                ),
                output_artifacts=("transcript.aligned",),
            ),
            PipelineNode(
                "build_utterances",
                "build_utterances",
                "4",
                ResourceQueue.CPU,
                CPU_RETRY,
                depends_on=("align_transcript",),
                input_artifacts=(ArtifactBinding("transcript", "transcript.aligned", "align_transcript"),),
                output_artifacts=("utterances.final",),
            ),
            PipelineNode(
                "build_search_chunks",
                "build_search_chunks",
                "2",
                ResourceQueue.GPU_NORMAL,
                GPU_RETRY,
                depends_on=("build_utterances",),
                input_artifacts=(ArtifactBinding("utterances", "utterances.final", "build_utterances"),),
                output_artifacts=("search.chunks",),
            ),
            PipelineNode(
                "embedding_indexing",
                "embedding_indexing",
                "1",
                ResourceQueue.GPU_NORMAL,
                GPU_RETRY,
                depends_on=("build_search_chunks",),
                required=False,
                input_artifacts=(ArtifactBinding("chunks", "search.chunks", "build_search_chunks"),),
                output_artifacts=("search.embedding_index",),
            ),
            PipelineNode(
                "generate_summary",
                "generate_summary",
                "2",
                ResourceQueue.GPU_NORMAL,
                GPU_RETRY,
                depends_on=("build_utterances",),
                required=False,
                input_artifacts=(ArtifactBinding("utterances", "utterances.final", "build_utterances"),),
                output_artifacts=("summary.recording",),
            ),
        ),
    )


# Backwards-compatible default for tests and callers that do not inject settings.
recording_processing = build_recording_processing()
