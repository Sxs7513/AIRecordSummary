import asyncio

import pytest

from l1_foundation.pipeline.contracts import RetryPolicy
from l1_foundation.pipeline.definitions.graph import PipelineDefinition, PipelineNode
from l1_foundation.pipeline.example import build_example_registry, example_pipeline, run_example
from l1_foundation.pipeline.registry import StageRegistry
from l2_core.audio_processing.definition import build_recording_processing, recording_processing
from l2_core.audio_processing.stages.build_search_chunks import BuildSearchChunksStage
from l2_core.audio_processing.stages.noop import NoopStage


def test_recording_processing_uses_diarization_segments_for_qwen_asr() -> None:
    nodes = {node.name: node for node in recording_processing.nodes}

    assert recording_processing.version == "26"
    assert nodes["diarize_pyannote"].depends_on == ("normalize_audio",)
    assert nodes["diarize_pyannote"].stage_version == "2"
    assert nodes["preprocess_asr_audio"].depends_on == ("normalize_audio",)
    assert nodes["preprocess_asr_audio"].stage_version == "5"
    assert nodes["transcribe_qwen_asr"].depends_on == ("preprocess_asr_audio", "diarize_pyannote")
    assert nodes["transcribe_qwen_asr"].input_artifacts[1].artifact_type == "diarization.pyannote"
    assert nodes["transcribe_qwen_asr"].stage_version == "11"
    assert nodes["correct_asr_windows"].depends_on == ("transcribe_qwen_asr",)
    assert nodes["correct_asr_windows"].stage_version == "7"
    assert nodes["align_transcript"].depends_on == ("correct_asr_windows", "preprocess_asr_audio", "diarize_pyannote")
    assert nodes["align_transcript"].stage_version == "9"
    assert nodes["build_utterances"].depends_on == ("align_transcript",)
    assert nodes["build_utterances"].stage_version == "9"
    assert nodes["build_search_chunks"].stage_version == "10"
    assert nodes["build_search_chunks"].stage_version == BuildSearchChunksStage.version


def test_recording_processing_indexes_and_summarizes_in_parallel() -> None:
    nodes = {node.name: node for node in recording_processing.nodes}

    assert nodes["embedding_indexing"].depends_on == ("build_search_chunks",)
    assert nodes["embedding_indexing"].stage_version == "8"
    assert nodes["generate_summary"].depends_on == ("build_utterances",)
    assert nodes["generate_summary"].stage_version == "2"
    assert nodes["summary_embedding_indexing"].depends_on == ("generate_summary",)
    assert nodes["summary_embedding_indexing"].input_artifacts[0].artifact_type == "summary.recording"
    assert nodes["summary_embedding_indexing"].required is False
    assert nodes["build_search_chunks"].output_artifacts == ("search.chunks",)


def test_pipeline_definition_rejects_cycles() -> None:
    retry_policy = RetryPolicy(max_attempts=1)

    with pytest.raises(ValueError, match="dependency cycle"):
        PipelineDefinition(
            name="cycle",
            version="1",
            nodes=(
                PipelineNode("first", "noop", "1", retry_policy, ("second",)),
                PipelineNode("second", "noop", "1", retry_policy, ("first",)),
            ),
        )


def test_stage_registry_is_explicit_and_rejects_duplicates() -> None:
    registry = StageRegistry()
    stage = NoopStage()

    registry.register(stage)

    assert registry.get("noop", "1") is stage
    with pytest.raises(ValueError, match="already registered"):
        registry.register(stage)


def test_pipeline_example_has_registered_stage_plugins() -> None:
    registry = build_example_registry()

    for node in example_pipeline.nodes:
        assert registry.get(node.stage_name, node.stage_version).name == node.stage_name


def test_pipeline_example_passes_upstream_output_to_dependent_stage() -> None:
    output = asyncio.run(run_example())

    assert output["consumed"] is True
    assert output["upstream_outputs"]["prepare"]["prepared"] is True
