import asyncio
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from l1_foundation.llm import (
    LlmGenerateResult,
    LlmProvider,
)
from l1_foundation.pipeline.contracts import ArtifactPayload, PipelineRunId, PipelineSubjectId, StageContext, StageRunId
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l1_foundation.worker import SyncWorkerClient
from l2_core.audio_processing.stages.build_search_chunks import BuildSearchChunksStage
from l2_core.audio_processing.stages.build_search_chunks.builder import SearchChunkBuilder
from l2_core.audio_processing.stages.build_search_chunks.contracts import TopicSection
from l2_core.audio_processing.stages.build_search_chunks.detector import TopicBoundaryDetector
from l2_core.audio_processing.stages.recording_models import BuildSearchChunksInput, Utterance, UtterancesOutput
from l2_core.rag.search_document import build_retrieval_text


class FakeWorkerClient:
    def __init__(self, text: str = '{"sections":[]}') -> None:
        self.text = text
        self.commands: list[Any] = []

    def execute(self, command: Any, *, result_type: type[LlmGenerateResult]) -> LlmGenerateResult:
        self.commands.append(command)
        return result_type(text=self.text, provider=LlmProvider.LOCAL, model="qwen-7b.gguf")


def test_build_search_chunks_keeps_source_utterance_provenance(tmp_path: Path) -> None:
    storage = ArtifactStore(tmp_path)
    context = StageContext(PipelineSubjectId(uuid4()), PipelineRunId(uuid4()), StageRunId(uuid4()), 1)
    utterances = UtterancesOutput(
        segments=[
            Utterance(
                utterance_index=0,
                start_ms=0,
                end_ms=1_000,
                text="第一句",
                speaker_cluster_id="A",
                speaker_label="Speaker A",
                source_diarization_segment_ids=["A:0:1000"],
            ),
            Utterance(
                utterance_index=1,
                start_ms=1_100,
                end_ms=2_000,
                text="第二句",
                speaker_cluster_id="B",
                speaker_label="Speaker B",
                source_diarization_segment_ids=["B:1100:2000"],
            ),
        ]
    )
    artifact = storage.write_json(
        context.subject_id,
        context.pipeline_run_id,
        context.stage_run_id,
        "build_utterances",
        ArtifactPayload(artifact_type="utterances.final", data=utterances.model_dump(mode="json")),
    )

    result = asyncio.run(BuildSearchChunksStage(storage, len).run(context, BuildSearchChunksInput(utterances=artifact)))

    assert result.output.chunks[0].source_utterance_indexes == [0, 1]
    assert result.output.chunks[0].source_diarization_segment_ids == ["A:0:1000", "B:1100:2000"]
    assert result.output.build_method == "deterministic_fallback"


def test_build_search_chunks_uses_continuous_topic_sections(tmp_path: Path) -> None:
    class FakeDetector:
        def detect(self, utterances: list[Utterance]) -> list[TopicSection]:
            assert len(utterances) == 2
            return [
                TopicSection(
                    start_utterance_index=0,
                    end_utterance_index=0,
                    topic="公司营收",
                    terms=["营收", "收入", "营收"],
                    search_context="这段对话正在询问公司的营收规模。",
                ),
                TopicSection(start_utterance_index=1, end_utterance_index=1, topic="话题二"),
            ]

        def release(self) -> None:
            return None

    storage = ArtifactStore(tmp_path)
    context = StageContext(PipelineSubjectId(uuid4()), PipelineRunId(uuid4()), StageRunId(uuid4()), 1)
    utterances = UtterancesOutput(
        segments=[
            Utterance(
                utterance_index=index,
                start_ms=index * 1000,
                end_ms=(index + 1) * 1000,
                text=f"第{index + 1}句",
                speaker_cluster_id="A",
                speaker_label="Speaker A",
                source_diarization_segment_ids=[f"A:{index * 1000}:{(index + 1) * 1000}"],
            )
            for index in range(2)
        ]
    )
    artifact = storage.write_json(
        context.subject_id,
        context.pipeline_run_id,
        context.stage_run_id,
        "build_utterances",
        ArtifactPayload(artifact_type="utterances.final", data=utterances.model_dump(mode="json")),
    )
    stage = BuildSearchChunksStage(
        storage,
        len,
        topic_detection_enabled=True,
        worker_client=cast(SyncWorkerClient, FakeWorkerClient()),
        topic_provider=LlmProvider.LOCAL,
    )
    stage._detector = cast(Any, FakeDetector())  # pyright: ignore[reportPrivateUsage]

    result = asyncio.run(stage.run(context, BuildSearchChunksInput(utterances=artifact)))

    assert result.output.build_method == "topic_boundary"
    assert [chunk.topic for chunk in result.output.chunks] == ["公司营收", "话题二"]
    assert result.output.chunks[0].terms == ["营收", "收入"]
    assert result.output.chunks[0].search_context == "这段对话正在询问公司的营收规模。"
    assert result.output.chunks[1].terms == []
    assert result.output.chunks[1].search_context is None
    assert all(chunk.build_method == "topic_boundary" for chunk in result.output.chunks)


def test_topic_boundary_detector_submits_serialized_worker_task(tmp_path: Path) -> None:
    client = FakeWorkerClient(
        '{"sections":[{"start_utterance_index":0,"end_utterance_index":0,"topic":"公司营收",'
        '"terms":["营收","收入"],"search_context":"询问公司目前做到的营收规模"}]}'
    )
    detector = TopicBoundaryDetector(cast(SyncWorkerClient, client), LlmProvider.LOCAL, 8192)
    utterance = Utterance(
        utterance_index=0,
        start_ms=0,
        end_ms=1000,
        text="测试",
        speaker_cluster_id="A",
        speaker_label="Speaker A",
        source_diarization_segment_ids=[],
    )

    section = detector.detect([utterance])[0]
    assert section.topic == "公司营收"
    assert section.terms == ["营收", "收入"]
    assert section.search_context == "询问公司目前做到的营收规模"
    assert client.commands[0].operation == "llm.generate.local"
    assert "不得编造原文没有的实体、数值、结论或事实" in client.commands[0].input.messages[0].content


def test_search_chunk_builder_counts_final_retrieval_text_and_splits_at_sentence_boundaries() -> None:
    utterance = Utterance(
        utterance_index=0,
        start_ms=0,
        end_ms=2_000,
        text="第一句。Second sentence.",
        speaker_cluster_id="A",
        speaker_label="Speaker A",
        source_diarization_segment_ids=["A:0:2000"],
    )
    first_retrieval_text = build_retrieval_text("Speaker A: 第一句。", "测试主题", ["术语"], "主题总结")
    builder = SearchChunkBuilder(len, len(first_retrieval_text), 180_000, 30)

    chunks = builder.build(
        [utterance],
        [TopicSection(start_utterance_index=0, end_utterance_index=0, topic="测试主题", terms=["术语"], search_context="主题总结")],
    )

    assert [chunk.text for chunk in chunks] == ["Speaker A: 第一句。", "Speaker A: Second sentence."]
    assert all(chunk.topic == "测试主题" for chunk in chunks)
    assert all(chunk.search_context == "主题总结" for chunk in chunks)
    assert all(chunk.source_utterance_indexes == [0] for chunk in chunks)


def test_search_chunk_builder_keeps_a_single_oversized_sentence_intact() -> None:
    utterance = Utterance(
        utterance_index=0,
        start_ms=0,
        end_ms=1_000,
        text="这是一句暂时不做进一步拆分的特别特别长的句子",
        speaker_cluster_id="A",
        speaker_label="Speaker A",
        source_diarization_segment_ids=[],
    )
    builder = SearchChunkBuilder(len, 5, 180_000, 30)

    chunks = builder.build([utterance], None)

    assert len(chunks) == 1
    assert chunks[0].text == f"Speaker A: {utterance.text}"


def test_search_chunk_builder_does_not_treat_decimal_points_as_sentence_boundaries() -> None:
    utterance = Utterance(
        utterance_index=0,
        start_ms=0,
        end_ms=1_000,
        text="营收是3.14亿元。下一句。",
        speaker_cluster_id="A",
        speaker_label="Speaker A",
        source_diarization_segment_ids=[],
    )
    first_sentence = "Speaker A: 营收是3.14亿元。"
    builder = SearchChunkBuilder(len, len(first_sentence), 180_000, 30)

    chunks = builder.build([utterance], None)

    assert [chunk.text for chunk in chunks] == [first_sentence, "Speaker A: 下一句。"]
