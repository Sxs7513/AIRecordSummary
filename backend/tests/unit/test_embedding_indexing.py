from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from l1_foundation.pipeline.contracts import ArtifactPayload, PipelineRunId, PipelineSubjectId, StageContext, StageRunId
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l1_foundation.worker import WorkerClient
from l2_core.audio_processing.stages.embedding_indexing import EmbeddingIndexingStage
from l2_core.audio_processing.stages.recording_models import EmbeddingIndexingInput, SearchChunk, SearchChunksOutput
from l2_core.audio_processing.worker_tasks import EmbeddingEncodeTaskResult


def _chunk(
    index: int,
    text: str,
    topic: str | None,
    *,
    terms: list[str] | None = None,
    search_context: str | None = None,
) -> SearchChunk:
    return SearchChunk(
        chunk_index=index,
        text=text,
        start_ms=index * 1_000,
        end_ms=(index + 1) * 1_000,
        speaker_labels=["Speaker A"],
        speaker_cluster_ids=["speaker-0"],
        source_utterance_indexes=[index],
        source_diarization_segment_ids=[f"A:{index * 1_000}:{(index + 1) * 1_000}"],
        topic=topic,
        terms=terms or [],
        search_context=search_context,
        topic_section_index=index if topic is not None else None,
        build_method="topic_boundary" if topic is not None else "deterministic_fallback",
    )


def test_embedding_indexing_adds_topic_to_model_input_without_changing_chunk_text(tmp_path: Path) -> None:
    storage = ArtifactStore(tmp_path)
    context = StageContext(PipelineSubjectId(uuid4()), PipelineRunId(uuid4()), StageRunId(uuid4()), 1)
    chunks = SearchChunksOutput(
        build_method="topic_boundary",
        chunks=[
            _chunk(0, "Speaker A: 讨论串扰问题。", "AWG 设计"),
            _chunk(
                1,
                "Speaker A: 你们大概做到多少？\nSpeaker B: 三千万左右。",
                "公司营收",
                terms=["营收", "收入", "销售额"],
                search_context="这段对话正在询问并回答公司目前的营收规模。",
            ),
            _chunk(2, "Speaker A: 普通正文。", None),
        ],
    )
    artifact = storage.write_json(
        context.subject_id,
        context.pipeline_run_id,
        context.stage_run_id,
        "build_search_chunks",
        ArtifactPayload(artifact_type="search.chunks", data=chunks.model_dump(mode="json")),
    )
    embedded_texts: list[str] = []

    class FakeWorkerClient:
        async def execute(self, command: Any, *, result_type: type[EmbeddingEncodeTaskResult], **_kwargs: object) -> EmbeddingEncodeTaskResult:
            embedded_texts.extend(command.input.texts)
            return result_type(provider="sentence_transformers", model_name="test/model", dimensions=2, vectors=[[0.1, 0.2] for _ in command.input.texts])

    stage = EmbeddingIndexingStage(
        storage,
        "test/model",
        tmp_path,
        dimensions=2,
        worker_client=cast(WorkerClient, FakeWorkerClient()),
    )

    result = asyncio.run(stage.run(context, EmbeddingIndexingInput(chunks=artifact)))

    assert embedded_texts == [
        "主题：AWG 设计\n正文：Speaker A: 讨论串扰问题。",
        "主题：公司营收\n标准术语：营收、收入、销售额\n"
        "语义上下文：这段对话正在询问并回答公司目前的营收规模。\n"
        "正文：Speaker A: 你们大概做到多少？\nSpeaker B: 三千万左右。",
        "Speaker A: 普通正文。",
    ]
    assert [chunk.text for chunk in result.output.chunks] == [
        "Speaker A: 讨论串扰问题。",
        "Speaker A: 你们大概做到多少？\nSpeaker B: 三千万左右。",
        "Speaker A: 普通正文。",
    ]


def test_embedding_device_prefers_cuda_then_mps(monkeypatch: Any, tmp_path: Path) -> None:
    stage = EmbeddingIndexingStage(ArtifactStore(tmp_path), "test/model", tmp_path, dimensions=2)

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return True

        class backends:
            class mps:
                @staticmethod
                def is_available() -> bool:
                    return True

    def import_torch(_name: str) -> type[FakeTorch]:
        return FakeTorch

    monkeypatch.setattr("l2_core.audio_processing.stages.embedding_indexing.import_module", import_torch)

    assert stage._resolve_device() == "cuda"  # pyright: ignore[reportPrivateUsage]


def test_embedding_device_uses_mps_when_cuda_is_unavailable(monkeypatch: Any, tmp_path: Path) -> None:
    stage = EmbeddingIndexingStage(ArtifactStore(tmp_path), "test/model", tmp_path, dimensions=2)

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class backends:
            class mps:
                @staticmethod
                def is_available() -> bool:
                    return True

    def import_torch(_name: str) -> type[FakeTorch]:
        return FakeTorch

    monkeypatch.setattr("l2_core.audio_processing.stages.embedding_indexing.import_module", import_torch)

    assert stage._resolve_device() == "mps"  # pyright: ignore[reportPrivateUsage]
