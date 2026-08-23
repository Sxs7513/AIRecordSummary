from __future__ import annotations

from typing import Any, cast

from l1_foundation.worker import SyncWorkerClient
from l2_core.audio_processing.stages.build_search_chunks.token_counter import EmbeddingTokenCounter
from l2_core.audio_processing.stages.summary_embedding_indexing import SummaryEmbeddingIndexer
from l2_core.audio_processing.worker_tasks import EmbeddingEncodeTaskResult


def test_summary_embedding_indexer_builds_one_bounded_retrieval_document() -> None:
    embedded_texts: list[str] = []

    class FakeWorkerClient:
        def execute(self, command: Any, *, result_type: type[EmbeddingEncodeTaskResult], **_kwargs: object) -> EmbeddingEncodeTaskResult:
            embedded_texts.extend(command.input.texts)
            return result_type(
                provider="sentence_transformers",
                model_name="test/model",
                dimensions=2,
                vectors=[[0.1, 0.2]],
            )

    def count_characters(text: str) -> int:
        return len(text)

    indexer = SummaryEmbeddingIndexer(
        cast(SyncWorkerClient, FakeWorkerClient()),
        cast(EmbeddingTokenCounter, count_characters),
        dimensions=2,
        max_tokens=120,
    )
    output = indexer.encode(
        "# 全局总结\n这是一次项目答辩汇报，介绍了当前进展。\n\n# 技术方案\n团队讲解了检索架构。",
        "项目汇报",
    )

    assert embedded_texts == [output.retrieval_text]
    assert "录音标题：项目汇报" in output.retrieval_text
    assert "项目答辩汇报" in output.retrieval_text
    assert len(output.retrieval_text) <= 120
    assert output.embedding == [0.1, 0.2]
    assert len(output.content_hash) == 64
