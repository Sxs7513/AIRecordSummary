from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

from sqlalchemy import Engine

from l1_foundation.settings import Settings
from l1_foundation.worker import ComputeCommand, SyncWorkerClient
from l2_core.rag.contracts import Evidence, EvidenceChunk, EvidenceRecording
from l2_core.rag.retrieval import RagRetriever
from l2_core.rag.worker_tasks import RerankInput, RerankResult, RerankScore


class FakeSettings:
    rag_rerank_enabled = True
    rag_rerank_candidate_limit = 20
    rag_rerank_max_total_tokens = 16_000
    rag_rerank_output_limit = 2


class FakeWorkerClient:
    def execute(self, command: ComputeCommand[RerankInput], *, result_type: type[RerankResult]) -> RerankResult:
        assert command.input.max_total_tokens == 16_000
        assert command.input.candidates[0].text.startswith(
            "主题：公司营收\n标准术语：营收、收入\n语义上下文：询问公司当前达到的营收规模\n正文："
        )
        return result_type(
            model_name="Qwen/Qwen3-Reranker-0.6B",
            scores=[
                RerankScore(candidate_id=command.input.candidates[1].candidate_id, score=0.9),
                RerankScore(candidate_id=command.input.candidates[0].candidate_id, score=0.2),
            ],
            input_tokens=80,
            skipped_candidates=1,
        )


def test_rerank_uses_expanded_evidence_text_and_rebuilds_indexes() -> None:
    recording_id = uuid4()
    chunk_ids = [uuid4(), uuid4(), uuid4()]
    evidence = [
        Evidence(
            index=index,
            recording=EvidenceRecording(id=recording_id, title="周会", file_name="meeting.mp3"),
            chunk=EvidenceChunk(
                id=chunk_id,
                text=f"说话人：扩展后的上下文 {index}",
                start_ms=index * 100,
                end_ms=index * 100 + 50,
                topic="公司营收" if index == 1 else None,
                terms=["营收", "收入"] if index == 1 else [],
                search_context="询问公司当前达到的营收规模" if index == 1 else None,
            ),
            score=1 / index,
            match_type="hybrid",
            url=f"/recordings/{recording_id}",
        )
        for index, chunk_id in enumerate(chunk_ids, start=1)
    ]
    retriever = RagRetriever(
        cast(Engine, cast(Any, object())),
        cast(Settings, cast(Any, FakeSettings())),
        cast(SyncWorkerClient, cast(Any, FakeWorkerClient())),
    )

    reranked, result = retriever.rerank_evidence("查询", evidence)

    assert [item.chunk.id for item in reranked] == [chunk_ids[1], chunk_ids[0]]
    assert [item.index for item in reranked] == [1, 2]
    assert [item.score for item in reranked] == [0.9, 0.2]
    assert result is not None and result.input_tokens == 80
