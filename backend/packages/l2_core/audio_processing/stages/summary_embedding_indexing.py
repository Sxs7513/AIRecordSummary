from __future__ import annotations

import asyncio
import hashlib

from sqlalchemy import Engine, text

from l1_foundation.pipeline.contracts import ArtifactPayload, RetryPolicy, StageContext, StageResult
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l1_foundation.worker import SyncWorkerClient
from l2_core.audio_processing.stages.build_search_chunks.token_counter import EmbeddingTokenCounter
from l2_core.audio_processing.stages.recording_models import (
    RecordingSummaryOutput,
    SummaryEmbeddingIndexingInput,
    SummaryEmbeddingIndexingOutput,
)
from l2_core.audio_processing.stages.summary.retrieval_text import build_summary_retrieval_text
from l2_core.audio_processing.worker_tasks import EmbeddingEncodeTaskResult, embedding_encode_command


class SummaryEmbeddingIndexer:
    def __init__(
        self,
        worker_client: SyncWorkerClient,
        token_counter: EmbeddingTokenCounter,
        dimensions: int,
        max_tokens: int = 512,
    ) -> None:
        self._worker_client = worker_client
        self._token_counter = token_counter
        self._dimensions = dimensions
        self._max_tokens = max_tokens

    def encode(self, summary_text: str, recording_title: str = "") -> SummaryEmbeddingIndexingOutput:
        retrieval_text = build_summary_retrieval_text(
            recording_title,
            summary_text,
            count_tokens=self._token_counter,
            max_tokens=self._max_tokens,
        )
        result = self._worker_client.execute(
            embedding_encode_command([retrieval_text]),
            result_type=EmbeddingEncodeTaskResult,
        )
        if result.dimensions != self._dimensions:
            raise ValueError(f"Summary embedding dimensions do not match configured {self._dimensions}")
        if result.provider != "sentence_transformers":
            raise ValueError("Summary embedding worker returned an unexpected provider")
        if len(result.vectors) != 1 or len(result.vectors[0]) != self._dimensions:
            raise ValueError("Summary embedding worker returned an invalid vector")
        return SummaryEmbeddingIndexingOutput(
            provider="sentence_transformers",
            model_name=result.model_name,
            dimensions=result.dimensions,
            retrieval_text=retrieval_text,
            content_hash=hashlib.sha256(retrieval_text.encode("utf-8")).hexdigest(),
            embedding=result.vectors[0],
        )


class SummaryEmbeddingIndexingStage:
    name = "summary_embedding_indexing"
    version = "1"
    retry_policy = RetryPolicy(initial_backoff_seconds=30)
    input_model = SummaryEmbeddingIndexingInput

    def __init__(self, artifact_store: ArtifactStore, indexer: SummaryEmbeddingIndexer, engine: Engine | None = None) -> None:
        self._artifact_store = artifact_store
        self._indexer = indexer
        self._engine = engine

    async def try_restore(
        self,
        context: StageContext,
        _input_payload: SummaryEmbeddingIndexingInput,
    ) -> StageResult[SummaryEmbeddingIndexingOutput] | None:
        return self._artifact_store.try_restore_json(
            context.pipeline_run_id,
            context.stage_run_id,
            self.name,
            self.version,
            "summary.embedding_index",
            SummaryEmbeddingIndexingOutput,
        )

    async def run(
        self,
        context: StageContext,
        input_payload: SummaryEmbeddingIndexingInput,
    ) -> StageResult[SummaryEmbeddingIndexingOutput]:
        summary = RecordingSummaryOutput.model_validate(self._artifact_store.read_json(input_payload.summary))
        output = await asyncio.to_thread(
            self._indexer.encode,
            summary.summary_text,
            self._recording_title(context.subject_id),
        )
        context.report_progress(100, "录音总结向量化完成")
        return StageResult(
            output=output,
            artifacts=(ArtifactPayload(artifact_type="summary.embedding_index", data=output.model_dump(mode="json")),),
        )

    def _recording_title(self, recording_id: object) -> str:
        if self._engine is None:
            return ""
        with self._engine.connect() as connection:
            value = connection.execute(
                text("select title from recordings where id = :recording_id"),
                {"recording_id": recording_id},
            ).scalar_one_or_none()
        return str(value or "")
