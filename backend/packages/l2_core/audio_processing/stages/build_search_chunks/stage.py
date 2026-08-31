from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Literal

from l1_foundation.llm import LlmProvider
from l1_foundation.pipeline.contracts import ArtifactPayload, RetryPolicy, StageContext, StageResult
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l1_foundation.worker import SyncWorkerClient
from l2_core.audio_processing.stages.build_search_chunks.builder import SearchChunkBuilder
from l2_core.audio_processing.stages.build_search_chunks.contracts import TopicSection
from l2_core.audio_processing.stages.build_search_chunks.detector import TopicBoundaryDetector
from l2_core.audio_processing.stages.recording_models import BuildSearchChunksInput, SearchChunksOutput, UtterancesOutput

logger = logging.getLogger("audio_processing")


class BuildSearchChunksStage:
    """Build bounded, topic-aware retrieval chunks from final utterances."""

    name = "build_search_chunks"
    version = "10"
    retry_policy = RetryPolicy(initial_backoff_seconds=30)
    input_model = BuildSearchChunksInput

    def __init__(
        self,
        artifact_store: ArtifactStore,
        token_counter: Callable[[str], int],
        max_tokens: int = 800,
        max_duration_ms: int = 180_000,
        max_utterances: int = 30,
        topic_detection_enabled: bool = False,
        worker_client: SyncWorkerClient | None = None,
        topic_provider: LlmProvider | None = None,
        topic_context_size: int = 8192,
    ) -> None:
        self._artifact_store = artifact_store
        self._builder = SearchChunkBuilder(token_counter, max_tokens, max_duration_ms, max_utterances)
        self._cache_config: dict[str, object] = {
            "max_tokens": max_tokens,
            "max_duration_ms": max_duration_ms,
            "max_utterances": max_utterances,
            "topic_detection_enabled": topic_detection_enabled,
            "topic_provider": topic_provider.value if topic_provider is not None else None,
            "topic_context_size": topic_context_size,
        }
        self._detector = (
            TopicBoundaryDetector(worker_client, topic_provider, topic_context_size)
            if topic_detection_enabled and worker_client is not None and topic_provider is not None
            else None
        )

    async def try_restore(self, context: StageContext, _input_payload: BuildSearchChunksInput) -> StageResult[SearchChunksOutput] | None:
        restored = self._artifact_store.try_restore_json(
            context.pipeline_run_id,
            context.stage_run_id,
            self.name,
            self.version,
            "search.chunks",
            SearchChunksOutput,
            input_fingerprint=context.input_fingerprint,
            allow_legacy_restore=context.allow_legacy_restore,
        )
        if restored is not None and self._detector is not None and restored.output.build_method == "deterministic_fallback":
            return None
        return restored

    def cache_config(self) -> dict[str, object]:
        return self._cache_config

    async def run(self, context: StageContext, input_payload: BuildSearchChunksInput) -> StageResult[SearchChunksOutput]:
        utterances = UtterancesOutput.model_validate(self._artifact_store.read_json(input_payload.utterances)).segments
        context.report_progress(5, "读取最终连续发言")
        sections: list[TopicSection] | None = None
        if self._detector is not None and utterances:
            context.report_progress(15, "识别连续主题区间")
            try:
                sections = await asyncio.to_thread(self._detector.detect, utterances)
                context.report_progress(70, f"识别到 {len(sections)} 个连续主题")
            except Exception as error:
                logger.warning("检索分块：主题识别失败，使用确定性分块：%s", error)
                logger.debug("检索分块主题识别异常详情", exc_info=True)
        build_method: Literal["topic_boundary", "deterministic_fallback"] = "topic_boundary" if sections is not None else "deterministic_fallback"
        context.report_progress(75, "构建有界检索分块")
        chunks = self._builder.build(utterances, sections)
        output = SearchChunksOutput(build_method=build_method, chunks=chunks)
        logger.info(
            "检索分块完成：连续发言=%d 主题区间=%d chunk=%d method=%s",
            len(utterances),
            len(sections or []),
            len(chunks),
            build_method,
        )
        context.report_progress(98, "检索分块构建完成")
        return StageResult(output=output, artifacts=(ArtifactPayload(artifact_type="search.chunks", data=output.model_dump(mode="json")),))
