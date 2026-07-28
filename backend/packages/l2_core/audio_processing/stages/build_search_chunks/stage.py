from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Literal

from l1_foundation.pipeline.contracts import ArtifactPayload, ResourceQueue, RetryPolicy, StageContext, StageResult
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l2_core.audio_processing.stages.build_search_chunks.builder import SearchChunkBuilder
from l2_core.audio_processing.stages.build_search_chunks.contracts import TopicSection
from l2_core.audio_processing.stages.build_search_chunks.detector import TopicBoundaryDetector
from l2_core.audio_processing.stages.recording_models import BuildSearchChunksInput, SearchChunksOutput, UtterancesOutput

logger = logging.getLogger("audio_processing")


class BuildSearchChunksStage:
    """Build bounded, topic-aware retrieval chunks from final utterances."""

    name = "build_search_chunks"
    version = "2"
    resource_queue = ResourceQueue.GPU_NORMAL
    retry_policy = RetryPolicy(initial_backoff_seconds=30)
    input_model = BuildSearchChunksInput

    def __init__(
        self,
        artifact_store: ArtifactStore,
        max_chars: int = 1_200,
        max_duration_ms: int = 180_000,
        max_utterances: int = 30,
        topic_detection_enabled: bool = False,
        topic_model_path: Path | None = None,
        topic_model_context_size: int = 8192,
        topic_model_verbose: bool = False,
    ) -> None:
        self._artifact_store = artifact_store
        self._builder = SearchChunkBuilder(max_chars, max_duration_ms, max_utterances)
        self._detector = (
            TopicBoundaryDetector(topic_model_path, topic_model_context_size, topic_model_verbose)
            if topic_detection_enabled and topic_model_path is not None
            else None
        )

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
            finally:
                self._detector.release()
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
