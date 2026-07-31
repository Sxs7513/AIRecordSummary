from __future__ import annotations

from l1_foundation.llm import (
    LlmBatchWorkerHandler,
    LlmGenerateBatchInput,
    LlmGenerateBatchResult,
    LlmGenerateInput,
    LlmGenerateResult,
    LlmProvider,
    LlmWorkerHandler,
    create_language_model_from_settings,
)
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l1_foundation.settings import Settings
from l1_foundation.task_runtime.resources import ResourceQueue
from l2_core.audio_processing.stages.recording_models import DiarizationOutput
from l2_core.audio_processing.worker_tasks import (
    AlignmentInferenceBatchInput,
    AlignmentInferenceBatchResult,
    AsrInferenceBatchInput,
    AsrInferenceBatchResult,
    AudioDiarizeTaskInput,
    EmbeddingEncodeTaskInput,
    EmbeddingEncodeTaskResult,
)
from l2_core.rag.worker_tasks import RerankInput, RerankResult
from l3_app.compute_worker.audio_handlers import (
    AlignmentInferenceBatchHandler,
    EmbeddingEncodeHandler,
    PyannoteInferenceHandler,
    build_funasr_handler,
    build_qwen_asr_handler,
)
from l3_app.compute_worker.registry import ComputeOperationRegistry, ComputeOperationSpec
from l3_app.compute_worker.rerank_handler import RerankHandler


def build_compute_operation_registry(settings: Settings) -> ComputeOperationRegistry:
    registry = ComputeOperationRegistry()
    artifact_store = ArtifactStore(settings.resolved_local_storage_root)
    for provider in LlmProvider:
        handler = LlmWorkerHandler(
            provider,
            lambda context_size, model_profile, selected=provider: create_language_model_from_settings(
                settings,
                selected,
                local_context_size=context_size,
                local_model_profile=model_profile,
            ),
        )
        registry.register(
            ComputeOperationSpec(
                name=f"llm.generate.{provider.value}",
                version="1",
                resource_queue=ResourceQueue.GPU_NORMAL if provider == LlmProvider.LOCAL else ResourceQueue.IO,
                input_type=LlmGenerateInput,
                result_type=LlmGenerateResult,
                handler=handler,
                release=handler.release,
            )
        )
        batch_handler = LlmBatchWorkerHandler(provider, handler)
        registry.register(
            ComputeOperationSpec(
                name=f"llm.generate_batch.{provider.value}",
                version="1",
                resource_queue=ResourceQueue.GPU_NORMAL if provider == LlmProvider.LOCAL else ResourceQueue.IO,
                input_type=LlmGenerateBatchInput,
                result_type=LlmGenerateBatchResult,
                handler=batch_handler,
                release=batch_handler.release,
            )
        )
    diarize = PyannoteInferenceHandler(settings, artifact_store)
    qwen_asr = build_qwen_asr_handler(settings)
    funasr = build_funasr_handler(settings)
    align = AlignmentInferenceBatchHandler(settings, artifact_store)
    embedding = EmbeddingEncodeHandler(settings, artifact_store)
    rerank = RerankHandler(settings)
    registry.register(
        ComputeOperationSpec(
            "diarization.pyannote.infer",
            "1",
            ResourceQueue.GPU_HIGH,
            AudioDiarizeTaskInput,
            DiarizationOutput,
            diarize,
            diarize.release,
        )
    )
    registry.register(
        ComputeOperationSpec(
            "asr.qwen_asr.infer_batch",
            "1",
            ResourceQueue.GPU_HIGH,
            AsrInferenceBatchInput,
            AsrInferenceBatchResult,
            qwen_asr,
            qwen_asr.release,
        )
    )
    registry.register(
        ComputeOperationSpec(
            "asr.funasr_nano.infer_batch",
            "1",
            ResourceQueue.GPU_HIGH,
            AsrInferenceBatchInput,
            AsrInferenceBatchResult,
            funasr,
            funasr.release,
        )
    )
    registry.register(
        ComputeOperationSpec(
            "alignment.qwen.infer_batch",
            "1",
            ResourceQueue.GPU_HIGH,
            AlignmentInferenceBatchInput,
            AlignmentInferenceBatchResult,
            align,
            align.release,
        )
    )
    registry.register(
        ComputeOperationSpec(
            "embedding.encode",
            "1",
            ResourceQueue.GPU_NORMAL,
            EmbeddingEncodeTaskInput,
            EmbeddingEncodeTaskResult,
            embedding,
            embedding.release,
        )
    )
    registry.register(
        ComputeOperationSpec(
            "rerank.score",
            "1",
            ResourceQueue.GPU_NORMAL,
            RerankInput,
            RerankResult,
            rerank,
            rerank.release,
        )
    )
    return registry
