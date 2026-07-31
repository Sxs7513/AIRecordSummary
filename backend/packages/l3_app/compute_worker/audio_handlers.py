from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l1_foundation.settings import Settings
from l1_foundation.worker import WorkerExecutionContext
from l2_core.audio_processing.stages.align_transcript import AlignTranscriptStage
from l2_core.audio_processing.stages.diarize_pyannote import PyannoteDiarizeStage
from l2_core.audio_processing.stages.embedding_indexing import EmbeddingIndexingStage
from l2_core.audio_processing.stages.recording_models import (
    DiarizationOutput,
)
from l2_core.audio_processing.stages.transcribe_funasr_nano.context import build_funasr_hotwords
from l2_core.audio_processing.stages.transcribe_funasr_nano.engine import FunAsrNanoConfig, FunAsrNanoEngine
from l2_core.audio_processing.stages.transcribe_qwen_asr.context import build_qwen_asr_context
from l2_core.audio_processing.stages.transcribe_qwen_asr.engine import QwenAsrConfig, QwenAsrEngine
from l2_core.audio_processing.worker_tasks import (
    AlignmentInferenceBatchInput,
    AlignmentInferenceBatchResult,
    AsrInferenceBatchInput,
    AsrInferenceBatchResult,
    AsrInferenceItemResult,
    AudioDiarizeTaskInput,
    EmbeddingEncodeTaskInput,
    EmbeddingEncodeTaskResult,
)


def _progress(context: WorkerExecutionContext) -> Callable[[int, str], None]:
    return lambda percent, message: context.report_progress(percent / 100, message)


class PyannoteInferenceHandler:
    def __init__(self, settings: Settings, artifact_store: ArtifactStore) -> None:
        self._storage_root = settings.resolved_local_storage_root
        self._diarizer = PyannoteDiarizeStage(
            self._storage_root,
            artifact_store,
            settings.pyannote_model,
            settings.pyannote_auth_token,
            settings.resolved_pyannote_cache_dir,
            settings.pyannote_use_local_config,
            settings.pyannote_segment_merge_max_gap_ms,
            settings.pyannote_segment_merge_max_duration_ms,
            settings.pyannote_short_segment_absorb_max_duration_ms,
            settings.pyannote_short_segment_absorb_max_gap_ms,
        )

    def __call__(self, value: AudioDiarizeTaskInput, context: WorkerExecutionContext) -> DiarizationOutput:
        return self._diarizer.infer(self._resolve(value.audio_storage_path), _progress(context))

    def release(self) -> None:
        self._diarizer.release()

    def _resolve(self, value: str) -> Path:
        path = (self._storage_root / value).resolve()
        if self._storage_root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Shared audio file does not exist: {value}")
        return path


class AsrInferenceBatchHandler:
    def __init__(self, engine: QwenAsrEngine | FunAsrNanoEngine, provider: str, storage_root: Path) -> None:
        self._engine = engine
        self._provider = provider
        self._storage_root = storage_root.resolve()

    def __call__(self, value: AsrInferenceBatchInput, context: WorkerExecutionContext) -> AsrInferenceBatchResult:
        paths = [self._resolve(item.audio_storage_path) for item in value.items]

        def item_completed(index: int, text: str) -> None:
            if text:
                context.emit_delta(text, value.items[index].item_id)

        result = self._engine.infer_batch(paths, _progress(context), context.raise_if_cancelled, item_completed)
        if len(result.texts) != len(value.items):
            raise RuntimeError(f"ASR inference result count mismatch: expected {len(value.items)}, got {len(result.texts)}")
        items: list[AsrInferenceItemResult] = []
        for item, text in zip(value.items, result.texts, strict=True):
            context.raise_if_cancelled()
            items.append(AsrInferenceItemResult(item_id=item.item_id, text=text, language=result.language))
        return AsrInferenceBatchResult(
            provider=self._provider,
            model_name=self._engine.model_name,
            items=items,
        )

    def release(self) -> None:
        self._engine.release()

    def _resolve(self, value: str) -> Path:
        path = (self._storage_root / value).resolve()
        if self._storage_root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Shared audio file does not exist: {value}")
        return path


class AlignmentInferenceBatchHandler:
    def __init__(self, settings: Settings, artifact_store: ArtifactStore) -> None:
        self._storage_root = settings.resolved_local_storage_root
        self._aligner = AlignTranscriptStage(
            self._storage_root,
            artifact_store,
            settings.transcript_alignment_model,
            settings.resolved_huggingface_hub_cache_dir,
        )

    def __call__(self, value: AlignmentInferenceBatchInput, context: WorkerExecutionContext) -> AlignmentInferenceBatchResult:
        return self._aligner.infer_batch(
            [(item, self._resolve(item.audio_storage_path)) for item in value.items],
            _progress(context),
            context.raise_if_cancelled,
        )

    def release(self) -> None:
        self._aligner.release()

    def _resolve(self, value: str) -> Path:
        path = (self._storage_root / value).resolve()
        if self._storage_root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Shared audio file does not exist: {value}")
        return path


class EmbeddingEncodeHandler:
    def __init__(self, settings: Settings, artifact_store: ArtifactStore) -> None:
        self._model_name = settings.embedding_model
        self._dimensions = settings.embedding_dimensions
        self._batch_size = settings.embedding_inference_batch_size
        self._encoder = EmbeddingIndexingStage(
            artifact_store,
            settings.embedding_model,
            settings.resolved_embedding_model_cache_dir,
            settings.embedding_dimensions,
        )

    def __call__(self, value: EmbeddingEncodeTaskInput, context: WorkerExecutionContext) -> EmbeddingEncodeTaskResult:
        context.report_progress(0.05, "加载 Embedding 模型")
        vectors: list[list[float]] = []
        total = len(value.texts)
        for offset in range(0, total, self._batch_size):
            context.raise_if_cancelled()
            batch = value.texts[offset : offset + self._batch_size]
            vectors.extend(self._encoder.encode(batch))
            context.report_progress(min(0.95, (offset + len(batch)) / total), "Embedding 编码中")
        if any(len(vector) != self._dimensions for vector in vectors):
            raise ValueError(f"Embedding dimension does not match configured {self._dimensions}")
        context.report_progress(1, "Embedding 编码完成")
        return EmbeddingEncodeTaskResult(
            provider="sentence_transformers",
            model_name=self._model_name,
            dimensions=self._dimensions,
            vectors=vectors,
        )

    def release(self) -> None:
        self._encoder.release()


def build_qwen_asr_handler(settings: Settings) -> AsrInferenceBatchHandler:
    engine = QwenAsrEngine(
        QwenAsrConfig(
            model_name=settings.qwen_asr_model,
            language=settings.qwen_asr_language,
            model_cache_root=settings.resolved_huggingface_hub_cache_dir,
            context=build_qwen_asr_context(
                settings.resolved_qwen_asr_context_config,
                settings.qwen_asr_max_context_items,
                settings.qwen_asr_context,
            ),
            max_inference_batch_size=1,
            num_beams=settings.qwen_asr_num_beams,
            speech_window_target_duration_ms=settings.asr_speech_window_target_duration_ms,
            speech_window_max_duration_ms=settings.asr_speech_window_max_duration_ms,
            speech_window_overlap_ms=settings.asr_speech_window_overlap_ms,
            tempo=settings.qwen_asr_tempo,
            enhance_low_volume_segments=settings.qwen_asr_enhance_low_volume_segments,
            low_volume_rms_threshold=settings.qwen_asr_low_volume_rms_threshold,
            low_volume_peak_threshold=settings.qwen_asr_low_volume_peak_threshold,
            low_volume_max_gain_db=settings.qwen_asr_low_volume_max_gain_db,
        )
    )
    return AsrInferenceBatchHandler(engine, "qwen_asr", settings.resolved_local_storage_root)


def build_funasr_handler(settings: Settings) -> AsrInferenceBatchHandler:
    hotwords = build_funasr_hotwords(settings.resolved_qwen_asr_context_config, settings.qwen_asr_max_context_items)
    engine = FunAsrNanoEngine(
        FunAsrNanoConfig(
            settings.funasr_nano_model,
            settings.qwen_asr_language,
            settings.resolved_funasr_nano_cache_dir,
            hotwords,
            1,
            settings.asr_speech_window_target_duration_ms,
            settings.asr_speech_window_max_duration_ms,
            settings.asr_speech_window_overlap_ms,
        )
    )
    return AsrInferenceBatchHandler(engine, "funasr_nano", settings.resolved_local_storage_root)
