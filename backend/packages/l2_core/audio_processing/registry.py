from __future__ import annotations

from sqlalchemy import Engine

from l1_foundation.llm import LlmProvider
from l1_foundation.pipeline.registry import StageRegistry
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l1_foundation.settings import Settings
from l1_foundation.worker import SyncWorkerClient, WorkerClient
from l2_core.audio_processing.stages.align_transcript import AlignTranscriptStage
from l2_core.audio_processing.stages.build_search_chunks import BuildSearchChunksStage
from l2_core.audio_processing.stages.build_search_chunks.token_counter import EmbeddingTokenCounter
from l2_core.audio_processing.stages.build_utterances import BuildUtterancesStage
from l2_core.audio_processing.stages.correct_text import CorrectAsrWindowsStage, LocalTextCorrector
from l2_core.audio_processing.stages.diarize_pyannote import PyannoteDiarizeStage
from l2_core.audio_processing.stages.embedding_indexing import EmbeddingIndexingStage
from l2_core.audio_processing.stages.normalize_audio import NormalizeAudioStage
from l2_core.audio_processing.stages.preprocess_asr_audio import PreprocessAsrAudioStage
from l2_core.audio_processing.stages.summary.stage import GenerateSummaryStage
from l2_core.audio_processing.stages.summary_embedding_indexing import SummaryEmbeddingIndexer, SummaryEmbeddingIndexingStage
from l2_core.audio_processing.stages.transcribe_qwen_asr import QwenAsrTranscribeStage
from l2_core.generation.service import GenerationService


def build_recording_stage_registry(
    settings: Settings,
    artifact_store: ArtifactStore,
    worker_client: SyncWorkerClient,
    async_worker_client: WorkerClient,
    generation_service: GenerationService | None = None,
    summary_stage: GenerateSummaryStage | None = None,
    engine: Engine | None = None,
) -> StageRegistry:
    """Build every recording-owned stage; stages orchestrate atomic compute calls themselves."""
    registry = StageRegistry()
    file_store = artifact_store.file_store
    registry.register(NormalizeAudioStage(file_store, artifact_store))
    registry.register(
        PreprocessAsrAudioStage(
            file_store,
            artifact_store,
            settings.asr_preprocess_recording_enabled,
        )
    )
    registry.register(BuildUtterancesStage(artifact_store))
    registry.register(
        BuildSearchChunksStage(
            artifact_store,
            EmbeddingTokenCounter(settings.embedding_model, settings.resolved_embedding_model_cache_dir),
            settings.search_chunk_max_token,
            settings.search_chunk_max_duration_ms,
            settings.search_chunk_max_utterances,
            settings.search_chunk_topic_detection_enabled,
            worker_client if settings.search_chunk_topic_detection_enabled else None,
            LlmProvider(settings.topic_detection_provider) if settings.search_chunk_topic_detection_enabled else None,
            settings.llm_correction_context_size,
        )
    )
    registry.register(
        PyannoteDiarizeStage(
            file_store,
            artifact_store,
            settings.pyannote_model,
            settings.pyannote_auth_token,
            settings.resolved_pyannote_cache_dir,
            settings.pyannote_use_local_config,
            settings.pyannote_segment_merge_max_gap_ms,
            settings.pyannote_segment_merge_max_duration_ms,
            settings.pyannote_short_segment_absorb_max_duration_ms,
            settings.pyannote_short_segment_absorb_max_gap_ms,
            async_worker_client,
        )
    )
    registry.register(
        QwenAsrTranscribeStage(
            file_store,
            artifact_store,
            settings.qwen_asr_model,
            settings.asr_speech_window_target_duration_ms,
            settings.asr_speech_window_max_duration_ms,
            settings.asr_speech_window_overlap_ms,
            settings.qwen_asr_tempo,
            settings.qwen_asr_enhance_low_volume_segments,
            settings.qwen_asr_low_volume_rms_threshold,
            settings.qwen_asr_low_volume_peak_threshold,
            settings.qwen_asr_low_volume_max_gain_db,
            async_worker_client,
        )
    )
    registry.register(
        CorrectAsrWindowsStage(
            artifact_store,
            LocalTextCorrector(
                pycorrector_enabled=settings.transcription_correction_enabled,
                llm_enabled=settings.llm_correction_enabled,
                worker_client=worker_client if settings.llm_correction_enabled else None,
                llm_provider=LlmProvider(settings.llm_correction_provider) if settings.llm_correction_enabled else None,
                llm_context_size=settings.llm_correction_context_size,
                llm_model_name=_llm_model_name(settings, LlmProvider(settings.llm_correction_provider)) if settings.llm_correction_enabled else None,
                prompt_config_path=settings.resolved_whisper_initial_prompt_config,
                llm_batch_max_units=settings.text_correction_batch_max_units,
                llm_batch_max_chars=settings.text_correction_batch_max_chars,
                llm_max_output_tokens=settings.llm_correction_max_output_tokens,
                llm_context_units=settings.text_correction_context_units,
            ),
            "pycorrector_llm"
            if settings.llm_correction_enabled and settings.transcription_correction_enabled
            else "llm"
            if settings.llm_correction_enabled
            else "pycorrector"
            if settings.transcription_correction_enabled
            else "rules",
            _llm_model_name(settings, LlmProvider(settings.llm_correction_provider)) if settings.llm_correction_enabled else None,
            settings.asr_window_correction_max_edit_ratio,
        )
    )
    registry.register(
        AlignTranscriptStage(
            file_store,
            artifact_store,
            settings.transcript_alignment_model,
            async_worker_client,
        )
    )
    registry.register(
        EmbeddingIndexingStage(
            artifact_store,
            settings.embedding_model,
            settings.resolved_embedding_model_cache_dir,
            settings.embedding_dimensions,
            worker_client=async_worker_client,
        )
    )
    registry.register(summary_stage or build_recording_summary_stage(settings, artifact_store, worker_client, generation_service))
    registry.register(SummaryEmbeddingIndexingStage(artifact_store, build_summary_embedding_indexer(settings, worker_client), engine))
    return registry


def build_recording_summary_stage(
    settings: Settings,
    artifact_store: ArtifactStore,
    worker_client: SyncWorkerClient,
    generation_service: GenerationService | None = None,
) -> GenerateSummaryStage:
    """Build the shared summary implementation for both pipeline and manual regeneration."""
    return GenerateSummaryStage(
        artifact_store,
        worker_client,
        LlmProvider(settings.recording_summary_provider),
        _llm_model_name(settings, LlmProvider(settings.recording_summary_provider)),
        settings.recording_summary_context_size,
        settings.resolved_recording_summary_prompt_config,
        settings.recording_summary_max_tokens,
        settings.recording_summary_rolling_enabled,
        settings.recording_summary_rolling_threshold_ms,
        settings.recording_summary_rolling_chunk_duration_ms,
        settings.recording_summary_rolling_chunk_max_chars,
        settings.recording_summary_rolling_chunk_max_tokens,
        settings.recording_summary_rolling_memory_max_chars,
        generation_service,
    )


def build_summary_embedding_indexer(settings: Settings, worker_client: SyncWorkerClient) -> SummaryEmbeddingIndexer:
    return SummaryEmbeddingIndexer(
        worker_client,
        EmbeddingTokenCounter(settings.embedding_model, settings.resolved_embedding_model_cache_dir),
        settings.embedding_dimensions,
        settings.recording_summary_embedding_max_tokens,
    )


def _llm_model_name(settings: Settings, provider: LlmProvider) -> str:
    if provider == LlmProvider.LOCAL:
        return settings.resolved_local_llm_model_path.name
    if provider == LlmProvider.ZHIPU:
        return settings.zhipu_model
    if provider == LlmProvider.GEMINI:
        return settings.gemini_model
    if provider == LlmProvider.QWEN:
        return settings.qwen_llm_model
    raise ValueError(f"Unsupported LLM provider: {provider}")
