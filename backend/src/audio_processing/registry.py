from __future__ import annotations

from audio_processing.stages.align_transcript import AlignTranscriptStage
from audio_processing.stages.build_search_chunks import BuildSearchChunksStage
from audio_processing.stages.build_utterances import BuildUtterancesStage
from audio_processing.stages.correct_text import CorrectAsrWindowsStage, LocalTextCorrector
from audio_processing.stages.diarize_pyannote import PyannoteDiarizeStage
from audio_processing.stages.embedding_indexing import EmbeddingIndexingStage
from audio_processing.stages.normalize_audio import NormalizeAudioStage
from audio_processing.stages.preprocess_asr_audio import PreprocessAsrAudioStage
from audio_processing.stages.summary.stage import GenerateSummaryStage
from audio_processing.stages.transcribe_funasr_nano import FunAsrNanoTranscribeStage
from audio_processing.stages.transcribe_funasr_nano.context import build_funasr_hotwords
from audio_processing.stages.transcribe_qwen_asr import QwenAsrTranscribeStage
from audio_processing.stages.transcribe_qwen_asr.context import build_qwen_asr_context
from generation.service import GenerationService
from pipeline.registry import StageRegistry
from pipeline.runtime.artifact_store import ArtifactStore
from settings import REPOSITORY_ROOT, Settings


def build_recording_stage_registry(
    settings: Settings,
    artifact_store: ArtifactStore,
    generation_service: GenerationService | None = None,
    summary_stage: GenerateSummaryStage | None = None,
) -> StageRegistry:
    """Build every recording-owned stage; resource admission belongs to ResourceScheduler."""
    registry = StageRegistry()
    registry.register(NormalizeAudioStage(settings.resolved_local_storage_root))
    registry.register(PreprocessAsrAudioStage(settings.resolved_local_storage_root, artifact_store, settings.asr_preprocess_recording_enabled))
    registry.register(BuildUtterancesStage(artifact_store))
    registry.register(
        BuildSearchChunksStage(
            artifact_store,
            settings.search_chunk_max_chars,
            settings.search_chunk_max_duration_ms,
            settings.search_chunk_max_utterances,
            settings.search_chunk_topic_detection_enabled,
            settings.resolved_llm_correction_model_path,
            settings.llm_correction_context_size,
            settings.local_llm_verbose,
        )
    )
    registry.register(
        PyannoteDiarizeStage(
            settings.resolved_local_storage_root,
            artifact_store,
            settings.pyannote_model,
            settings.pyannote_auth_token,
            settings.resolved_pyannote_cache_dir,
            settings.pyannote_use_local_config,
            settings.pyannote_segment_merge_max_gap_ms,
            settings.pyannote_segment_merge_max_duration_ms,
            settings.pyannote_short_segment_absorb_max_duration_ms,
        )
    )
    registry.register(
        QwenAsrTranscribeStage(
            settings.resolved_local_storage_root,
            artifact_store,
            settings.qwen_asr_model,
            settings.qwen_asr_language,
            settings.resolved_huggingface_hub_cache_dir,
            build_qwen_asr_context(settings.resolved_qwen_asr_context_config, settings.qwen_asr_max_context_items, settings.qwen_asr_context),
            settings.qwen_asr_max_inference_batch_size,
            settings.asr_speech_window_target_duration_ms,
            settings.asr_speech_window_max_duration_ms,
            settings.asr_speech_window_overlap_ms,
        )
    )
    funasr_hotwords = build_funasr_hotwords(settings.resolved_qwen_asr_context_config, settings.qwen_asr_max_context_items)
    registry.register(
        FunAsrNanoTranscribeStage(
            settings.resolved_local_storage_root,
            artifact_store,
            settings.funasr_nano_model,
            settings.qwen_asr_language,
            settings.resolved_funasr_nano_cache_dir,
            "、".join(funasr_hotwords),
            settings.qwen_asr_max_inference_batch_size,
            settings.asr_speech_window_target_duration_ms,
            settings.asr_speech_window_max_duration_ms,
            settings.asr_speech_window_overlap_ms,
            funasr_hotwords,
        )
    )
    registry.register(
        CorrectAsrWindowsStage(
            artifact_store,
            LocalTextCorrector(
                repository_root=REPOSITORY_ROOT,
                pycorrector_enabled=settings.transcription_correction_enabled,
                llm_enabled=settings.llm_correction_enabled,
                llm_model_repo=settings.llm_correction_model_repo,
                llm_model_file=settings.llm_correction_model_file,
                llm_context_size=settings.llm_correction_context_size,
                prompt_config_path=settings.resolved_whisper_initial_prompt_config,
                llm_batch_max_units=settings.text_correction_batch_max_units,
                llm_batch_max_chars=settings.text_correction_batch_max_chars,
                llm_context_units=0,
            ),
            "pycorrector_llm" if settings.llm_correction_enabled else "pycorrector" if settings.transcription_correction_enabled else "rules",
            settings.llm_correction_model_repo if settings.llm_correction_enabled else None,
            settings.asr_window_correction_max_edit_ratio,
        )
    )
    registry.register(
        AlignTranscriptStage(
            settings.resolved_local_storage_root,
            artifact_store,
            settings.transcript_alignment_model,
            settings.resolved_huggingface_hub_cache_dir,
        )
    )
    registry.register(
        EmbeddingIndexingStage(artifact_store, settings.embedding_model, settings.resolved_embedding_model_cache_dir, settings.embedding_dimensions)
    )
    registry.register(summary_stage or build_recording_summary_stage(settings, artifact_store, generation_service))
    return registry


def build_recording_summary_stage(
    settings: Settings, artifact_store: ArtifactStore, generation_service: GenerationService | None = None
) -> GenerateSummaryStage:
    """Build the shared summary implementation for both pipeline and manual regeneration."""
    if settings.recording_summary_provider != "local_llm":
        raise ValueError("Only RECORDING_SUMMARY_PROVIDER=local_llm is supported by the Python pipeline")
    return GenerateSummaryStage(
        artifact_store,
        settings.resolved_local_llm_model_path,
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
        settings.local_llm_verbose,
    )
