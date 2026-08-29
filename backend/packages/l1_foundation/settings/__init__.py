from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self
from urllib.parse import quote

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from l1_foundation.model_ref import OnlineModelRef


def _find_repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "package.json").is_file() and (candidate / "backend").is_dir():
            return candidate
    raise RuntimeError("Could not locate the AIRecordSummary repository root")


REPOSITORY_ROOT = _find_repository_root()
ROOT_ENV_FILE = REPOSITORY_ROOT / ".env"
ROOT_ENV_LOCAL_FILE = REPOSITORY_ROOT / ".env.local"
LlmProviderName = Literal["local", "zhipu", "gemini", "qwen"]
RagAdjudicationSearchProvider = Literal["gemini", "chrome_ai_overview"]
RagAdjudicationAuditPromptVariant = Literal["relation_rules", "free_discovery"]


class Settings(BaseSettings):
    """Typed adapter for the repository-root .env configuration."""

    model_config = SettingsConfigDict(
        env_file=(ROOT_ENV_FILE, ROOT_ENV_LOCAL_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI Record Summary API"
    app_env: str = "development"
    api_prefix: str = "/api"
    db_host: str = Field(validation_alias="DB_HOST")
    db_port: int = Field(gt=0, le=65535, validation_alias="DB_PORT")
    db_user: str = Field(validation_alias="DB_USER")
    db_password: str = Field(validation_alias="DB_PASSWORD")
    db_name: str = Field(validation_alias="DB_NAME")
    db_admin_database: str = Field(validation_alias="DB_ADMIN_DATABASE")
    db_ssl: bool = Field(validation_alias="DB_SSL")
    database_url: str | None = Field(default=None, validation_alias="DATABASE_URL")
    kafka_bootstrap_servers: str = Field(default="127.0.0.1:9092", validation_alias="KAFKA_BOOTSTRAP_SERVERS")
    kafka_client_id: str = Field(default="ai-record-summary", validation_alias="KAFKA_CLIENT_ID")
    kafka_request_timeout_ms: int = Field(default=30_000, gt=0, validation_alias="KAFKA_REQUEST_TIMEOUT_MS")
    kafka_consumer_max_poll_interval_ms: int = Field(default=900_000, gt=0, validation_alias="KAFKA_CONSUMER_MAX_POLL_INTERVAL_MS")
    processing_consumer_max_poll_interval_ms: int = Field(
        default=7_200_000,
        gt=0,
        validation_alias="PROCESSING_CONSUMER_MAX_POLL_INTERVAL_MS",
    )
    outbox_relay_batch_size: int = Field(default=100, ge=1, le=1000, validation_alias="OUTBOX_RELAY_BATCH_SIZE")
    outbox_relay_poll_seconds: float = Field(default=0.5, ge=0.05, le=60, validation_alias="OUTBOX_RELAY_POLL_SECONDS")
    outbox_relay_lease_seconds: int = Field(default=60, ge=5, le=3600, validation_alias="OUTBOX_RELAY_LEASE_SECONDS")
    outbox_relay_max_attempts: int = Field(default=20, ge=1, le=1000, validation_alias="OUTBOX_RELAY_MAX_ATTEMPTS")
    outbox_relay_metrics_seconds: float = Field(default=30, ge=1, le=3600, validation_alias="OUTBOX_RELAY_METRICS_SECONDS")
    outbox_retention_days: int = Field(default=14, ge=1, le=365, validation_alias="OUTBOX_RETENTION_DAYS")
    redis_url: str = Field(default="redis://127.0.0.1:6379/0", validation_alias="REDIS_URL")
    redis_stream_maxlen: int = Field(default=10_000, gt=0, validation_alias="REDIS_STREAM_MAXLEN")
    redis_terminal_ttl_seconds: int = Field(default=86_400, gt=0, validation_alias="REDIS_TERMINAL_TTL_SECONDS")
    redis_stream_block_ms: int = Field(default=15_000, gt=0, validation_alias="REDIS_STREAM_BLOCK_MS")
    local_storage_root: Path = Field(validation_alias="LOCAL_STORAGE_ROOT")
    asr_lab_project_dataset_root: Path = Field(
        default=Path("datasets/asr-encrypted"),
        validation_alias="ASR_LAB_PROJECT_DATASET_ROOT",
    )
    audio_model_cache_root: Path = Field(default=Path("model-cache"), validation_alias="AUDIO_MODEL_CACHE_ROOT")
    asr_provider: Literal["qwen_asr", "funasr_nano"] = Field(default="qwen_asr", validation_alias="ASR_PROVIDER")
    qwen_asr_model: str = Field(default="Qwen/Qwen3-ASR-1.7B", validation_alias="QWEN_ASR_MODEL")
    funasr_nano_model: str = Field(default="FunAudioLLM/Fun-ASR-Nano-2512", validation_alias="FUNASR_NANO_MODEL")
    qwen_asr_language: str = Field(default="auto", validation_alias="QWEN_ASR_LANGUAGE")
    qwen_asr_context_config: Path = Field(default=Path("config/initial-prompt.json"), validation_alias="QWEN_ASR_CONTEXT_CONFIG")
    qwen_asr_context: str = Field(default="", validation_alias="QWEN_ASR_CONTEXT")
    qwen_asr_max_context_items: int = Field(default=200, ge=0, validation_alias="QWEN_ASR_MAX_CONTEXT_ITEMS")
    qwen_asr_max_inference_batch_size: int = Field(default=1, ge=1, validation_alias="QWEN_ASR_MAX_INFERENCE_BATCH_SIZE")
    qwen_asr_num_beams: int = Field(default=2, ge=1, le=8, validation_alias="QWEN_ASR_NUM_BEAMS")
    qwen_asr_tempo: float = Field(default=1.0, ge=0.5, le=2.0, validation_alias="QWEN_ASR_TEMPO")
    asr_lab_training_python_bin: Path = Field(
        default=Path("backend/packages/l2_core/trainers/qwen-asr-lora/.venv/bin/python"),
        validation_alias="ASR_LAB_TRAINING_PYTHON_BIN",
    )
    asr_lab_training_module: str = Field(default="qwen_asr_lora", validation_alias="ASR_LAB_TRAINING_MODULE")
    asr_lab_training_model: str = Field(default="Qwen/Qwen3-ASR-1.7B-hf", validation_alias="ASR_LAB_TRAINING_MODEL")
    asr_lab_worker_poll_seconds: float = Field(default=2.0, ge=0.2, le=60, validation_alias="ASR_LAB_WORKER_POLL_SECONDS")
    qwen_asr_enhance_low_volume_segments: bool = Field(default=True, validation_alias="QWEN_ASR_ENHANCE_LOW_VOLUME_SEGMENTS")
    asr_preprocess_recording_enabled: bool = Field(
        default=False, validation_alias=AliasChoices("ASR_AUDIO_PREPROCESSING_ENABLED", "ASR_PREPROCESS_RECORDING_ENABLED")
    )
    transcript_alignment_enabled: bool = Field(default=True, validation_alias="TRANSCRIPT_ALIGNMENT_ENABLED")
    transcript_alignment_provider: Literal["qwen_forced_aligner"] = Field(default="qwen_forced_aligner", validation_alias="TRANSCRIPT_ALIGNMENT_PROVIDER")
    transcript_alignment_model: str = Field(default="Qwen/Qwen3-ForcedAligner-0.6B", validation_alias="TRANSCRIPT_ALIGNMENT_MODEL")
    asr_speech_window_target_duration_ms: int = Field(default=30_000, gt=0, validation_alias="ASR_SPEECH_WINDOW_TARGET_DURATION_MS")
    asr_speech_window_max_duration_ms: int = Field(default=80_000, gt=0, validation_alias="ASR_SPEECH_WINDOW_MAX_DURATION_MS")
    asr_speech_window_overlap_ms: int = Field(default=500, ge=0, validation_alias="ASR_SPEECH_WINDOW_OVERLAP_MS")
    asr_window_correction_max_edit_ratio: float = Field(default=0.35, ge=0, le=1, validation_alias="ASR_WINDOW_CORRECTION_MAX_EDIT_RATIO")
    pyannote_segment_merge_max_gap_ms: int = Field(default=3_000, ge=0, validation_alias="PYANNOTE_SEGMENT_MERGE_MAX_GAP_MS")
    pyannote_segment_merge_max_duration_ms: int = Field(default=80_000, gt=0, validation_alias="PYANNOTE_SEGMENT_MERGE_MAX_DURATION_MS")
    pyannote_short_segment_absorb_max_duration_ms: int = Field(
        default=2_000,
        ge=0,
        validation_alias="PYANNOTE_SHORT_SEGMENT_ABSORB_MAX_DURATION_MS",
    )
    pyannote_short_segment_absorb_max_gap_ms: int = Field(
        default=2_000,
        ge=0,
        validation_alias="PYANNOTE_SHORT_SEGMENT_ABSORB_MAX_GAP_MS",
    )
    qwen_asr_low_volume_rms_threshold: float = Field(default=0.01, ge=0, le=1, validation_alias="QWEN_ASR_LOW_VOLUME_RMS_THRESHOLD")
    qwen_asr_low_volume_peak_threshold: float = Field(default=0.08, ge=0, le=1, validation_alias="QWEN_ASR_LOW_VOLUME_PEAK_THRESHOLD")
    qwen_asr_low_volume_max_gain_db: float = Field(default=9.0, ge=0, le=24, validation_alias="QWEN_ASR_LOW_VOLUME_MAX_GAIN_DB")
    qwen_asr_speaker_segment_min_duration_ms: int = Field(default=1200, ge=0, validation_alias="QWEN_ASR_SPEAKER_SEGMENT_MIN_DURATION_MS")
    qwen_asr_speaker_segment_merge_max_gap_ms: int = Field(default=2000, ge=-1, validation_alias="QWEN_ASR_SPEAKER_SEGMENT_MERGE_MAX_GAP_MS")
    qwen_asr_speaker_segment_merge_max_duration_ms: int = Field(default=60_000, ge=0, validation_alias="QWEN_ASR_SPEAKER_SEGMENT_MERGE_MAX_DURATION_MS")
    transcription_correction_enabled: bool = Field(default=False, validation_alias="TRANSCRIPTION_CORRECTION_ENABLED")
    whisper_initial_prompt_config: Path = Field(default=Path("config/initial-prompt.json"), validation_alias="WHISPER_INITIAL_PROMPT_CONFIG")
    llm_default_provider: LlmProviderName = Field(default="gemini", validation_alias="LLM_DEFAULT_PROVIDER")
    llm_correction_enabled: bool = Field(default=True, validation_alias="LLM_CORRECTION_ENABLED")
    llm_correction_provider: LlmProviderName = Field(
        default="gemini",
        validation_alias=AliasChoices("LLM_CORRECTION_PROVIDER", "LLM_DEFAULT_PROVIDER"),
    )
    llm_correction_context_size: int = Field(default=8192, gt=0, validation_alias="LLM_CORRECTION_CONTEXT_SIZE")
    llm_correction_max_output_tokens: int = Field(default=65_536, gt=0, validation_alias="LLM_CORRECTION_MAX_OUTPUT_TOKENS")
    text_correction_batch_max_units: int = Field(default=16, gt=0, validation_alias="TEXT_CORRECTION_BATCH_MAX_UNITS")
    text_correction_batch_max_chars: int = Field(default=4000, gt=0, validation_alias="TEXT_CORRECTION_BATCH_MAX_CHARS")
    text_correction_context_units: int = Field(default=1, ge=0, validation_alias="TEXT_CORRECTION_CONTEXT_UNITS")
    search_chunk_topic_detection_enabled: bool = Field(default=True, validation_alias="SEARCH_CHUNK_TOPIC_DETECTION_ENABLED")
    topic_detection_provider: LlmProviderName = Field(
        default="gemini",
        validation_alias=AliasChoices("TOPIC_DETECTION_PROVIDER", "LLM_DEFAULT_PROVIDER"),
    )
    search_chunk_max_token: int = Field(default=800, gt=0, validation_alias="SEARCH_CHUNK_MAX_TOKEN")
    search_chunk_max_duration_ms: int = Field(default=180_000, gt=0, validation_alias="SEARCH_CHUNK_MAX_DURATION_MS")
    search_chunk_max_utterances: int = Field(default=30, gt=0, validation_alias="SEARCH_CHUNK_MAX_UTTERANCES")
    rag_chunk_context_window_utterances: int = Field(default=1, ge=0, le=10, validation_alias="RAG_CHUNK_CONTEXT_WINDOW_UTTERANCES")
    rag_hybrid_search_enabled: bool = Field(default=True, validation_alias="RAG_HYBRID_SEARCH_ENABLED")
    rag_query_term_expansion_enabled: bool = Field(default=True, validation_alias="RAG_QUERY_TERM_EXPANSION_ENABLED")
    rag_vector_candidate_limit: int = Field(default=30, gt=0, le=200, validation_alias="RAG_VECTOR_CANDIDATE_LIMIT")
    rag_lexical_candidate_limit: int = Field(default=30, gt=0, le=200, validation_alias="RAG_LEXICAL_CANDIDATE_LIMIT")
    rag_fused_candidate_limit: int = Field(default=20, gt=0, le=200, validation_alias="RAG_FUSED_CANDIDATE_LIMIT")
    rag_rrf_k: int = Field(default=60, gt=0, validation_alias="RAG_RRF_K")
    rag_original_vector_weight: float = Field(default=0.7, ge=0, validation_alias="RAG_ORIGINAL_VECTOR_WEIGHT")
    rag_expanded_vector_weight: float = Field(default=0.2, ge=0, validation_alias="RAG_EXPANDED_VECTOR_WEIGHT")
    rag_lexical_weight: float = Field(default=0.1, ge=0, validation_alias="RAG_LEXICAL_WEIGHT")
    rag_recording_profile_search_enabled: bool = Field(default=True, validation_alias="RAG_RECORDING_PROFILE_SEARCH_ENABLED")
    rag_recording_profile_candidate_limit: int = Field(default=3, gt=0, le=20, validation_alias="RAG_RECORDING_PROFILE_CANDIDATE_LIMIT")
    rag_recording_profile_min_score: float = Field(default=0.30, ge=-1, le=1, validation_alias="RAG_RECORDING_PROFILE_MIN_SCORE")
    rag_recording_profile_scoped_chunk_limit: int = Field(
        default=2,
        gt=0,
        le=10,
        validation_alias="RAG_RECORDING_PROFILE_SCOPED_CHUNK_LIMIT",
    )
    embedding_model: str = Field(default="Qwen/Qwen3-Embedding-4B", validation_alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=2560, gt=0, validation_alias="EMBEDDING_DIMENSIONS")
    embedding_inference_batch_size: int = Field(default=8, gt=0, le=64, validation_alias="EMBEDDING_INFERENCE_BATCH_SIZE")
    embedding_model_cache_dir: Path = Field(default=Path("model-cache/embedding"), validation_alias="EMBEDDING_MODEL_CACHE_DIR")
    recording_summary_provider: LlmProviderName = Field(
        default="gemini",
        validation_alias=AliasChoices("RECORDING_SUMMARY_PROVIDER", "LLM_DEFAULT_PROVIDER"),
    )
    recording_summary_prompt_config: Path = Field(default=Path("config/initial-prompt.json"), validation_alias="RECORDING_SUMMARY_PROMPT_CONFIG")
    recording_summary_context_size: int = Field(default=262_144, gt=0, validation_alias="RECORDING_SUMMARY_CONTEXT_SIZE")
    recording_summary_max_tokens: int = Field(default=4_096, gt=0, validation_alias="RECORDING_SUMMARY_MAX_TOKENS")
    recording_summary_rolling_enabled: bool = Field(default=False, validation_alias="RECORDING_SUMMARY_ROLLING_ENABLED")
    recording_summary_rolling_threshold_ms: int = Field(default=1_800_000, gt=0, validation_alias="RECORDING_SUMMARY_ROLLING_THRESHOLD_MS")
    recording_summary_rolling_chunk_duration_ms: int = Field(default=600_000, gt=0, validation_alias="RECORDING_SUMMARY_ROLLING_CHUNK_DURATION_MS")
    recording_summary_rolling_chunk_max_chars: int = Field(default=8000, gt=0, validation_alias="RECORDING_SUMMARY_ROLLING_CHUNK_MAX_CHARS")
    recording_summary_rolling_chunk_max_tokens: int = Field(default=1800, gt=0, validation_alias="RECORDING_SUMMARY_ROLLING_CHUNK_MAX_TOKENS")
    recording_summary_rolling_memory_max_chars: int = Field(default=6000, gt=0, validation_alias="RECORDING_SUMMARY_ROLLING_MEMORY_MAX_CHARS")
    recording_summary_embedding_max_tokens: int = Field(default=512, gt=0, validation_alias="RECORDING_SUMMARY_EMBEDDING_MAX_TOKENS")
    local_llm_model_repo: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct-GGUF",
        validation_alias=AliasChoices("LOCAL_LLM_MODEL_REPO", "LLM_CORRECTION_MODEL_REPO"),
    )
    local_llm_model_file: str = Field(
        default="qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        validation_alias=AliasChoices("LOCAL_LLM_MODEL_FILE", "LLM_CORRECTION_MODEL_FILE"),
    )
    local_llm_verbose: bool = Field(default=False, validation_alias="LOCAL_LLM_VERBOSE")
    rag_context_size: int = Field(default=16_384, gt=0, validation_alias="RAG_CONTEXT_SIZE")
    rag_local_model_repo: str = Field(default="Qwen/Qwen3-4B-GGUF", validation_alias="RAG_LOCAL_MODEL_REPO")
    rag_local_model_file: str = Field(default="Qwen3-4B-Q4_K_M.gguf", validation_alias="RAG_LOCAL_MODEL_FILE")
    rag_route_model_profile: Literal["default", "rag"] = Field(
        default="default",
        validation_alias="RAG_ROUTE_MODEL_PROFILE",
    )
    rag_node_model_profile: Literal["default", "rag"] = Field(
        default="default",
        validation_alias="RAG_NODE_MODEL_PROFILE",
    )
    rag_plan_local_input_tokens: int = Field(default=4_000, gt=0, validation_alias="RAG_PLAN_LOCAL_INPUT_TOKENS")
    rag_run_max_total_tokens: int = Field(default=50_000, gt=0, validation_alias="RAG_RUN_MAX_TOTAL_TOKENS")
    rag_evaluation_stale_run_seconds: float = Field(default=120.0, ge=30, validation_alias="RAG_EVALUATION_STALE_RUN_SECONDS")
    rag_rerank_enabled: bool = Field(default=True, validation_alias="RAG_RERANK_ENABLED")
    rag_rerank_model: str = Field(default="Qwen/Qwen3-Reranker-0.6B", validation_alias="RAG_RERANK_MODEL")
    rag_rerank_model_cache_dir: Path = Field(default=Path("model-cache/rerank"), validation_alias="RAG_RERANK_MODEL_CACHE_DIR")
    rag_rerank_candidate_limit: int = Field(default=20, gt=0, le=200, validation_alias="RAG_RERANK_CANDIDATE_LIMIT")
    rag_rerank_max_total_tokens: int = Field(default=16_000, gt=0, validation_alias="RAG_RERANK_MAX_TOTAL_TOKENS")
    rag_rerank_inference_batch_size: int = Field(default=1, gt=0, le=32, validation_alias="RAG_RERANK_INFERENCE_BATCH_SIZE")
    rag_rerank_output_limit: int = Field(default=20, gt=0, le=100, validation_alias="RAG_RERANK_OUTPUT_LIMIT")
    rag_sql_statement_timeout_ms: int = Field(default=15_000, gt=0, validation_alias="RAG_SQL_STATEMENT_TIMEOUT_MS")
    rag_checkpoint_ttl_seconds: int = Field(default=604_800, gt=0, validation_alias="RAG_CHECKPOINT_TTL_SECONDS")
    rag_asr_adjudication_enabled: bool = Field(default=False, validation_alias="RAG_ASR_ADJUDICATION_ENABLED")
    rag_asr_adjudication_web_search_enabled: bool = Field(
        default=False,
        validation_alias="RAG_ASR_ADJUDICATION_WEB_SEARCH_ENABLED",
    )
    rag_asr_adjudication_auto_resolve_confidence: float = Field(
        default=0.95,
        ge=0,
        le=1,
        validation_alias="RAG_ASR_ADJUDICATION_AUTO_RESOLVE_CONFIDENCE",
    )
    rag_asr_adjudication_audit_prompt_variant: RagAdjudicationAuditPromptVariant = Field(
        default="relation_rules",
        validation_alias="RAG_ASR_ADJUDICATION_AUDIT_PROMPT_VARIANT",
    )
    rag_asr_adjudication_audit_model: str = Field(
        default="gemini-gemini-3.5-flash-lite",
        min_length=1,
        validation_alias="RAG_ASR_ADJUDICATION_AUDIT_MODEL",
    )
    rag_asr_adjudication_construct_model: str = Field(
        default="gemini-gemini-3.5-flash-lite",
        min_length=1,
        validation_alias="RAG_ASR_ADJUDICATION_CONSTRUCT_MODEL",
    )
    rag_asr_adjudication_decision_model: str = Field(
        default="gemini-gemini-3.5-flash-lite",
        min_length=1,
        validation_alias="RAG_ASR_ADJUDICATION_DECISION_MODEL",
    )
    rag_asr_adjudication_audit_min_request_interval_seconds: float = Field(
        default=15.0,
        ge=0,
        validation_alias="RAG_ASR_ADJUDICATION_AUDIT_MIN_REQUEST_INTERVAL_SECONDS",
    )
    rag_asr_adjudication_search_provider: RagAdjudicationSearchProvider = Field(
        default="gemini",
        validation_alias="RAG_ASR_ADJUDICATION_SEARCH_PROVIDER",
    )
    rag_asr_adjudication_search_model: str = Field(
        default="gemini-2.5-flash-lite",
        validation_alias="RAG_ASR_ADJUDICATION_SEARCH_MODEL",
    )
    gemini_native_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        validation_alias="GEMINI_NATIVE_BASE_URL",
    )
    rag_asr_adjudication_chrome_aio_timeout_seconds: float = Field(
        default=45.0,
        gt=0,
        validation_alias="RAG_ASR_ADJUDICATION_CHROME_AIO_TIMEOUT_SECONDS",
    )
    rag_asr_adjudication_chrome_aio_poll_interval_seconds: float = Field(
        default=1.0,
        gt=0,
        validation_alias="RAG_ASR_ADJUDICATION_CHROME_AIO_POLL_INTERVAL_SECONDS",
    )
    rag_online_default_model: str = Field(
        default="gemini-gemini-3.5-flash-lite",
        min_length=1,
        validation_alias="RAG_ONLINE_DEFAULT_MODEL",
    )
    zhipu_api_key: str | None = Field(default=None, validation_alias="ZHIPU_API_KEY")
    zhipu_model: str = Field(default="glm-4.5-flash", validation_alias="ZHIPU_MODEL")
    zhipu_base_url: str = Field(default="https://open.bigmodel.cn/api/paas/v4", validation_alias="ZHIPU_BASE_URL")
    zhipu_timeout_seconds: float = Field(default=120.0, gt=0, validation_alias="ZHIPU_TIMEOUT_SECONDS")
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.5-flash-lite", validation_alias="GEMINI_MODEL")
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai",
        validation_alias="GEMINI_BASE_URL",
    )
    gemini_timeout_seconds: float = Field(default=300.0, gt=0, validation_alias="GEMINI_TIMEOUT_SECONDS")
    gemini_min_request_interval_seconds: float = Field(
        default=5.0,
        ge=0,
        validation_alias="GEMINI_MIN_REQUEST_INTERVAL_SECONDS",
    )
    qwen_ai_platform_api_key: str | None = Field(default=None, validation_alias="QWEN_AI_PLATFORM_API_KEY")
    qwen_llm_model: str = Field(default="qwen3.8-flash", validation_alias="QWEN_LLM_MODEL")
    qwen_llm_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias="QWEN_LLM_BASE_URL",
    )
    qwen_llm_timeout_seconds: float = Field(default=300.0, gt=0, validation_alias="QWEN_LLM_TIMEOUT_SECONDS")
    qwen_llm_min_request_interval_seconds: float = Field(
        default=0,
        ge=0,
        validation_alias="QWEN_LLM_MIN_REQUEST_INTERVAL_SECONDS",
    )
    pyannote_auth_token: str | None = Field(default=None, validation_alias="PYANNOTE_AUTH_TOKEN")
    pyannote_model: str = Field(default="pyannote/speaker-diarization-3.1", validation_alias="PYANNOTE_MODEL")
    pyannote_use_local_config: bool = Field(default=True, validation_alias="PYANNOTE_USE_LOCAL_CONFIG")
    compute_worker_host: str = Field(default="127.0.0.1", validation_alias="COMPUTE_WORKER_HOST")
    compute_worker_port: int = Field(default=8010, gt=0, le=65535, validation_alias="COMPUTE_WORKER_PORT")
    compute_worker_internal_token: str | None = Field(default=None, validation_alias="COMPUTE_WORKER_INTERNAL_TOKEN")
    compute_worker_completed_ttl_seconds: float = Field(default=1800, gt=0, validation_alias="COMPUTE_WORKER_COMPLETED_TTL_SECONDS")
    compute_worker_max_tasks: int = Field(default=100, gt=0, validation_alias="COMPUTE_WORKER_MAX_TASKS")
    compute_worker_heartbeat_seconds: float = Field(default=15, gt=0, validation_alias="COMPUTE_WORKER_HEARTBEAT_SECONDS")
    compute_inline_result_limit_bytes: int = Field(default=256 * 1024, gt=0, validation_alias="COMPUTE_INLINE_RESULT_LIMIT_BYTES")
    compute_reply_wait_timeout_seconds: float = Field(default=30.0, gt=0, validation_alias="COMPUTE_REPLY_WAIT_TIMEOUT_SECONDS")
    compute_worker_cancel_grace_seconds: float = Field(default=10.0, gt=0, validation_alias="COMPUTE_WORKER_CANCEL_GRACE_SECONDS")
    observability_enabled: bool = Field(default=True, validation_alias="OBSERVABILITY_ENABLED")
    observability_api_host: str = Field(default="127.0.0.1", validation_alias="OBSERVABILITY_API_HOST")
    observability_api_port: int = Field(default=8003, gt=0, le=65535, validation_alias="OBSERVABILITY_API_PORT")
    session_cookie_name: str = Field(default="ai_record_summary_session", validation_alias="SESSION_COOKIE_NAME")
    session_ttl_days: int = Field(default=14, ge=1, le=90, validation_alias="SESSION_TTL_DAYS")
    session_cookie_secure: bool = Field(default=False, validation_alias="SESSION_COOKIE_SECURE")
    bootstrap_admin_email: str = Field(default="admin@local.test", validation_alias="BOOTSTRAP_ADMIN_EMAIL")
    bootstrap_admin_password: str = Field(default="change-me-now", validation_alias="BOOTSTRAP_ADMIN_PASSWORD")
    bootstrap_workspace_name: str = Field(default="默认工作区", validation_alias="BOOTSTRAP_WORKSPACE_NAME")

    @model_validator(mode="after")
    def validate_hybrid_retrieval_settings(self) -> Self:
        OnlineModelRef.parse(self.rag_online_default_model)
        OnlineModelRef.parse(self.rag_asr_adjudication_audit_model)
        OnlineModelRef.parse(self.rag_asr_adjudication_construct_model)
        OnlineModelRef.parse(self.rag_asr_adjudication_decision_model)
        if self.rag_original_vector_weight == 0 and self.rag_expanded_vector_weight == 0 and self.rag_lexical_weight == 0:
            raise ValueError("RAG retrieval weights cannot all be zero")
        if self.rag_fused_candidate_limit > self.rag_vector_candidate_limit + self.rag_lexical_candidate_limit:
            raise ValueError("RAG fused candidate limit cannot exceed the sum of branch candidate limits")
        if self.rag_rerank_output_limit > self.rag_rerank_candidate_limit:
            raise ValueError("RAG rerank output limit cannot exceed its candidate limit")
        return self

    def _database_url_for(self, database: str) -> str:
        if self.database_url is not None and database == self.db_name:
            return self.database_url
        ssl_suffix = "?sslmode=require" if self.db_ssl else ""
        user = quote(self.db_user, safe="")
        password = quote(self.db_password, safe="")
        return f"postgresql+psycopg://{user}:{password}@{self.db_host}:{self.db_port}/{database}{ssl_suffix}"

    @property
    def sqlalchemy_database_url(self) -> str:
        return self._database_url_for(self.db_name)

    @property
    def sqlalchemy_admin_database_url(self) -> str:
        return self._database_url_for(self.db_admin_database)

    @property
    def resolved_local_storage_root(self) -> Path:
        """Resolve the existing root-level LOCAL_STORAGE_ROOT against the repository."""
        if self.local_storage_root.is_absolute():
            return self.local_storage_root
        return (REPOSITORY_ROOT / self.local_storage_root).resolve()

    @property
    def resolved_asr_lab_project_dataset_root(self) -> Path:
        if self.asr_lab_project_dataset_root.is_absolute():
            return self.asr_lab_project_dataset_root
        return (REPOSITORY_ROOT / self.asr_lab_project_dataset_root).resolve()

    @property
    def resolved_whisper_initial_prompt_config(self) -> Path:
        if self.whisper_initial_prompt_config.is_absolute():
            return self.whisper_initial_prompt_config
        return (REPOSITORY_ROOT / self.whisper_initial_prompt_config).resolve()

    @property
    def resolved_qwen_asr_context_config(self) -> Path:
        return self._resolve_repository_path(self.qwen_asr_context_config)

    @property
    def resolved_huggingface_hub_cache_dir(self) -> Path:
        return self._resolve_repository_path(self.audio_model_cache_root) / "huggingface" / "hub"

    @property
    def resolved_audio_model_cache_root(self) -> Path:
        return self._resolve_repository_path(self.audio_model_cache_root)

    @property
    def resolved_asr_lab_training_python_bin(self) -> Path:
        value = self.asr_lab_training_python_bin
        candidate = value if value.is_absolute() else REPOSITORY_ROOT / value
        # A venv's Python executable is normally a symlink to the base
        # interpreter. Resolving that symlink would bypass the venv's
        # site-packages when the executable is passed to subprocess.
        return candidate.absolute()

    @property
    def resolved_funasr_nano_cache_dir(self) -> Path:
        return self.resolved_huggingface_hub_cache_dir

    @property
    def resolved_embedding_model_cache_dir(self) -> Path:
        return self._resolve_repository_path(self.embedding_model_cache_dir)

    @property
    def resolved_pyannote_cache_dir(self) -> Path:
        return self.resolved_huggingface_hub_cache_dir

    @property
    def resolved_recording_summary_prompt_config(self) -> Path:
        return self._resolve_repository_path(self.recording_summary_prompt_config)

    @property
    def resolved_local_llm_model_path(self) -> Path:
        model_file = self.local_llm_model_file.split(",", maxsplit=1)[0].strip()
        if not model_file:
            raise ValueError("LOCAL_LLM_MODEL_FILE must contain a model filename")
        return self.resolved_audio_model_cache_root / "llm-correction" / self.local_llm_model_repo.replace("/", "__") / model_file

    @property
    def resolved_llm_correction_model_path(self) -> Path:
        """Compatibility alias for the single local Qwen model path."""
        return self.resolved_local_llm_model_path

    @property
    def resolved_rag_local_model_path(self) -> Path:
        model_file = self.rag_local_model_file.split(",", maxsplit=1)[0].strip()
        if not model_file:
            raise ValueError("RAG_LOCAL_MODEL_FILE must contain a model filename")
        return self.resolved_audio_model_cache_root / "rag-llm" / self.rag_local_model_repo.replace("/", "__") / model_file

    @property
    def resolved_rag_rerank_model_cache_dir(self) -> Path:
        return self._resolve_repository_path(self.rag_rerank_model_cache_dir)

    @staticmethod
    def _resolve_repository_path(value: Path) -> Path:
        return value if value.is_absolute() else (REPOSITORY_ROOT / value).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]  # Values are supplied by the configured root .env source.
