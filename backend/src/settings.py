from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self
from urllib.parse import quote

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ROOT_ENV_FILE = REPOSITORY_ROOT / ".env"


class Settings(BaseSettings):
    """Typed adapter for the repository-root .env configuration."""

    model_config = SettingsConfigDict(env_file=ROOT_ENV_FILE, env_file_encoding="utf-8", extra="ignore")

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
    local_storage_root: Path = Field(validation_alias="LOCAL_STORAGE_ROOT")
    audio_model_cache_root: Path = Field(default=Path("model-cache"), validation_alias="AUDIO_MODEL_CACHE_ROOT")
    asr_provider: Literal["qwen_asr", "funasr_nano"] = Field(default="qwen_asr", validation_alias="ASR_PROVIDER")
    qwen_asr_model: str = Field(default="Qwen/Qwen3-ASR-1.7B", validation_alias="QWEN_ASR_MODEL")
    funasr_nano_model: str = Field(default="FunAudioLLM/Fun-ASR-Nano-2512", validation_alias="FUNASR_NANO_MODEL")
    qwen_asr_language: str = Field(default="auto", validation_alias="QWEN_ASR_LANGUAGE")
    qwen_asr_context_config: Path = Field(default=Path("config/initial-prompt.json"), validation_alias="QWEN_ASR_CONTEXT_CONFIG")
    qwen_asr_context: str = Field(default="", validation_alias="QWEN_ASR_CONTEXT")
    qwen_asr_max_context_items: int = Field(default=200, ge=0, validation_alias="QWEN_ASR_MAX_CONTEXT_ITEMS")
    qwen_asr_max_inference_batch_size: int = Field(default=4, ge=1, validation_alias="QWEN_ASR_MAX_INFERENCE_BATCH_SIZE")
    asr_lab_training_python_bin: Path = Field(default=Path("backend/.venv/bin/python"), validation_alias="ASR_LAB_TRAINING_PYTHON_BIN")
    asr_lab_training_script: Path = Field(
        default=Path("backend/scripts/train_qwen_asr_lora.py"),
        validation_alias="ASR_LAB_TRAINING_SCRIPT",
    )
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
    asr_window_correction_max_edit_ratio: float = Field(default=0.25, ge=0, le=1, validation_alias="ASR_WINDOW_CORRECTION_MAX_EDIT_RATIO")
    pyannote_segment_merge_max_gap_ms: int = Field(default=3_000, ge=0, validation_alias="PYANNOTE_SEGMENT_MERGE_MAX_GAP_MS")
    pyannote_segment_merge_max_duration_ms: int = Field(default=80_000, gt=0, validation_alias="PYANNOTE_SEGMENT_MERGE_MAX_DURATION_MS")
    pyannote_short_segment_absorb_max_duration_ms: int = Field(
        default=2_000,
        ge=0,
        validation_alias="PYANNOTE_SHORT_SEGMENT_ABSORB_MAX_DURATION_MS",
    )
    qwen_asr_low_volume_rms_threshold: float = Field(default=0.004, ge=0, le=1, validation_alias="QWEN_ASR_LOW_VOLUME_RMS_THRESHOLD")
    qwen_asr_low_volume_peak_threshold: float = Field(default=0.025, ge=0, le=1, validation_alias="QWEN_ASR_LOW_VOLUME_PEAK_THRESHOLD")
    qwen_asr_speaker_segment_min_duration_ms: int = Field(default=1200, ge=0, validation_alias="QWEN_ASR_SPEAKER_SEGMENT_MIN_DURATION_MS")
    qwen_asr_speaker_segment_merge_max_gap_ms: int = Field(default=2000, ge=-1, validation_alias="QWEN_ASR_SPEAKER_SEGMENT_MERGE_MAX_GAP_MS")
    qwen_asr_speaker_segment_merge_max_duration_ms: int = Field(default=60_000, ge=0, validation_alias="QWEN_ASR_SPEAKER_SEGMENT_MERGE_MAX_DURATION_MS")
    transcription_correction_enabled: bool = Field(default=True, validation_alias="TRANSCRIPTION_CORRECTION_ENABLED")
    whisper_initial_prompt_config: Path = Field(default=Path("config/initial-prompt.json"), validation_alias="WHISPER_INITIAL_PROMPT_CONFIG")
    llm_correction_enabled: bool = Field(default=True, validation_alias="LLM_CORRECTION_ENABLED")
    llm_correction_model_repo: str = Field(default="Qwen/Qwen2.5-7B-Instruct-GGUF", validation_alias="LLM_CORRECTION_MODEL_REPO")
    llm_correction_model_file: str = Field(default="qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", validation_alias="LLM_CORRECTION_MODEL_FILE")
    llm_correction_context_size: int = Field(default=8192, gt=0, validation_alias="LLM_CORRECTION_CONTEXT_SIZE")
    text_correction_batch_max_units: int = Field(default=16, gt=0, validation_alias="TEXT_CORRECTION_BATCH_MAX_UNITS")
    text_correction_batch_max_chars: int = Field(default=4000, gt=0, validation_alias="TEXT_CORRECTION_BATCH_MAX_CHARS")
    search_chunk_topic_detection_enabled: bool = Field(default=True, validation_alias="SEARCH_CHUNK_TOPIC_DETECTION_ENABLED")
    search_chunk_max_chars: int = Field(default=1200, gt=0, validation_alias="SEARCH_CHUNK_MAX_CHARS")
    search_chunk_max_duration_ms: int = Field(default=180_000, gt=0, validation_alias="SEARCH_CHUNK_MAX_DURATION_MS")
    search_chunk_max_utterances: int = Field(default=30, gt=0, validation_alias="SEARCH_CHUNK_MAX_UTTERANCES")
    rag_chunk_context_window_utterances: int = Field(default=1, ge=0, le=10, validation_alias="RAG_CHUNK_CONTEXT_WINDOW_UTTERANCES")
    rag_hybrid_search_enabled: bool = Field(default=True, validation_alias="RAG_HYBRID_SEARCH_ENABLED")
    rag_vector_candidate_limit: int = Field(default=30, gt=0, le=200, validation_alias="RAG_VECTOR_CANDIDATE_LIMIT")
    rag_lexical_candidate_limit: int = Field(default=30, gt=0, le=200, validation_alias="RAG_LEXICAL_CANDIDATE_LIMIT")
    rag_fused_candidate_limit: int = Field(default=20, gt=0, le=200, validation_alias="RAG_FUSED_CANDIDATE_LIMIT")
    rag_rrf_k: int = Field(default=60, gt=0, validation_alias="RAG_RRF_K")
    rag_vector_weight: float = Field(default=1.0, ge=0, validation_alias="RAG_VECTOR_WEIGHT")
    rag_lexical_weight: float = Field(default=1.0, ge=0, validation_alias="RAG_LEXICAL_WEIGHT")
    embedding_model: str = Field(default="Qwen/Qwen3-Embedding-4B", validation_alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=2560, gt=0, validation_alias="EMBEDDING_DIMENSIONS")
    embedding_model_cache_dir: Path = Field(default=Path("model-cache/embedding"), validation_alias="EMBEDDING_MODEL_CACHE_DIR")
    recording_summary_provider: str = Field(default="local_llm", validation_alias="RECORDING_SUMMARY_PROVIDER")
    recording_summary_prompt_config: Path = Field(default=Path("config/initial-prompt.json"), validation_alias="RECORDING_SUMMARY_PROMPT_CONFIG")
    recording_summary_context_size: int = Field(default=262_144, gt=0, validation_alias="RECORDING_SUMMARY_CONTEXT_SIZE")
    recording_summary_max_tokens: int = Field(default=4_096, gt=0, validation_alias="RECORDING_SUMMARY_MAX_TOKENS")
    recording_summary_rolling_enabled: bool = Field(default=False, validation_alias="RECORDING_SUMMARY_ROLLING_ENABLED")
    recording_summary_rolling_threshold_ms: int = Field(default=1_800_000, gt=0, validation_alias="RECORDING_SUMMARY_ROLLING_THRESHOLD_MS")
    recording_summary_rolling_chunk_duration_ms: int = Field(default=600_000, gt=0, validation_alias="RECORDING_SUMMARY_ROLLING_CHUNK_DURATION_MS")
    recording_summary_rolling_chunk_max_chars: int = Field(default=8000, gt=0, validation_alias="RECORDING_SUMMARY_ROLLING_CHUNK_MAX_CHARS")
    recording_summary_rolling_chunk_max_tokens: int = Field(default=1800, gt=0, validation_alias="RECORDING_SUMMARY_ROLLING_CHUNK_MAX_TOKENS")
    recording_summary_rolling_memory_max_chars: int = Field(default=6000, gt=0, validation_alias="RECORDING_SUMMARY_ROLLING_MEMORY_MAX_CHARS")
    local_llm_model_repo: str = Field(default="DevQuasar/Qwen.Qwen3.5-9B-GGUF", validation_alias="LOCAL_LLM_MODEL_REPO")
    local_llm_model_file: str = Field(default="Qwen.Qwen3.5-9B.Q8_0.gguf", validation_alias="LOCAL_LLM_MODEL_FILE")
    local_llm_verbose: bool = Field(default=False, validation_alias="LOCAL_LLM_VERBOSE")
    rag_context_size: int = Field(default=16_384, gt=0, validation_alias="RAG_CONTEXT_SIZE")
    pyannote_auth_token: str | None = Field(default=None, validation_alias="PYANNOTE_AUTH_TOKEN")
    pyannote_model: str = Field(default="pyannote/speaker-diarization-3.1", validation_alias="PYANNOTE_MODEL")
    pyannote_use_local_config: bool = Field(default=True, validation_alias="PYANNOTE_USE_LOCAL_CONFIG")
    pipeline_worker_queue: str = Field(default="cpu", validation_alias="PIPELINE_WORKER_QUEUE")
    pipeline_embedded_workers_enabled: bool = Field(default=True, validation_alias="PIPELINE_EMBEDDED_WORKERS_ENABLED")
    session_cookie_name: str = Field(default="ai_record_summary_session", validation_alias="SESSION_COOKIE_NAME")
    session_ttl_days: int = Field(default=14, ge=1, le=90, validation_alias="SESSION_TTL_DAYS")
    session_cookie_secure: bool = Field(default=False, validation_alias="SESSION_COOKIE_SECURE")
    bootstrap_admin_email: str = Field(default="admin@local.test", validation_alias="BOOTSTRAP_ADMIN_EMAIL")
    bootstrap_admin_password: str = Field(default="change-me-now", validation_alias="BOOTSTRAP_ADMIN_PASSWORD")
    bootstrap_workspace_name: str = Field(default="默认工作区", validation_alias="BOOTSTRAP_WORKSPACE_NAME")

    @model_validator(mode="after")
    def validate_hybrid_retrieval_settings(self) -> Self:
        if self.rag_vector_weight == 0 and self.rag_lexical_weight == 0:
            raise ValueError("RAG vector and lexical weights cannot both be zero")
        if self.rag_fused_candidate_limit > self.rag_vector_candidate_limit + self.rag_lexical_candidate_limit:
            raise ValueError("RAG fused candidate limit cannot exceed the sum of branch candidate limits")
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
        return self._resolve_repository_path(self.asr_lab_training_python_bin)

    @property
    def resolved_asr_lab_training_script(self) -> Path:
        return self._resolve_repository_path(self.asr_lab_training_script)

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
        return REPOSITORY_ROOT / "model-cache/local-llm" / self.local_llm_model_repo.replace("/", "__") / model_file

    @property
    def resolved_llm_correction_model_path(self) -> Path:
        model_file = self.llm_correction_model_file.split(",", maxsplit=1)[0].strip()
        if not model_file:
            raise ValueError("LLM_CORRECTION_MODEL_FILE must contain a model filename")
        return REPOSITORY_ROOT / "model-cache/llm-correction" / self.llm_correction_model_repo.replace("/", "__") / model_file

    @staticmethod
    def _resolve_repository_path(value: Path) -> Path:
        return value if value.is_absolute() else (REPOSITORY_ROOT / value).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]  # Values are supplied by the configured root .env source.
