from pathlib import Path

import pytest
from pydantic import ValidationError

from l1_foundation.settings import REPOSITORY_ROOT, Settings


def settings_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "DB_HOST": "localhost",
        "DB_PORT": 5432,
        "DB_USER": "postgres",
        "DB_PASSWORD": "postgres",
        "DB_NAME": "ai_record_summary",
        "DB_ADMIN_DATABASE": "postgres",
        "DB_SSL": False,
        "LOCAL_STORAGE_ROOT": "uploads",
    }
    payload.update(overrides)
    return payload


def test_qwen_asr_defaults() -> None:
    settings = Settings(_env_file=None, **settings_payload())

    assert settings.qwen_asr_model == "Qwen/Qwen3-ASR-1.7B"
    assert settings.qwen_asr_num_beams == 2
    assert settings.qwen_asr_tempo == 1.0
    assert settings.qwen_asr_enhance_low_volume_segments is True
    assert settings.qwen_asr_low_volume_rms_threshold == 0.01
    assert settings.qwen_asr_low_volume_peak_threshold == 0.08
    assert settings.qwen_asr_low_volume_max_gain_db == 9
    assert settings.asr_window_correction_max_edit_ratio == 0.35
    assert settings.pyannote_short_segment_absorb_max_gap_ms == 2_000
    assert settings.asr_lab_training_model == "Qwen/Qwen3-ASR-1.7B-hf"
    assert settings.asr_lab_training_module == "qwen_asr_lora"
    assert settings.resolved_asr_lab_training_python_bin == (
        REPOSITORY_ROOT / "backend/packages/l2_core/trainers/qwen-asr-lora/.venv/bin/python"
    ).absolute()
    assert ".venv" in settings.resolved_asr_lab_training_python_bin.parts
    assert settings.compute_worker_host == "127.0.0.1"
    assert settings.compute_worker_port == 8010
    assert settings.compute_worker_max_tasks == 100
    assert settings.processing_consumer_max_poll_interval_ms == 7_200_000
    assert not hasattr(settings, "compute_worker_enabled")


def test_asr_tempo_and_pyannote_absorb_gap_can_be_configured() -> None:
    settings = Settings(
        _env_file=None,
        **settings_payload(
            QWEN_ASR_TEMPO=0.9,
            QWEN_ASR_NUM_BEAMS=4,
            QWEN_ASR_ENHANCE_LOW_VOLUME_SEGMENTS=False,
            QWEN_ASR_LOW_VOLUME_RMS_THRESHOLD=0.006,
            QWEN_ASR_LOW_VOLUME_PEAK_THRESHOLD=0.03,
            QWEN_ASR_LOW_VOLUME_MAX_GAIN_DB=6,
            PYANNOTE_SHORT_SEGMENT_ABSORB_MAX_GAP_MS=1_500,
        ),
    )

    assert settings.qwen_asr_tempo == 0.9
    assert settings.qwen_asr_num_beams == 4
    assert settings.qwen_asr_enhance_low_volume_segments is False
    assert settings.qwen_asr_low_volume_rms_threshold == 0.006
    assert settings.qwen_asr_low_volume_peak_threshold == 0.03
    assert settings.qwen_asr_low_volume_max_gain_db == 6
    assert settings.pyannote_short_segment_absorb_max_gap_ms == 1_500


def test_database_url_encodes_credentials() -> None:
    settings = Settings(_env_file=None, **settings_payload(DB_USER="user@example", DB_PASSWORD="a/b"))

    assert "user%40example:a%2Fb@" in settings.sqlalchemy_database_url


def test_relative_storage_root_is_resolved_from_repository_root() -> None:
    settings = Settings(_env_file=None, **settings_payload(LOCAL_STORAGE_ROOT="uploads"))

    assert settings.resolved_local_storage_root == REPOSITORY_ROOT / Path("uploads")


def test_summary_defaults_to_large_context_without_rolling() -> None:
    settings = Settings(_env_file=None, **settings_payload())

    assert settings.llm_default_provider == "gemini"
    assert settings.llm_correction_provider == "gemini"
    assert settings.topic_detection_provider == "gemini"
    assert settings.recording_summary_provider == "gemini"
    assert settings.rag_online_default_model == "gemini-gemini-3.5-flash-lite"
    assert settings.rag_asr_adjudication_search_provider == "gemini"
    assert settings.rag_asr_adjudication_audit_prompt_variant == "relation_rules"
    assert settings.rag_asr_adjudication_audit_model == "gemini-gemini-3.5-flash-lite"
    assert settings.rag_asr_adjudication_construct_model == "gemini-gemini-3.5-flash-lite"
    assert settings.rag_asr_adjudication_decision_model == "gemini-gemini-3.5-flash-lite"
    assert settings.rag_asr_adjudication_audit_min_request_interval_seconds == 15
    assert settings.rag_asr_adjudication_search_model == "gemini-2.5-flash-lite"
    assert settings.rag_asr_adjudication_chrome_aio_timeout_seconds == 45
    assert settings.rag_asr_adjudication_chrome_aio_poll_interval_seconds == 1
    assert settings.gemini_model == "gemini-3.5-flash-lite"
    assert settings.gemini_min_request_interval_seconds == 5
    assert settings.qwen_llm_model == "qwen3.8-flash"
    assert settings.qwen_llm_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert settings.qwen_llm_min_request_interval_seconds == 0
    assert settings.llm_correction_max_output_tokens == 65_536
    assert settings.text_correction_context_units == 1
    assert settings.local_llm_model_repo == "Qwen/Qwen2.5-7B-Instruct-GGUF"
    assert settings.recording_summary_context_size == 262_144
    assert settings.recording_summary_rolling_enabled is False


def test_adjudication_audit_prompt_variant_can_select_free_discovery() -> None:
    settings = Settings(
        _env_file=None,
        **settings_payload(RAG_ASR_ADJUDICATION_AUDIT_PROMPT_VARIANT="free_discovery"),
    )

    assert settings.rag_asr_adjudication_audit_prompt_variant == "free_discovery"

    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            **settings_payload(RAG_ASR_ADJUDICATION_AUDIT_PROMPT_VARIANT="unknown"),
        )


def test_adjudication_agent_models_can_be_overridden() -> None:
    settings = Settings(
        _env_file=None,
        **settings_payload(
            RAG_ASR_ADJUDICATION_AUDIT_MODEL="qwen-qwen3.8-max",
            RAG_ASR_ADJUDICATION_CONSTRUCT_MODEL="qwen-qwen3.8-max",
            RAG_ASR_ADJUDICATION_DECISION_MODEL="qwen-qwen3.8-max",
            RAG_ASR_ADJUDICATION_AUDIT_MIN_REQUEST_INTERVAL_SECONDS=12,
        ),
    )

    assert settings.rag_asr_adjudication_audit_model == "qwen-qwen3.8-max"
    assert settings.rag_asr_adjudication_construct_model == "qwen-qwen3.8-max"
    assert settings.rag_asr_adjudication_decision_model == "qwen-qwen3.8-max"
    assert settings.rag_asr_adjudication_audit_min_request_interval_seconds == 12


def test_default_llm_provider_can_select_zhipu_without_changing_the_rag_model() -> None:
    settings = Settings(_env_file=None, **settings_payload(LLM_DEFAULT_PROVIDER="zhipu"))

    assert settings.llm_correction_provider == "zhipu"
    assert settings.topic_detection_provider == "zhipu"
    assert settings.recording_summary_provider == "zhipu"
    assert settings.rag_online_default_model == "gemini-gemini-3.5-flash-lite"


def test_default_llm_provider_can_select_qwen_without_changing_the_rag_model() -> None:
    settings = Settings(
        _env_file=None,
        **settings_payload(
            LLM_DEFAULT_PROVIDER="qwen",
            QWEN_AI_PLATFORM_API_KEY="test-key",
        ),
    )

    assert settings.llm_correction_provider == "qwen"
    assert settings.topic_detection_provider == "qwen"
    assert settings.recording_summary_provider == "qwen"
    assert settings.rag_online_default_model == "gemini-gemini-3.5-flash-lite"
    assert settings.qwen_ai_platform_api_key == "test-key"


def test_rag_online_models_require_provider_qualified_online_references() -> None:
    with pytest.raises(ValidationError, match="<provider>-<model>"):
        Settings(_env_file=None, **settings_payload(RAG_ONLINE_DEFAULT_MODEL="gemini"))

    with pytest.raises(ValidationError, match="provider must be one of"):
        Settings(_env_file=None, **settings_payload(RAG_ASR_ADJUDICATION_AUDIT_MODEL="local-qwen3-4b"))

    with pytest.raises(ValidationError, match="<provider>-<model>"):
        Settings(_env_file=None, **settings_payload(RAG_ASR_ADJUDICATION_CONSTRUCT_MODEL="qwen"))

    with pytest.raises(ValidationError, match="provider must be one of"):
        Settings(_env_file=None, **settings_payload(RAG_ASR_ADJUDICATION_DECISION_MODEL="local-qwen3-4b"))


def test_hybrid_retrieval_defaults_are_enabled_and_bounded() -> None:
    settings = Settings(_env_file=None, **settings_payload())

    assert settings.rag_hybrid_search_enabled is True
    assert settings.rag_vector_candidate_limit == 30
    assert settings.rag_lexical_candidate_limit == 30
    assert settings.rag_fused_candidate_limit == 20
    assert settings.rag_rrf_k == 60
    assert settings.rag_original_vector_weight == 0.7
    assert settings.rag_expanded_vector_weight == 0.2
    assert settings.rag_lexical_weight == 0.1
    assert settings.rag_local_model_repo == "Qwen/Qwen3-4B-GGUF"
    assert settings.rag_route_model_profile == "default"
    assert settings.rag_node_model_profile == "default"
    assert settings.rag_plan_local_input_tokens == 4_000
    assert settings.rag_run_max_total_tokens == 50_000
    assert settings.rag_rerank_model == "Qwen/Qwen3-Reranker-0.6B"
    assert settings.rag_rerank_max_total_tokens == 16_000
    assert settings.rag_rerank_output_limit == 8


def test_rag_route_model_profile_can_switch_between_7b_and_4b() -> None:
    settings = Settings(_env_file=None, **settings_payload(RAG_ROUTE_MODEL_PROFILE="rag"))

    assert settings.rag_route_model_profile == "rag"

    with pytest.raises(ValidationError):
        Settings(_env_file=None, **settings_payload(RAG_ROUTE_MODEL_PROFILE="unknown"))


def test_rag_node_model_profile_can_switch_between_7b_and_4b() -> None:
    settings = Settings(_env_file=None, **settings_payload(RAG_NODE_MODEL_PROFILE="rag"))

    assert settings.rag_node_model_profile == "rag"

    with pytest.raises(ValidationError):
        Settings(_env_file=None, **settings_payload(RAG_NODE_MODEL_PROFILE="unknown"))


def test_hybrid_retrieval_rejects_all_zero_weights() -> None:
    with pytest.raises(ValidationError, match="cannot all be zero"):
        Settings(
            _env_file=None,
            **settings_payload(
                RAG_ORIGINAL_VECTOR_WEIGHT=0,
                RAG_EXPANDED_VECTOR_WEIGHT=0,
                RAG_LEXICAL_WEIGHT=0,
            ),
        )
