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


def test_default_asr_provider_is_qwen() -> None:
    settings = Settings(_env_file=None, **settings_payload())

    assert settings.asr_provider == "qwen_asr"
    assert settings.qwen_asr_model == "Qwen/Qwen3-ASR-1.7B"
    assert settings.asr_lab_training_model == "Qwen/Qwen3-ASR-1.7B-hf"
    assert settings.asr_lab_training_module == "qwen_asr_lora"
    assert settings.resolved_asr_lab_training_python_bin == (
        REPOSITORY_ROOT / "backend/packages/l2_core/trainers/qwen-asr-lora/.venv/bin/python"
    ).absolute()
    assert ".venv" in settings.resolved_asr_lab_training_python_bin.parts


def test_funasr_nano_can_be_selected_as_the_recording_asr_provider() -> None:
    settings = Settings(_env_file=None, **settings_payload(ASR_PROVIDER="funasr_nano"))

    assert settings.asr_provider == "funasr_nano"
    assert settings.funasr_nano_model == "FunAudioLLM/Fun-ASR-Nano-2512"
    assert settings.resolved_huggingface_hub_cache_dir == REPOSITORY_ROOT / "model-cache" / "huggingface" / "hub"
    assert settings.resolved_funasr_nano_cache_dir == REPOSITORY_ROOT / "model-cache" / "huggingface" / "hub"


def test_database_url_encodes_credentials() -> None:
    settings = Settings(_env_file=None, **settings_payload(DB_USER="user@example", DB_PASSWORD="a/b"))

    assert "user%40example:a%2Fb@" in settings.sqlalchemy_database_url


def test_relative_storage_root_is_resolved_from_repository_root() -> None:
    settings = Settings(_env_file=None, **settings_payload(LOCAL_STORAGE_ROOT="uploads"))

    assert settings.resolved_local_storage_root == REPOSITORY_ROOT / Path("uploads")


def test_summary_defaults_to_large_context_without_rolling() -> None:
    settings = Settings(_env_file=None, **settings_payload())

    assert settings.recording_summary_context_size == 262_144
    assert settings.recording_summary_rolling_enabled is False


def test_hybrid_retrieval_defaults_are_enabled_and_bounded() -> None:
    settings = Settings(_env_file=None, **settings_payload())

    assert settings.rag_hybrid_search_enabled is True
    assert settings.rag_vector_candidate_limit == 30
    assert settings.rag_lexical_candidate_limit == 30
    assert settings.rag_fused_candidate_limit == 20
    assert settings.rag_rrf_k == 60


def test_hybrid_retrieval_rejects_two_zero_weights() -> None:
    with pytest.raises(ValidationError, match="cannot both be zero"):
        Settings(
            _env_file=None,
            **settings_payload(RAG_VECTOR_WEIGHT=0, RAG_LEXICAL_WEIGHT=0),
        )
