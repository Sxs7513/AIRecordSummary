from __future__ import annotations

from pathlib import Path

from l1_foundation.llm.contracts import LanguageModel, LlmProvider
from l1_foundation.llm.gemini import GeminiLanguageModel
from l1_foundation.llm.local import LocalLlamaLanguageModel
from l1_foundation.llm.qwen import QwenLanguageModel
from l1_foundation.llm.zhipu import ZhipuLanguageModel
from l1_foundation.settings import Settings


def create_language_model(
    provider: LlmProvider,
    *,
    local_model_path: Path | None = None,
    local_context_size: int | None = None,
    local_verbose: bool = False,
    zhipu_api_key: str | None = None,
    zhipu_model: str | None = None,
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4",
    zhipu_timeout_seconds: float = 120.0,
    gemini_api_key: str | None = None,
    gemini_model: str | None = None,
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
    gemini_timeout_seconds: float = 300.0,
    gemini_min_request_interval_seconds: float = 5.0,
    qwen_ai_platform_api_key: str | None = None,
    qwen_llm_model: str | None = None,
    qwen_llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    qwen_llm_timeout_seconds: float = 300.0,
    qwen_llm_min_request_interval_seconds: float = 0,
) -> LanguageModel:
    if provider == LlmProvider.LOCAL:
        if local_model_path is None or local_context_size is None:
            raise ValueError("local_model_path and local_context_size are required for provider=local")
        return LocalLlamaLanguageModel(local_model_path, local_context_size, local_verbose)
    if provider == LlmProvider.ZHIPU:
        return ZhipuLanguageModel(
            api_key=zhipu_api_key or "",
            model=zhipu_model or "",
            base_url=zhipu_base_url,
            timeout_seconds=zhipu_timeout_seconds,
        )
    if provider == LlmProvider.GEMINI:
        return GeminiLanguageModel(
            api_key=gemini_api_key or "",
            model=gemini_model or "",
            base_url=gemini_base_url,
            timeout_seconds=gemini_timeout_seconds,
            min_request_interval_seconds=gemini_min_request_interval_seconds,
        )
    if provider == LlmProvider.QWEN:
        return QwenLanguageModel(
            api_key=qwen_ai_platform_api_key or "",
            model=qwen_llm_model or "",
            base_url=qwen_llm_base_url,
            timeout_seconds=qwen_llm_timeout_seconds,
            min_request_interval_seconds=qwen_llm_min_request_interval_seconds,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def create_language_model_from_settings(
    settings: Settings,
    provider: LlmProvider,
    *,
    local_context_size: int,
    local_model_profile: str = "default",
) -> LanguageModel:
    local_model_path = (
        settings.resolved_rag_local_model_path if local_model_profile == "rag" else settings.resolved_local_llm_model_path
    )
    return create_language_model(
        provider,
        local_model_path=local_model_path,
        local_context_size=local_context_size,
        local_verbose=settings.local_llm_verbose,
        zhipu_api_key=settings.zhipu_api_key,
        zhipu_model=settings.zhipu_model,
        zhipu_base_url=settings.zhipu_base_url,
        zhipu_timeout_seconds=settings.zhipu_timeout_seconds,
        gemini_api_key=settings.gemini_api_key,
        gemini_model=settings.gemini_model,
        gemini_base_url=settings.gemini_base_url,
        gemini_timeout_seconds=settings.gemini_timeout_seconds,
        gemini_min_request_interval_seconds=settings.gemini_min_request_interval_seconds,
        qwen_ai_platform_api_key=settings.qwen_ai_platform_api_key,
        qwen_llm_model=settings.qwen_llm_model,
        qwen_llm_base_url=settings.qwen_llm_base_url,
        qwen_llm_timeout_seconds=settings.qwen_llm_timeout_seconds,
        qwen_llm_min_request_interval_seconds=settings.qwen_llm_min_request_interval_seconds,
    )
