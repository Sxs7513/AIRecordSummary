from __future__ import annotations

from l1_foundation.llm.contracts import LlmProvider
from l1_foundation.llm.openai_compatible import OpenAiCompatibleLanguageModel


class GeminiLanguageModel(OpenAiCompatibleLanguageModel):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
        timeout_seconds: float = 300.0,
    ) -> None:
        super().__init__(
            provider=LlmProvider.GEMINI,
            provider_label="Gemini",
            api_key_name="GEMINI_API_KEY",
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            strict_json_schema=True,
            max_temperature=2,
            include_temperature=False,
            reasoning_effort="minimal",
            stream_include_usage=True,
            rate_limit_max_attempts=3,
            rate_limit_retry_seconds=10,
        )
