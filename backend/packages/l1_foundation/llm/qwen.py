from __future__ import annotations

from collections.abc import Sequence

from l1_foundation.llm.contracts import ChatMessage, CompletionOptions, JsonObject, LlmProvider
from l1_foundation.llm.openai_compatible import OpenAiCompatibleLanguageModel, SynchronousRequestRateLimiter


class QwenLanguageModel(OpenAiCompatibleLanguageModel):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout_seconds: float = 300.0,
        min_request_interval_seconds: float = 0,
        request_rate_limiter: SynchronousRequestRateLimiter | None = None,
    ) -> None:
        super().__init__(
            provider=LlmProvider.QWEN,
            provider_label="Qwen",
            api_key_name="QWEN_AI_PLATFORM_API_KEY",
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            strict_json_schema=True,
            max_temperature=2,
            stream_include_usage=True,
            min_request_interval_seconds=min_request_interval_seconds,
            request_rate_limiter=request_rate_limiter,
        )

    def _payload(self, messages: Sequence[ChatMessage], options: CompletionOptions, *, stream: bool) -> JsonObject:
        payload = super()._payload(messages, options, stream=stream)
        max_tokens = payload.pop("max_tokens")
        payload["max_completion_tokens"] = max_tokens
        payload["enable_thinking"] = False
        return payload
