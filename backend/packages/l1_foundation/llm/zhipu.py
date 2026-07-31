from __future__ import annotations

from l1_foundation.llm.contracts import LlmProvider
from l1_foundation.llm.openai_compatible import OpenAiCompatibleLanguageModel


class ZhipuLanguageModel(OpenAiCompatibleLanguageModel):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__(
            provider=LlmProvider.ZHIPU,
            provider_label="Zhipu",
            api_key_name="ZHIPU_API_KEY",
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            strict_json_schema=False,
            max_temperature=1,
        )

