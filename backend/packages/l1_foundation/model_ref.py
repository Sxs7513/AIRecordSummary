from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

type OnlineProviderName = Literal["zhipu", "gemini", "qwen"]

_ONLINE_PROVIDERS = frozenset({"zhipu", "gemini", "qwen"})


@dataclass(frozen=True, slots=True)
class OnlineModelRef:
    provider: OnlineProviderName
    model: str

    @classmethod
    def parse(cls, value: str) -> OnlineModelRef:
        normalized = value.strip()
        provider, separator, model = normalized.partition("-")
        if not separator or not provider or not model:
            raise ValueError("online model must use '<provider>-<model>' format")
        if provider not in _ONLINE_PROVIDERS:
            supported = ", ".join(sorted(_ONLINE_PROVIDERS))
            raise ValueError(f"online model provider must be one of: {supported}")
        return cls(provider=cast(OnlineProviderName, provider), model=model)

    def __str__(self) -> str:
        return f"{self.provider}-{self.model}"
