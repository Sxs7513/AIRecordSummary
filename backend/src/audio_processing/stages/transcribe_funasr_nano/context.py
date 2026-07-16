from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class FunAsrPromptConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    hotwords: list[str] = Field(default_factory=list)
    phrases: list[str] = Field(default_factory=list)


def build_funasr_hotwords(config_path: Path, max_items: int) -> list[str]:
    try:
        config = FunAsrPromptConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    if not config.enabled:
        return []
    values = list(dict.fromkeys(item.strip() for item in [*config.hotwords, *config.phrases] if item.strip()))
    return values[:max_items] if max_items > 0 else values
