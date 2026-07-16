from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class QwenAsrPromptConfig(BaseModel):
    """The subset of the shared transcription prompt configuration used by Qwen ASR."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    hotwords: list[str] = Field(default_factory=list)
    phrases: list[str] = Field(default_factory=list)


def build_qwen_asr_context(config_path: Path, max_items: int, extra_context: str = "") -> str:
    """Build the ASR biasing context from the shared hotword configuration."""
    extra = extra_context.strip()
    try:
        config = QwenAsrPromptConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return extra

    if not config.enabled:
        return extra

    limit = max_items if max_items > 0 else None
    selected_hotwords = _take_unique(config.hotwords, limit)
    remaining = limit - len(selected_hotwords) if limit is not None else None
    selected_phrases = _take_unique(config.phrases, remaining)
    sections = [
        extra,
        "这是半导体、光电子、材料物理相关的技术行业讨论。以下内容是可能出现的专业术语、人名或固定表达，仅在语音和上下文匹配时优先采用，不要主动补充未出现的信息。",
        f"专业术语：{'、'.join(selected_hotwords)}" if selected_hotwords else "",
        f"固定表达：{'、'.join(selected_phrases)}" if selected_phrases else "",
    ]
    return "\n".join(section for section in sections if section)


def _take_unique(items: list[str], max_items: int | None) -> list[str]:
    """Trim blank and duplicate entries while preserving the config-file order."""
    selected: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        if max_items is not None and len(selected) >= max_items:
            break
        seen.add(normalized)
        selected.append(normalized)
    return selected
