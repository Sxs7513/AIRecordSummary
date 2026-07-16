from __future__ import annotations

import json
from pathlib import Path

from audio_processing.stages.transcribe_funasr_nano.context import build_funasr_hotwords
from audio_processing.stages.transcribe_qwen_asr.context import build_qwen_asr_context


def test_build_qwen_asr_context_uses_hotwords_before_phrases(tmp_path: Path) -> None:
    config_path = tmp_path / "prompt.json"
    config_path.write_text(
        json.dumps({"enabled": True, "hotwords": ["硅光", "硅光", "光开关"], "phrases": ["固定表达", "另一个表达"]}),
        encoding="utf-8",
    )

    context = build_qwen_asr_context(config_path, max_items=3, extra_context="会议主题：器件设计")

    assert "会议主题：器件设计" in context
    assert "专业术语：硅光、光开关" in context
    assert "固定表达：固定表达" in context
    assert "另一个表达" not in context


def test_build_qwen_asr_context_returns_extra_context_when_config_is_disabled(tmp_path: Path) -> None:
    config_path = tmp_path / "prompt.json"
    config_path.write_text('{"enabled": false, "hotwords": ["硅光"]}', encoding="utf-8")

    assert build_qwen_asr_context(config_path, max_items=200, extra_context="自定义术语") == "自定义术语"


def test_build_funasr_hotwords_uses_the_same_term_order_and_limit(tmp_path: Path) -> None:
    config_path = tmp_path / "prompt.json"
    config_path.write_text(
        json.dumps({"enabled": True, "hotwords": ["硅光", "硅光", "光开关"], "phrases": ["固定表达", "另一个表达"]}),
        encoding="utf-8",
    )

    assert build_funasr_hotwords(config_path, max_items=3) == ["硅光", "光开关", "固定表达"]
