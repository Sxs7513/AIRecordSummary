import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from l1_foundation.settings import REPOSITORY_ROOT
from l2_core.audio_processing.stages.correct_text import LocalTextCorrector
from l2_core.audio_processing.stages.correct_text.local_llm import LlamaModel, LocalLlmCorrector


def make_corrector(tmp_path: Path, *, pycorrector_enabled: bool = False, llm_enabled: bool = False) -> LocalTextCorrector:
    prompt_config = tmp_path / "prompt.json"
    prompt_config.write_text(
        '{"hotwords":["Qwen"],"replacements":{"错字":"正字"},"regexReplacements":[{"pattern":" +","replace":" "}]}',
        encoding="utf-8",
    )
    return LocalTextCorrector(
        repository_root=REPOSITORY_ROOT,
        pycorrector_enabled=pycorrector_enabled,
        llm_enabled=llm_enabled,
        llm_model_repo="Qwen/Qwen2.5-7B-Instruct-GGUF",
        llm_model_file="model.gguf",
        llm_context_size=1024,
        prompt_config_path=prompt_config,
    )


def test_local_text_corrector_applies_configured_rules_without_models(tmp_path: Path) -> None:
    corrector = make_corrector(tmp_path)

    assert asyncio.run(corrector.correct(["  Qwen  错字  "])) == ["Qwen 正字"]


def test_local_text_corrector_reports_correction_phases(tmp_path: Path) -> None:
    corrector = make_corrector(tmp_path)
    progress: list[tuple[int, str]] = []

    assert asyncio.run(corrector.correct(["错字"], lambda percent, message: progress.append((percent, message)))) == ["正字"]

    assert [percent for percent, _message in progress] == [5, 35, 40, 95]
    assert progress[-1][1] == "文本润色完成"


def test_local_llm_corrector_reports_each_completed_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def __call__(self, prompt: str, **kwargs: object) -> Mapping[str, object]:
            return {"choices": [{"text": '{"items":[{"unit_index":0,"text":"第一段已润色"},{"unit_index":1,"text":"第二段已润色"}]}'}]}

    corrector = LocalLlmCorrector(tmp_path / "unused.gguf", 1024, {})
    monkeypatch.setattr(corrector, "_load_model", lambda: cast(LlamaModel, FakeModel()))
    progress: list[tuple[int, int]] = []

    assert corrector.correct(["第一段", "第二段"], lambda completed, total: progress.append((completed, total))) == ["第一段已润色", "第二段已润色"]
    assert progress == [(0, 1), (1, 1)]


def test_local_llm_prompt_uses_hotwords_and_code_owned_protocol(tmp_path: Path) -> None:
    corrector = LocalLlmCorrector(
        tmp_path / "unused.gguf",
        1024,
        {
            "hotwords": ["硅光"],
            "phrases": ["固定表达"],
            "people": ["张三"],
        },
    )

    prompt = corrector._build_prompt(["原始文本"], ["Speaker A"], [0])

    assert "专业词表：硅光" in prompt
    assert "固定表达：固定表达" in prompt
    assert "人名：张三" in prompt
    assert '"unit_index":数字' in prompt
    assert "独立出现的“嗯”“好”“对”“不是”“可以”等短回复" in prompt
    assert "应尽量保持原有语序" in prompt


def test_local_text_corrector_falls_back_when_optional_providers_fail(tmp_path: Path) -> None:
    corrector = make_corrector(tmp_path, pycorrector_enabled=True, llm_enabled=True)

    assert asyncio.run(corrector.correct(["错字"])) == ["正字"]


def test_backend_stages_do_not_execute_legacy_lib_scripts() -> None:
    audio_processing_root = REPOSITORY_ROOT / "backend/packages/l2_core/audio_processing"
    qwen_stage = (audio_processing_root / "stages/transcribe_qwen_asr/__init__.py").read_text(encoding="utf-8")
    text_stage = (audio_processing_root / "stages/correct_text/__init__.py").read_text(encoding="utf-8")

    assert "lib/audio-transcoding-analysis" not in qwen_stage
    assert "lib/audio-transcoding-analysis" not in text_stage
    assert "asyncio.to_thread" in qwen_stage
