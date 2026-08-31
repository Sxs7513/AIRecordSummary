import asyncio
import json
from pathlib import Path
from typing import Any, cast

from l1_foundation.llm import (
    LlmGenerateBatchItemResult,
    LlmGenerateBatchResult,
    LlmGenerateResult,
    LlmProvider,
)
from l1_foundation.settings import REPOSITORY_ROOT
from l1_foundation.worker import SyncWorkerClient
from l2_core.audio_processing.stages.correct_text import LocalTextCorrector
from l2_core.audio_processing.stages.correct_text.llm import LLM_CORRECTION_SYSTEM_PROMPT, LlmCorrector


class FakeWorkerClient:
    def __init__(self, text: str = "", provider: LlmProvider = LlmProvider.LOCAL) -> None:
        self.text = text
        self.provider = provider
        self.model_name = "qwen-7b.gguf" if provider == LlmProvider.LOCAL else "gemini-test"
        self.commands: list[Any] = []

    def execute(self, command: Any, *, result_type: type[LlmGenerateBatchResult], on_progress: Any = None) -> LlmGenerateBatchResult:
        self.commands.append(command)
        if on_progress is not None:
            on_progress(1.0, "done")
        return result_type(
            items=[
                LlmGenerateBatchItemResult(
                    item_id=item.item_id,
                    result=LlmGenerateResult(text=self.text, provider=self.provider, model=self.model_name),
                )
                for item in command.input.items
            ]
        )


def test_number_normalization_prompt_requires_contextual_model_judgment_without_examples() -> None:
    prompt = "\n".join(LLM_CORRECTION_SYSTEM_PROMPT)

    assert "只有上下文能够明确确认时才能修改" in prompt
    assert "不得改变数值、范围、精度或含义" in prompt
    assert "二零二六年" not in prompt
    assert "三极管" not in prompt


def test_correction_prompt_allows_moderate_contextual_repairs_without_rewriting() -> None:
    prompt = "\n".join(LLM_CORRECTION_SYSTEM_PROMPT)

    assert "可以修复上下文能够明确支持的局部漏字、错词和语序问题" in prompt
    assert "不得总结、扩写、压缩或大范围重写" in prompt
    assert "无法确定是否需要修改时，保留原文" in prompt
    assert "不得为了简写而删除标点" in prompt
    assert "不使用公式排版或公式包裹" in prompt


def make_corrector(tmp_path: Path, *, pycorrector_enabled: bool = False, llm_enabled: bool = False) -> LocalTextCorrector:
    prompt_config = tmp_path / "prompt.json"
    prompt_config.write_text(
        '{"hotwords":["Qwen"],"replacements":{"错字":"正字"},"regexReplacements":[{"pattern":" +","replace":" "}]}',
        encoding="utf-8",
    )
    return LocalTextCorrector(
        pycorrector_enabled=pycorrector_enabled,
        llm_enabled=llm_enabled,
        worker_client=cast(SyncWorkerClient, FakeWorkerClient()) if llm_enabled else None,
        llm_provider=LlmProvider.LOCAL if llm_enabled else None,
        llm_context_size=8192,
        llm_model_name="qwen-7b.gguf" if llm_enabled else None,
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


def test_local_llm_corrector_reports_each_completed_text(tmp_path: Path) -> None:
    client = FakeWorkerClient('{"items":[{"unit_index":0,"text":"第一段已润色"},{"unit_index":1,"text":"第二段已润色"}]}')
    corrector = LlmCorrector(cast(SyncWorkerClient, client), LlmProvider.LOCAL, 8192, "qwen-7b.gguf", {})
    progress: list[tuple[int, int]] = []

    assert corrector.correct(["第一段", "第二段"], lambda completed, total: progress.append((completed, total))) == ["第一段已润色", "第二段已润色"]
    assert progress == [(0, 1), (1, 1)]


def test_online_llm_corrector_sends_the_full_transcript_in_one_request() -> None:
    source = [f"第{index}段" + "原文" * 40 for index in range(20)]
    expected = [text + "。" for text in source]
    response = '{"items":[' + ",".join(
        f'{{"unit_index":{index},"text":{json.dumps(text, ensure_ascii=False)}}}'
        for index, text in enumerate(expected)
    ) + "]}"
    client = FakeWorkerClient(response, LlmProvider.GEMINI)
    corrector = LlmCorrector(
        cast(SyncWorkerClient, client),
        LlmProvider.GEMINI,
        131_072,
        "gemini-3.5-flash-lite",
        {},
        batch_max_units=2,
        batch_max_chars=100,
        max_output_tokens=65_536,
    )

    assert corrector.correct(source) == expected
    assert len(client.commands) == 1
    requests = client.commands[0].input.items
    assert len(requests) == 1
    assert requests[0].request.options.max_tokens > 4096
    prompt = requests[0].request.messages[-1].content
    assert '"unit_index": 0' in prompt
    assert '"unit_index": 19' in prompt
    assert "完整录音文本" in prompt
    assert '"context_before"' not in prompt
    assert '"context_after"' not in prompt


def test_local_llm_corrector_keeps_configured_batch_limits() -> None:
    corrector = LlmCorrector(
        cast(SyncWorkerClient, FakeWorkerClient()),
        LlmProvider.LOCAL,
        8192,
        "qwen-7b.gguf",
        {},
        batch_max_units=2,
        batch_max_chars=100,
    )

    assert corrector._batches(["一", "二", "三", "四", "五"]) == [[0, 1], [2, 3], [4]]  # pyright: ignore[reportPrivateUsage]


def test_text_polishing_only_reports_model_loading_for_local_provider(tmp_path: Path) -> None:
    prompt_config = tmp_path / "prompt.json"
    prompt_config.write_text("{}", encoding="utf-8")
    response = '{"items":[{"unit_index":0,"text":"已润色"}]}'

    async def progress_messages(provider: LlmProvider) -> list[str]:
        corrector = LocalTextCorrector(
            pycorrector_enabled=False,
            llm_enabled=True,
            worker_client=cast(SyncWorkerClient, FakeWorkerClient(response, provider)),
            llm_provider=provider,
            llm_context_size=8192,
            llm_model_name="qwen-7b.gguf" if provider == LlmProvider.LOCAL else "gemini-test",
            prompt_config_path=prompt_config,
        )
        progress: list[tuple[int, str]] = []
        await corrector.correct(["原文"], lambda percent, message: progress.append((percent, message)))
        return [message for _percent, message in progress]

    local_messages = asyncio.run(progress_messages(LlmProvider.LOCAL))
    online_messages = asyncio.run(progress_messages(LlmProvider.GEMINI))

    assert "加载文本润色模型" in local_messages
    assert "开始文本润色" in online_messages
    assert all("加载" not in message for message in online_messages)


def test_local_llm_prompt_uses_hotwords_and_code_owned_protocol(tmp_path: Path) -> None:
    corrector = LlmCorrector(
        cast(SyncWorkerClient, FakeWorkerClient()),
        LlmProvider.LOCAL,
        8192,
        "qwen-7b.gguf",
        {
            "hotwords": ["硅光"],
            "phrases": ["固定表达"],
            "people": ["张三"],
        },
    )

    messages = corrector._build_messages(  # pyright: ignore[reportPrivateUsage]
        ["前文", "原始文本", "后文"],
        ["Speaker A", "Speaker B", "Speaker A"],
        [1],
    )
    prompt = "\n".join(message.content for message in messages)

    assert "专业词表：硅光" in prompt
    assert "固定表达：固定表达" in prompt
    assert "人名：张三" in prompt
    assert "均来自同一份录音" in prompt
    assert "可能跨越不同说话人" in prompt
    assert "不要把窗口重叠误判为口吃或无意义重复" in prompt
    assert '"context_before": [{"unit_index": 0' in prompt
    assert '"items": [{"unit_index": 1' in prompt
    assert '"context_after": [{"unit_index": 2' in prompt
    assert "即使无需修改也必须原样返回" in prompt


def test_local_text_corrector_falls_back_when_optional_providers_fail(tmp_path: Path) -> None:
    corrector = make_corrector(tmp_path, pycorrector_enabled=True, llm_enabled=True)

    assert asyncio.run(corrector.correct(["错字"])) == ["正字"]


def test_backend_stages_do_not_execute_legacy_lib_scripts() -> None:
    audio_processing_root = REPOSITORY_ROOT / "backend/packages/l2_core/audio_processing"
    qwen_stage = (audio_processing_root / "stages/transcribe_qwen_asr/__init__.py").read_text(encoding="utf-8")
    text_stage = (audio_processing_root / "stages/correct_text/__init__.py").read_text(encoding="utf-8")

    assert "lib/audio-transcoding-analysis" not in qwen_stage
    assert "lib/audio-transcoding-analysis" not in text_stage
    assert "asr_inference_batch_command" in qwen_stage
    assert "_worker_client.execute(" in qwen_stage
