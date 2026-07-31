from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from l1_foundation.llm import (
    LlmGenerateResult,
    LlmProvider,
)
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l1_foundation.worker import SyncWorkerClient
from l2_core.audio_processing.stages.recording_models import Utterance
from l2_core.audio_processing.stages.summary.stage import GenerateSummaryStage


class FakeWorkerClient:
    def execute(self, command: Any, *, result_type: type[LlmGenerateResult]) -> LlmGenerateResult:
        provider = command.input.provider
        model_name = "qwen-7b.gguf" if provider == LlmProvider.LOCAL else "gemini-test"
        return result_type(text="summary", provider=provider, model=model_name)


def test_rolling_summary_requires_explicit_enablement_and_duration_threshold(tmp_path: Path) -> None:
    stage = _stage(tmp_path, rolling_enabled=True, rolling_threshold_ms=1_000)

    assert stage.should_use_rolling_summary([_utterance(0, 500), _utterance(600, 1_500)]) is True
    assert stage.should_use_rolling_summary([_utterance(0, 999)]) is False
    assert _stage(tmp_path, rolling_enabled=False, rolling_threshold_ms=1).should_use_rolling_summary([_utterance(0, 10_000)]) is False


def test_rolling_summary_builds_chunks_by_duration_and_text_budget(tmp_path: Path) -> None:
    stage = _stage(tmp_path, rolling_enabled=True, rolling_chunk_duration_ms=1_500, rolling_chunk_max_chars=10_000)

    chunks = stage.build_rolling_chunks([_utterance(0, 600), _utterance(700, 1_200), _utterance(1_300, 1_800)])

    assert [len(chunk.utterances) for chunk in chunks] == [2, 1]
    assert [chunk.index for chunk in chunks] == [1, 2]


def test_summary_only_reports_model_loading_for_local_provider(tmp_path: Path) -> None:
    def progress_messages(provider: LlmProvider) -> list[str]:
        stage = _stage(tmp_path, provider=provider)
        progress: list[tuple[int, str]] = []
        stage.generate([_utterance(0, 500)], lambda percent, message: progress.append((percent, message)))
        return [message for _percent, message in progress]

    local_messages = progress_messages(LlmProvider.LOCAL)
    online_messages = progress_messages(LlmProvider.GEMINI)

    assert "加载总结模型" in local_messages
    assert any("token 大上下文" in message for message in local_messages)
    assert "开始生成录音总结" in online_messages
    assert all("加载" not in message for message in online_messages)
    assert all("token 大上下文" not in message for message in online_messages)


def test_large_context_log_is_only_emitted_for_local_provider(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO", logger="audio_processing"):
        _stage(tmp_path, provider=LlmProvider.LOCAL).generate([_utterance(0, 500)], lambda _percent, _message: None)
    assert "token 大上下文单次总结" in caplog.text

    caplog.clear()
    with caplog.at_level("INFO", logger="audio_processing"):
        _stage(tmp_path, provider=LlmProvider.GEMINI).generate([_utterance(0, 500)], lambda _percent, _message: None)
    assert "token 大上下文单次总结" not in caplog.text
    assert "使用在线模型单次总结 provider=gemini" in caplog.text


def _stage(
    tmp_path: Path,
    *,
    provider: LlmProvider = LlmProvider.LOCAL,
    rolling_enabled: bool = False,
    rolling_threshold_ms: int = 1_800_000,
    rolling_chunk_duration_ms: int = 600_000,
    rolling_chunk_max_chars: int = 8000,
) -> GenerateSummaryStage:
    return GenerateSummaryStage(
        artifact_store=ArtifactStore(tmp_path),
        worker_client=cast(SyncWorkerClient, FakeWorkerClient()),
        provider=provider,
        model_name="qwen-7b.gguf" if provider == LlmProvider.LOCAL else "gemini-test",
        context_size=262_144,
        prompt_config_path=Path("missing.json"),
        rolling_enabled=rolling_enabled,
        rolling_threshold_ms=rolling_threshold_ms,
        rolling_chunk_duration_ms=rolling_chunk_duration_ms,
        rolling_chunk_max_chars=rolling_chunk_max_chars,
    )


def _utterance(start_ms: int, end_ms: int) -> Utterance:
    return Utterance(
        utterance_index=start_ms,
        start_ms=start_ms,
        end_ms=end_ms,
        text="测试文本",
        speaker_cluster_id="speaker-1",
        speaker_label="Speaker A",
        source_diarization_segment_ids=[],
    )
