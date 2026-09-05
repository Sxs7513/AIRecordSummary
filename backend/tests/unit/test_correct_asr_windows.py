import asyncio
from pathlib import Path
from typing import cast
from uuid import uuid4

from l1_foundation.pipeline.contracts import ArtifactPayload, PipelineRunId, PipelineSubjectId, StageContext, StageRunId
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l2_core.audio_processing.stages.correct_text import CorrectAsrWindowsStage, LocalTextCorrector
from l2_core.audio_processing.stages.recording_models import (
    AsrWindowTranscript,
    AsrWindowTranscriptOutput,
    CorrectAsrWindowsInput,
)


class FakeCorrector:
    async def correct(self, texts: list[str], *_args: object) -> list[str]:
        return [f"{text}。" for text in texts]

    def release(self) -> None:
        return None


def test_correct_asr_windows_preserves_qwen_source_metadata(tmp_path: Path) -> None:
    storage = ArtifactStore(tmp_path)
    context = StageContext(PipelineSubjectId(uuid4()), PipelineRunId(uuid4()), StageRunId(uuid4()), 1)
    raw = AsrWindowTranscriptOutput(
        provider="qwen_asr",
        model_name="Qwen/Qwen3-ASR-1.7B",
        language="Chinese",
        windows=[
            AsrWindowTranscript(
                window_index=0,
                input_start_ms=0,
                input_end_ms=10_500,
                core_start_ms=0,
                core_end_ms=10_000,
                language="Chinese",
                text="测试文本",
                core_diarization_segment_ids=["A:0:10000"],
            )
        ],
    )
    artifact = storage.write_json(
        context.subject_id,
        context.pipeline_run_id,
        context.stage_run_id,
        "transcribe_qwen_asr",
        ArtifactPayload(artifact_type="transcript.asr_windows", data=raw.model_dump(mode="json")),
    )
    stage = CorrectAsrWindowsStage(
        storage,
        cast(LocalTextCorrector, FakeCorrector()),
        "rules",
        None,
    )

    result = asyncio.run(stage.run(context, CorrectAsrWindowsInput(transcript=artifact)))

    assert result.output.asr_provider == "qwen_asr"
    assert result.output.asr_model_name == "Qwen/Qwen3-ASR-1.7B"
    assert result.output.language == "Chinese"
    assert result.output.windows[0].text == "测试文本。"


def test_correct_asr_windows_normalizes_inline_latex_to_plain_text() -> None:
    assert CorrectAsrWindowsStage._normalize_latex_notation("参数为 $\\Delta T_{1}$ 和 $\\Delta T_{2}$。") == "参数为 Δ T_1 和 Δ T_2。"  # pyright: ignore[reportPrivateUsage]
