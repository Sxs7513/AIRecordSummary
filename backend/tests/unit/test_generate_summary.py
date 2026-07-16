from __future__ import annotations

from pathlib import Path

from audio_processing.stages.recording_models import Utterance
from audio_processing.stages.summary.stage import GenerateSummaryStage
from pipeline.runtime.artifact_store import ArtifactStore


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


def _stage(
    tmp_path: Path,
    *,
    rolling_enabled: bool = False,
    rolling_threshold_ms: int = 1_800_000,
    rolling_chunk_duration_ms: int = 600_000,
    rolling_chunk_max_chars: int = 8000,
) -> GenerateSummaryStage:
    return GenerateSummaryStage(
        artifact_store=ArtifactStore(tmp_path),
        model_path=Path("missing.gguf"),
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
