from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import Engine

from l2_core.audio_processing.contracts import RecordingId
from l2_core.audio_processing.projections import RecordingProjectionService
from l2_core.audio_processing.stages.recording_models import TranscriptOutput, TranscriptSegment


class FakeResult:
    def __init__(self, scalar: UUID | None = None) -> None:
        self._scalar = scalar

    def scalar_one(self) -> UUID:
        assert self._scalar is not None
        return self._scalar

    def scalar_one_or_none(self) -> UUID | None:
        return self._scalar


class FakeConnection:
    def __init__(self, transcription_id: UUID) -> None:
        self.transcription_id = transcription_id
        self.executions: list[tuple[str, Mapping[str, object]]] = []

    def execute(self, statement: object, parameters: Mapping[str, object]) -> FakeResult:
        sql = str(statement)
        self.executions.append((sql, parameters))
        return FakeResult(self.transcription_id if "insert into transcriptions" in sql else None)


class FakeTransaction(AbstractContextManager[FakeConnection]):
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def __enter__(self) -> FakeConnection:
        return self.connection

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return None


class FakeEngine:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def begin(self) -> FakeTransaction:
        return FakeTransaction(self.connection)


def test_transcript_projection_persists_original_full_text_without_changing_polished_text() -> None:
    connection = FakeConnection(uuid4())
    service = RecordingProjectionService(cast(Engine, cast(Any, FakeEngine(connection))))
    output = TranscriptOutput(
        provider="qwen_asr",
        model_name="Qwen/Qwen3-ASR-1.7B",
        language="Chinese",
        segments=[
            TranscriptSegment(
                source_diarization_segment_id="Speaker A:0:1000",
                start_ms=0,
                end_ms=1000,
                text="润色文本",
                original_text="ASR 原文",
                speaker_cluster_id="Speaker A",
                speaker_label="Speaker A",
            )
        ],
        original_full_text="第一段原文\n第二段原文",
    )

    service.project(RecordingId(uuid4()), "align_transcript", output)

    parameters = next(parameters for sql, parameters in connection.executions if "insert into transcriptions" in sql)
    assert parameters["full_text"] == "润色文本"
    assert parameters["original_full_text"] == "第一段原文\n第二段原文"
    segment_parameters = next(parameters for sql, parameters in connection.executions if "insert into transcription_segments" in sql)
    assert segment_parameters["text"] == "润色文本"
    assert segment_parameters["original_text"] == "ASR 原文"
