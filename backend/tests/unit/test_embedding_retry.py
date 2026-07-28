from __future__ import annotations

import json
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, Self, cast
from uuid import uuid4

import pytest
from sqlalchemy import Engine

from l1_foundation.infrastructure.storage.local import LocalStorage
from l2_core.application.recordings import RecordingService, RecordingStageNotRetryableError


class FakeResult:
    def __init__(self, row: dict[str, object]) -> None:
        self._row = row

    def mappings(self) -> Self:
        return self

    def one_or_none(self) -> dict[str, object]:
        return self._row


class FakeConnection:
    def __init__(self, row: dict[str, object]) -> None:
        self._row = row

    def execute(self, _statement: object, _parameters: object) -> FakeResult:
        return FakeResult(self._row)


class FakeConnectionContext(AbstractContextManager[FakeConnection]):
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection

    def __enter__(self) -> FakeConnection:
        return self._connection

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return None


class FakeEngine:
    def __init__(self, row: dict[str, object]) -> None:
        self._connection = FakeConnection(row)

    def connect(self) -> FakeConnectionContext:
        return FakeConnectionContext(self._connection)


def test_embedding_retry_restores_deleted_search_chunks_from_stage_output(tmp_path: Path) -> None:
    recording_id = uuid4()
    pipeline_run_id = uuid4()
    producer_stage_run_id = uuid4()
    embedding_stage_run_id = uuid4()
    uri = f"artifacts/{recording_id}/{pipeline_run_id}/build_search_chunks/search_chunks.json"
    row: dict[str, object] = {
        "embedding_stage_run_id": embedding_stage_run_id,
        "embedding_stage_status": "succeeded",
        "producer_stage_run_id": producer_stage_run_id,
        "pipeline_run_id": pipeline_run_id,
        "output_payload": {
            "output": {
                "chunks": [
                    {
                        "chunk_index": 0,
                        "text": "Speaker A: 测试",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "speaker_labels": ["Speaker A"],
                        "speaker_cluster_ids": ["speaker-0"],
                        "source_utterance_indexes": [0],
                        "source_diarization_segment_ids": ["speaker-0:0:1000"],
                    }
                ]
            },
            "artifact_types": ["search.chunks"],
        },
        "uri": uri,
        "artifact_version": "1",
    }
    storage = LocalStorage(tmp_path / "uploads")
    storage.initialize()
    service = RecordingService(cast(Engine, cast(Any, FakeEngine(row))), storage)

    restored_stage_run_id, status = service._restore_embedding_retry_input(recording_id)

    assert restored_stage_run_id == embedding_stage_run_id
    assert status == "succeeded"
    restored = json.loads(storage.resolve(uri).read_text(encoding="utf-8"))
    assert restored["chunks"][0]["text"] == "Speaker A: 测试"


def test_embedding_retry_rejects_missing_file_without_durable_stage_output(tmp_path: Path) -> None:
    recording_id = uuid4()
    pipeline_run_id = uuid4()
    uri = f"artifacts/{recording_id}/{pipeline_run_id}/build_search_chunks/search_chunks.json"
    row: dict[str, object] = {
        "embedding_stage_run_id": uuid4(),
        "embedding_stage_status": "succeeded",
        "producer_stage_run_id": uuid4(),
        "pipeline_run_id": pipeline_run_id,
        "output_payload": None,
        "uri": uri,
        "artifact_version": "1",
    }
    storage = LocalStorage(tmp_path / "uploads")
    storage.initialize()
    service = RecordingService(cast(Engine, cast(Any, FakeEngine(row))), storage)

    with pytest.raises(RecordingStageNotRetryableError, match="没有可用于恢复"):
        service._restore_embedding_retry_input(recording_id)

    assert not storage.resolve(uri).exists()
