from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import Engine

from l2_core.audio_processing.contracts import RecordingId
from l2_core.audio_processing.projections import RecordingProjectionService
from l2_core.audio_processing.stages.recording_models import EmbeddedSearchChunk, EmbeddingIndexingOutput


class FakeResult:
    def __init__(self, scalar: object | None = None, rows: list[object] | None = None) -> None:
        self._scalar = scalar
        self._rows = rows or []

    def scalar_one(self) -> object:
        assert self._scalar is not None
        return self._scalar

    def scalars(self) -> Iterator[object]:
        return iter(self._rows)


class FakeConnection:
    def __init__(self, embedding_model_id: UUID) -> None:
        self.embedding_model_id = embedding_model_id
        self.executions: list[tuple[str, Mapping[str, object]]] = []

    def execute(self, statement: object, parameters: Mapping[str, object]) -> FakeResult:
        sql = str(statement)
        self.executions.append((sql, parameters))
        if "insert into embedding_models" in sql:
            return FakeResult(scalar=self.embedding_model_id)
        return FakeResult()


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


def test_embedding_projection_replaces_chunks_idempotently() -> None:
    recording_id = RecordingId(uuid4())
    embedding_model_id = uuid4()
    connection = FakeConnection(embedding_model_id)
    engine = cast(Engine, cast(Any, FakeEngine(connection)))
    service = RecordingProjectionService(engine)
    output = EmbeddingIndexingOutput(
        provider="sentence_transformers",
        model_name="Qwen/Qwen3-Embedding-4B",
        dimensions=3,
        chunks=[
            EmbeddedSearchChunk(
                chunk_index=0,
                text="  Speaker A:  测试内容  ",
                start_ms=100,
                end_ms=200,
                speaker_labels=["Speaker A"],
                speaker_cluster_ids=["speaker-0"],
                source_utterance_indexes=[],
                source_diarization_segment_ids=[],
                topic="公司营收",
                terms=["营收", "收入"],
                search_context="询问公司目前达到的营收规模",
                embedding=[0.1, 0.2, 0.3],
            )
        ],
    )

    service.project(recording_id, "embedding_indexing", output)
    service.project(recording_id, "embedding_indexing", output)

    deletes = [parameters for sql, parameters in connection.executions if "delete from recording_search_chunks" in sql]
    inserts = [parameters for sql, parameters in connection.executions if "insert into recording_search_chunks" in sql]
    assert deletes == [{"recording_id": recording_id}, {"recording_id": recording_id}]
    assert len(inserts) == 2
    assert inserts[0]["embedding_model_id"] == embedding_model_id
    assert inserts[0]["normalized_text"] == "主题:公司营收 标准术语:营收、收入 语义上下文:询问公司目前达到的营收规模 正文:speaker a: 测试内容"
    metadata = json.loads(cast(str, inserts[0]["metadata"]))
    assert metadata["topic"] == "公司营收"
    assert metadata["terms"] == ["营收", "收入"]
    assert metadata["search_context"] == "询问公司目前达到的营收规模"
    assert inserts[0]["embedding"] == "[0.1,0.2,0.3]"
