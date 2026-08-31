from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import Engine

from l1_foundation.settings import Settings
from l2_core.auth.contracts import CurrentUser
from l2_core.rag_evaluation.service import RagEvaluationService


class FakeMappingsResult:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def mappings(self) -> FakeMappingsResult:
        return self

    def __iter__(self) -> Iterator[dict[str, object]]:
        return iter(self._rows)


class FakeConnection:
    def __init__(self) -> None:
        self.executions: list[tuple[str, dict[str, object]]] = []

    def execute(self, statement: object, parameters: dict[str, object]) -> FakeMappingsResult:
        self.executions.append((str(statement), parameters))
        return FakeMappingsResult([])


class FakeConnectionContext(AbstractContextManager[FakeConnection]):
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection

    def __enter__(self) -> FakeConnection:
        return self._connection

    def __exit__(self, *args: object) -> None:
        return None


class FakeEngine:
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection

    def connect(self) -> FakeConnectionContext:
        return FakeConnectionContext(self._connection)


def test_search_chunks_prioritizes_exact_normalized_keyword_matches() -> None:
    connection = FakeConnection()
    service = RagEvaluationService(cast(Engine, cast(Any, FakeEngine(connection))), cast(Settings, object()))
    user = CurrentUser(
        id=uuid4(),
        email="user@example.com",
        display_name="Test User",
        current_workspace_id=uuid4(),
        memberships=(),
    )

    assert service.search_chunks(user, query="  \uff21\uff30\uff29 \u7248\u672c  ") == []

    sql, values = connection.executions[0]
    assert "chunks.normalized_original_text" in sql
    assert "or word_similarity(:query, coalesce" in sql
    assert "order by (position(:query in coalesce" in sql
    assert values["query"] == "api \u7248\u672c"
