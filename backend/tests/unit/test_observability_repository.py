from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

from repository import ObservabilityRepository
from sqlalchemy import Engine


class FakeResult:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def mappings(self) -> FakeResult:
        return self

    def __iter__(self) -> Any:
        return iter(self._rows)

    def one_or_none(self) -> dict[str, object] | None:
        return self._rows[0] if self._rows else None

    def one(self) -> dict[str, object]:
        return self._rows[0]


class FakeConnection:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows
        self.sql = ""
        self.parameters: dict[str, object] = {}

    def execute(self, statement: object, parameters: dict[str, object]) -> FakeResult:
        self.sql = str(statement)
        self.parameters = parameters
        return FakeResult(self._rows)


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


def test_run_list_returns_conversation_state_without_hiding_deleted_links() -> None:
    workspace_id = uuid4()
    user_id = uuid4()
    run_id = uuid4()
    conversation_id = uuid4()
    now = datetime.now(UTC)
    connection = FakeConnection(
        [
            {
                "generation_run_id": run_id,
                "conversation_id": conversation_id,
                "conversation_navigable": False,
                "conversation_deleted": True,
                "started_at": now,
                "finished_at": now,
                "invocation_count": 2,
                "failed_invocation_count": 0,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "run_status": "succeeded",
            }
        ]
    )
    repository = ObservabilityRepository(cast(Engine, cast(Any, FakeEngine(connection))))

    runs = repository.list_runs(workspace_id, user_id, now - timedelta(days=1), now + timedelta(seconds=1), 20, 0)

    assert runs[0]["conversation_id"] == conversation_id
    assert runs[0]["conversation_navigable"] is False
    assert runs[0]["conversation_deleted"] is True
    assert runs[0]["total_tokens"] == 120
    assert "conversation_messages messages" in connection.sql
    assert "left join lateral" in connection.sql
    assert "conversation.owner_user_id = :user_id" in connection.sql
    assert "operation = 'answer'" in connection.sql
    assert "status = 'succeeded'" in connection.sql
    assert "join completed_runs using (generation_run_id)" in connection.sql
    assert connection.parameters["workspace_id"] == workspace_id
    assert connection.parameters["user_id"] == user_id


class SequentialFakeConnection(FakeConnection):
    def __init__(self, result_sets: list[list[dict[str, object]]]) -> None:
        super().__init__([])
        self._result_sets = iter(result_sets)

    def execute(self, statement: object, parameters: dict[str, object]) -> FakeResult:
        self.sql += str(statement)
        self.parameters = parameters
        return FakeResult(next(self._result_sets))


def test_overview_and_token_p90_only_include_runs_with_completed_answers() -> None:
    workspace_id = uuid4()
    now = datetime.now(UTC)
    connection = SequentialFakeConnection(
        [
            [
                {
                    "run_count": 3,
                    "invocation_count": 12,
                    "failed_invocation_count": 1,
                    "prompt_tokens": 1000,
                    "completion_tokens": 250,
                    "average_invocation_elapsed_ms": 120.5,
                }
            ],
            [
                {
                    "operation": "answer",
                    "sample_run_count": 3,
                    "invocation_count": 3,
                    "prompt_tokens_p90": 500,
                    "completion_tokens_p90": 150,
                    "total_tokens_p90": 650,
                }
            ],
        ]
    )
    repository = ObservabilityRepository(cast(Engine, cast(Any, FakeEngine(connection))))

    overview = repository.overview(workspace_id, now - timedelta(days=7), now)

    assert overview["total_tokens"] == 1250
    assert overview["token_p90_by_operation"] == [
        {
            "operation": "answer",
            "sample_run_count": 3,
            "invocation_count": 3,
            "prompt_tokens_p90": 500,
            "completion_tokens_p90": 150,
            "total_tokens_p90": 650,
        }
    ]
    assert connection.sql.count("operation = 'answer'") == 2
    assert connection.sql.count("status = 'succeeded'") == 2
    assert "percentile_disc(0.9)" in connection.sql
    assert "usage_by_run_and_operation" in connection.sql


def test_deleted_conversation_snapshot_is_loaded_by_generation_run() -> None:
    workspace_id = uuid4()
    run_id = uuid4()
    conversation_id = uuid4()
    now = datetime.now(UTC)
    connection = SequentialFakeConnection(
        [
            [
                {
                    "id": conversation_id,
                    "title": "已删除对话",
                    "owner_user_id": None,
                    "archived_at": now,
                    "created_at": now,
                    "updated_at": now,
                }
            ],
            [
                {
                    "id": uuid4(),
                    "conversation_id": conversation_id,
                    "role": "user",
                    "sequence": 1,
                    "content_blocks": [{"type": "text", "value": "问题"}],
                    "sources": [],
                    "generation_run_id": None,
                    "status": "completed",
                    "error_message": None,
                    "created_at": now,
                    "updated_at": now,
                }
            ],
        ]
    )
    repository = ObservabilityRepository(cast(Engine, cast(Any, FakeEngine(connection))))

    snapshot = repository.run_conversation(workspace_id, run_id)

    assert snapshot is not None
    assert cast(dict[str, object], snapshot["conversation"])["deleted"] is True
    assert len(cast(list[dict[str, object]], snapshot["messages"])) == 1
    assert "trigger_message.generation_run_id = :run_id" in connection.sql
