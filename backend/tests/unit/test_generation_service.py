from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

from l2_core.generation.contracts import CreateGenerationCommand, GenerationNotFoundError
from l2_core.generation.service import GenerationService


def test_ensure_returns_postgres_snapshot_when_redis_state_expired() -> None:
    run_id = uuid4()
    command = MagicMock(spec=CreateGenerationCommand)
    postgres_snapshot = MagicMock()
    redis_runtime = MagicMock()
    redis_runtime.get_snapshot.return_value = None
    postgres_store = MagicMock()
    postgres_store.get_snapshot.return_value = postgres_snapshot
    service = _service(redis_runtime, postgres_store)

    result = service.ensure(run_id, command)

    assert result is postgres_snapshot
    postgres_store.get_snapshot.assert_called_once_with(run_id)
    redis_runtime.create_generation.assert_not_called()


def test_ensure_returns_postgres_terminal_snapshot_over_stale_redis_state() -> None:
    run_id = uuid4()
    command = MagicMock(spec=CreateGenerationCommand)
    postgres_snapshot = MagicMock()
    redis_runtime = MagicMock()
    redis_runtime.get_snapshot.return_value = (MagicMock(), "1-0")
    postgres_store = MagicMock()
    postgres_store.get_snapshot.return_value = postgres_snapshot
    service = _service(redis_runtime, postgres_store)

    result = service.ensure(run_id, command)

    assert result is postgres_snapshot
    redis_runtime.get_snapshot.assert_not_called()
    redis_runtime.create_generation.assert_not_called()


def test_ensure_creates_redis_state_when_generation_does_not_exist() -> None:
    run_id = uuid4()
    command = MagicMock(spec=CreateGenerationCommand)
    created_snapshot = MagicMock()
    redis_runtime = MagicMock()
    redis_runtime.get_snapshot.return_value = None
    redis_runtime.create_generation.return_value = created_snapshot
    postgres_store = MagicMock()
    postgres_store.get_snapshot.side_effect = GenerationNotFoundError(str(run_id))
    service = _service(redis_runtime, postgres_store)

    result = service.ensure(run_id, command)

    assert result is created_snapshot
    postgres_store.get_snapshot.assert_called_once_with(run_id)
    redis_runtime.create_generation.assert_called_once_with(command, run_id)


def _service(redis_runtime: MagicMock, postgres_store: MagicMock) -> GenerationService:
    service = GenerationService.__new__(GenerationService)
    service._redis_runtime = redis_runtime  # pyright: ignore[reportPrivateUsage]
    service._postgres_store = postgres_store  # pyright: ignore[reportPrivateUsage]
    return service
