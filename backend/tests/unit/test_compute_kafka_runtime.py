from __future__ import annotations

import asyncio
from typing import cast
from uuid import uuid4

from l1_foundation.messaging import new_event
from l1_foundation.streaming import RedisStreamStore
from l1_foundation.worker import ComputeCancelRequest, ExecutionScope
from l3_app.compute_worker.kafka_runtime import KafkaComputeCancelHandler, execution_scope_cancel_id, should_publish_compute_state
from l3_app.compute_worker.runtime import ComputeWorkerRuntime


class _Runtime:
    def __init__(self) -> None:
        self.cancelled_scopes: list[ExecutionScope] = []

    def cancel_scope(self, scope: ExecutionScope) -> list[object]:
        self.cancelled_scopes.append(scope)
        return []


class _Redis:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    async def request_cancel(self, task_id: str) -> None:
        self.cancelled.append(task_id)


def test_compute_state_skips_live_only_events_and_throttles_progress() -> None:
    assert not should_publish_compute_state("delta", None, 10.0)
    assert not should_publish_compute_state("heartbeat", None, 10.0)
    assert should_publish_compute_state("progress", None, 10.0)
    assert not should_publish_compute_state("progress", 10.0, 10.9)
    assert should_publish_compute_state("progress", 10.0, 11.0)


def test_compute_state_always_publishes_lifecycle_and_terminal_events() -> None:
    for event_type in ("started", "retrying", "completed", "failed", "cancelled"):
        assert should_publish_compute_state(event_type, 10.0, 10.1)


def test_compute_cancel_handler_projects_scope_and_cancels_local_runtime() -> None:
    scope = ExecutionScope(kind="generation", id=uuid4())
    runtime = _Runtime()
    redis = _Redis()
    handler = KafkaComputeCancelHandler(cast(ComputeWorkerRuntime, runtime), cast(RedisStreamStore, redis))
    event = new_event(
        "compute.cancel.requested",
        "test",
        payload=ComputeCancelRequest(execution_scope=scope).model_dump(mode="json"),
    )

    asyncio.run(handler.handle(event))

    assert redis.cancelled == [execution_scope_cancel_id(scope)]
    assert runtime.cancelled_scopes == [scope]
