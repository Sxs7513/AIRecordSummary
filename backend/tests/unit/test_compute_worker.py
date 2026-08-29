from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Event
from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from l1_foundation.settings import Settings
from l1_foundation.task_runtime.resources import ResourceQueue
from l1_foundation.worker.contracts import (
    ComputeCancelledEvent,
    ComputeCompletedEvent,
    ComputeDeltaEvent,
    ComputeEvent,
    ComputeProgressEvent,
    ComputeTaskRequest,
    ComputeTaskSnapshot,
    ComputeTaskStatus,
    ExecutionScope,
    WorkerExecutionContext,
)
from l3_app.compute_worker.executor import ComputeExecutionPool
from l3_app.compute_worker.registry import ComputeOperationRegistry, ComputeOperationSpec
from l3_app.compute_worker.registry_factory import build_compute_operation_registry
from l3_app.compute_worker.routes import WorkerMetricsResponse, router
from l3_app.compute_worker.runtime import ComputeTaskConflictError, ComputeWorkerRuntime


class EchoInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str


class EchoResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str


def echo_handler(value: EchoInput, context: WorkerExecutionContext) -> EchoResult:
    context.report_progress(0.5, "half")
    context.emit_delta(value.text)
    return EchoResult(text=value.text)


def registry() -> ComputeOperationRegistry:
    value = ComputeOperationRegistry()
    value.register(
        ComputeOperationSpec(
            name="test.echo",
            version="1",
            resource_queue=ResourceQueue.CPU,
            input_type=EchoInput,
            result_type=EchoResult,
            handler=echo_handler,
        )
    )
    return value


def request(*, task_id: UUID | None = None, text: str = "hello", wait_for_subscriber: bool = False) -> ComputeTaskRequest:
    return ComputeTaskRequest(
        task_id=task_id or uuid4(),
        operation="test.echo",
        operation_version="1",
        resource_queue=ResourceQueue.CPU,
        input={"text": text},
        wait_for_subscriber=wait_for_subscriber,
    )


async def _consume_events(runtime: ComputeWorkerRuntime, task_id: UUID) -> None:
    async for _ in runtime.events(task_id):
        pass


async def _collect_events(runtime: ComputeWorkerRuntime, task_id: UUID) -> list[ComputeEvent]:
    return [event async for event in runtime.events(task_id)]


def test_runtime_streams_live_events_and_keeps_the_final_snapshot(tmp_path: Path) -> None:
    async def scenario() -> tuple[list[ComputeEvent], ComputeTaskStatus, ComputeTaskSnapshot]:
        runtime = ComputeWorkerRuntime(registry(), ComputeExecutionPool())
        await runtime.start()
        task_request = request()
        try:
            queued = await runtime.submit(task_request)
            events = [event async for event in runtime.events(task_request.task_id)]
            snapshot = runtime.status(task_request.task_id)
            return events, queued.status, snapshot
        finally:
            await runtime.stop()

    events, queued_status, snapshot = asyncio.run(scenario())

    assert queued_status == ComputeTaskStatus.QUEUED
    assert any(isinstance(event, ComputeProgressEvent) for event in events)
    assert any(isinstance(event, ComputeDeltaEvent) for event in events)
    assert isinstance(events[-1], ComputeCompletedEvent)
    assert snapshot.status == ComputeTaskStatus.SUCCEEDED
    assert snapshot.result == {"text": "hello"}


def test_runtime_reuses_identical_task_id_and_rejects_conflicting_input(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = ComputeWorkerRuntime(registry(), ComputeExecutionPool())
        await runtime.start()
        task_id = uuid4()
        try:
            first = await runtime.submit(request(task_id=task_id))
            second = await runtime.submit(request(task_id=task_id))
            assert second.created_at == first.created_at
            try:
                await runtime.submit(request(task_id=task_id, text="different"))
            except ComputeTaskConflictError:
                pass
            else:
                raise AssertionError("conflicting task input must be rejected")
        finally:
            await runtime.stop()

    asyncio.run(scenario())


def test_runtime_releases_operation_after_success(tmp_path: Path) -> None:
    released = Event()
    value = ComputeOperationRegistry()
    value.register(
        ComputeOperationSpec(
            name="test.echo",
            version="1",
            resource_queue=ResourceQueue.CPU,
            input_type=EchoInput,
            result_type=EchoResult,
            handler=echo_handler,
            release=released.set,
        )
    )

    async def scenario() -> tuple[ComputeTaskSnapshot, bool]:
        runtime = ComputeWorkerRuntime(value, ComputeExecutionPool())
        await runtime.start()
        task_request = request()
        try:
            await runtime.submit(task_request)
            await asyncio.wait_for(
                asyncio.create_task(_consume_events(runtime, task_request.task_id)),
                timeout=1,
            )
            return runtime.status(task_request.task_id), released.is_set()
        finally:
            await runtime.stop()

    snapshot, released_before_shutdown = asyncio.run(scenario())

    assert snapshot.status == ComputeTaskStatus.SUCCEEDED
    assert released_before_shutdown


def test_runtime_releases_operation_after_failure(tmp_path: Path) -> None:
    released = Event()

    def failing_handler(value: EchoInput, context: WorkerExecutionContext) -> EchoResult:
        raise RuntimeError("inference failed")

    value = ComputeOperationRegistry()
    value.register(
        ComputeOperationSpec(
            name="test.echo",
            version="1",
            resource_queue=ResourceQueue.CPU,
            input_type=EchoInput,
            result_type=EchoResult,
            handler=failing_handler,
            release=released.set,
        )
    )

    async def scenario() -> tuple[ComputeTaskSnapshot, bool]:
        runtime = ComputeWorkerRuntime(value, ComputeExecutionPool())
        await runtime.start()
        task_request = request()
        try:
            await runtime.submit(task_request)
            await asyncio.wait_for(
                asyncio.create_task(_consume_events(runtime, task_request.task_id)),
                timeout=1,
            )
            return runtime.status(task_request.task_id), released.is_set()
        finally:
            await runtime.stop()

    snapshot, released_before_shutdown = asyncio.run(scenario())

    assert snapshot.status == ComputeTaskStatus.FAILED
    assert released_before_shutdown


def test_runtime_cancels_active_execution_scope_and_releases_operation(tmp_path: Path) -> None:
    started = Event()
    released = Event()

    def cancellable_handler(value: EchoInput, context: WorkerExecutionContext) -> EchoResult:
        started.set()
        while not context.is_cancel_requested:
            Event().wait(0.005)
        context.raise_if_cancelled()
        return EchoResult(text=value.text)

    value = ComputeOperationRegistry()
    value.register(
        ComputeOperationSpec(
            name="test.echo",
            version="1",
            resource_queue=ResourceQueue.CPU,
            input_type=EchoInput,
            result_type=EchoResult,
            handler=cancellable_handler,
            release=released.set,
        )
    )

    async def scenario() -> tuple[list[ComputeEvent], ComputeTaskSnapshot]:
        runtime = ComputeWorkerRuntime(value, ComputeExecutionPool())
        await runtime.start()
        scope = ExecutionScope(kind="generation", id=uuid4())
        task_request = request().model_copy(update={"execution_scope": scope})
        try:
            await runtime.submit(task_request)
            events_task = asyncio.create_task(_collect_events(runtime, task_request.task_id))
            assert await asyncio.to_thread(started.wait, 1)
            runtime.cancel_scope(scope)
            events = await asyncio.wait_for(events_task, timeout=1)
            return events, runtime.status(task_request.task_id)
        finally:
            await runtime.stop()

    events, snapshot = asyncio.run(scenario())

    assert isinstance(events[-1], ComputeCancelledEvent)
    assert snapshot.status == ComputeTaskStatus.CANCELLED
    assert released.is_set()


def test_runtime_fences_task_submitted_after_scope_cancellation(tmp_path: Path) -> None:
    invoked = Event()

    def tracked_handler(value: EchoInput, context: WorkerExecutionContext) -> EchoResult:
        invoked.set()
        return EchoResult(text=value.text)

    value = ComputeOperationRegistry()
    value.register(
        ComputeOperationSpec(
            name="test.echo",
            version="1",
            resource_queue=ResourceQueue.CPU,
            input_type=EchoInput,
            result_type=EchoResult,
            handler=tracked_handler,
        )
    )

    async def scenario() -> tuple[list[ComputeEvent], ComputeTaskSnapshot]:
        runtime = ComputeWorkerRuntime(value, ComputeExecutionPool())
        await runtime.start()
        scope = ExecutionScope(kind="processing", id=uuid4())
        runtime.cancel_scope(scope)
        task_request = request().model_copy(update={"execution_scope": scope})
        try:
            snapshot = await runtime.submit(task_request)
            events = await _collect_events(runtime, task_request.task_id)
            return events, snapshot
        finally:
            await runtime.stop()

    events, snapshot = asyncio.run(scenario())

    assert snapshot.status == ComputeTaskStatus.CANCELLED
    assert isinstance(events[-1], ComputeCancelledEvent)
    assert not invoked.is_set()


def test_batch_llm_operation_registers_shared_model_cleanup() -> None:
    operation = build_compute_operation_registry(Settings()).resolve("llm.generate_batch.local", "1")

    assert operation.release is not None


def test_streaming_task_starts_only_after_the_first_subscriber_attaches(tmp_path: Path) -> None:
    async def scenario() -> tuple[ComputeTaskStatus, list[ComputeEvent]]:
        runtime = ComputeWorkerRuntime(registry(), ComputeExecutionPool())
        await runtime.start()
        task_request = request(wait_for_subscriber=True)
        try:
            await runtime.submit(task_request)
            await asyncio.sleep(0)
            before_subscription = runtime.status(task_request.task_id).status
            events = [event async for event in runtime.events(task_request.task_id)]
            return before_subscription, events
        finally:
            await runtime.stop()

    before_subscription, events = asyncio.run(scenario())

    assert before_subscription == ComputeTaskStatus.QUEUED
    assert any(isinstance(event, ComputeDeltaEvent) for event in events)
    assert isinstance(events[-1], ComputeCompletedEvent)


def test_worker_http_routes_expose_health_and_metrics_only(tmp_path: Path) -> None:
    async def scenario() -> WorkerMetricsResponse:
        app = FastAPI()
        runtime = ComputeWorkerRuntime(registry(), ComputeExecutionPool())
        app.state.compute_worker_runtime = runtime
        app.include_router(router, prefix="/internal/v1/compute")
        await runtime.start()
        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
                assert (await client.get("/internal/v1/compute/healthz")).status_code == 200
                ready = await client.get("/internal/v1/compute/readyz")
                assert ready.json() == {"status": "ready", "registered_operations": 1}
                assert (await client.post("/internal/v1/compute/tasks", json={})).status_code == 404
                metrics_response = await client.get("/internal/v1/compute/metrics")
                return WorkerMetricsResponse.model_validate(metrics_response.json())
        finally:
            await runtime.stop()

    metrics = asyncio.run(scenario())
    assert metrics.ready is True
    assert metrics.total_tasks == 0
