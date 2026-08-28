from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from secrets import compare_digest
from threading import Event
from time import monotonic
from uuid import UUID

from l1_foundation.files import FileStore
from l1_foundation.infrastructure.storage.local import LocalStorage
from l1_foundation.task_runtime.resources import ResourceQueue
from l1_foundation.worker.contracts import (
    ComputeCancelledEvent,
    ComputeCompletedEvent,
    ComputeDeltaEvent,
    ComputeEvent,
    ComputeFailedEvent,
    ComputeHeartbeatEvent,
    ComputeProgressEvent,
    ComputeQueuedEvent,
    ComputeStartedEvent,
    ComputeTaskError,
    ComputeTaskRequest,
    ComputeTaskSnapshot,
    ComputeTaskStatus,
    ExecutionScope,
    JsonObject,
    WorkerExecutionContext,
)
from l3_app.compute_worker.executor import ComputeExecutionPool
from l3_app.compute_worker.registry import ComputeOperationRegistry, RegisteredComputeOperation

logger = logging.getLogger("worker")
TERMINAL_EVENT_TYPES = (ComputeCompletedEvent, ComputeFailedEvent, ComputeCancelledEvent)


class ComputeWorkerNotRunningError(RuntimeError):
    pass


class ComputeTaskNotFoundError(LookupError):
    pass


class ComputeTaskConflictError(ValueError):
    pass


class ComputeWorkerCapacityError(RuntimeError):
    pass


class ComputeResourceQueueMismatchError(ValueError):
    pass


class ComputeExecutionCancelled(RuntimeError):
    pass


@dataclass(slots=True)
class _TaskState:
    request: ComputeTaskRequest
    operation: RegisteredComputeOperation
    request_hash: str
    status: ComputeTaskStatus
    created_at: datetime
    progress: float | None = None
    message: str | None = None
    result: JsonObject | None = None
    error: ComputeTaskError | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancel_event: Event = field(default_factory=Event)
    subscribers: set[asyncio.Queue[ComputeEvent]] = field(default_factory=lambda: set[asyncio.Queue[ComputeEvent]]())
    execution_task: asyncio.Task[None] | None = None
    cancel_requested_at: datetime | None = None
    cancel_timeout_reported: bool = False

    def snapshot(self) -> ComputeTaskSnapshot:
        return ComputeTaskSnapshot(
            task_id=self.request.task_id,
            operation=self.request.operation,
            operation_version=self.request.operation_version,
            resource_queue=self.request.resource_queue,
            status=self.status,
            progress=self.progress,
            message=self.message,
            result=self.result,
            error=self.error,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )


class _RuntimeExecutionContext(WorkerExecutionContext):
    def __init__(self, runtime: ComputeWorkerRuntime, task_id: UUID, cancel_event: Event, loop: asyncio.AbstractEventLoop) -> None:
        self._runtime = runtime
        self._task_id = task_id
        self._cancel_event = cancel_event
        self._loop = loop

    @property
    def is_cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancel_requested:
            raise ComputeExecutionCancelled("Compute task was cancelled")

    def report_progress(self, progress: float, message: str | None = None) -> None:
        bounded = min(1.0, max(0.0, progress))
        self._loop.call_soon_threadsafe(self._runtime.report_progress_from_handler, self._task_id, bounded, message)

    def emit_delta(self, text: str, item_id: str | None = None) -> None:
        if text:
            self._loop.call_soon_threadsafe(self._runtime.emit_delta_from_handler, self._task_id, text, item_id)


@dataclass(frozen=True, slots=True)
class ComputeWorkerMetrics:
    ready: bool
    registered_operations: int
    total_tasks: int
    queued_tasks: int
    running_tasks: int
    succeeded_tasks: int
    failed_tasks: int
    cancelled_tasks: int
    cancel_timeout_tasks: int


class ComputeWorkerRuntime:
    """Single-process transient task manager backed by the shared resource scheduler."""

    def __init__(
        self,
        registry: ComputeOperationRegistry,
        execution_pool: ComputeExecutionPool,
        *,
        file_store: FileStore | None = None,
        result_prefix: str = "compute-tasks",
        output_root: Path | None = None,
        completed_ttl_seconds: float = 1800,
        max_tasks: int = 100,
        heartbeat_seconds: float = 15,
        cancel_grace_seconds: float = 10,
        internal_token: str | None = None,
    ) -> None:
        if completed_ttl_seconds <= 0:
            raise ValueError("completed_ttl_seconds must be positive")
        if max_tasks < 1:
            raise ValueError("max_tasks must be positive")
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        self._registry = registry
        self._execution_pool = execution_pool
        if file_store is None:
            if output_root is None:
                raise ValueError("file_store is required")
            local_storage = LocalStorage(output_root)
            local_storage.initialize()
            file_store = local_storage
            result_prefix = ""
        self._file_store = file_store
        self._result_prefix = result_prefix.strip("/")
        self._completed_ttl = timedelta(seconds=completed_ttl_seconds)
        self._max_tasks = max_tasks
        self._heartbeat_seconds = heartbeat_seconds
        self._cancel_grace = timedelta(seconds=cancel_grace_seconds)
        self._internal_token = internal_token.strip() if internal_token else None
        self._tasks: dict[UUID, _TaskState] = {}
        self._scope_tasks: dict[ExecutionScope, set[UUID]] = {}
        self._cancelled_scopes: set[ExecutionScope] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._cleanup_task: asyncio.Task[None] | None = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def authorize(self, token: str | None) -> bool:
        if self._internal_token is None:
            return True
        return token is not None and compare_digest(token, self._internal_token)

    async def start(self) -> None:
        if self._ready:
            return
        self._loop = asyncio.get_running_loop()
        self._execution_pool.start()
        self._ready = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop(), name="compute-worker-cleanup")
        logger.info("Compute worker started registered_operations=%d max_tasks=%d", self._registry.operation_count, self._max_tasks)

    async def stop(self, shutdown_grace_seconds: float = 5.0) -> None:
        if not self._ready:
            return
        self._ready = False
        cleanup_task = self._cleanup_task
        self._cleanup_task = None
        if cleanup_task is not None:
            cleanup_task.cancel()
            await asyncio.gather(cleanup_task, return_exceptions=True)
        active_tasks = [state.execution_task for state in self._tasks.values() if state.execution_task is not None and not state.execution_task.done()]
        for state in self._tasks.values():
            state.cancel_event.set()
        for task in active_tasks:
            task.cancel()
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        try:
            await self._execution_pool.submit(ResourceQueue.GPU_NORMAL, self._registry.release_all)
        except Exception:
            logger.info("Compute worker handler cleanup failed", exc_info=True)
        finally:
            self._execution_pool.stop(shutdown_grace_seconds)
        logger.info("Compute worker stopped remaining_tasks=%d", len(self._tasks))

    async def submit(self, request: ComputeTaskRequest) -> ComputeTaskSnapshot:
        self._require_ready()
        request_hash = self._request_hash(request)
        existing = self._tasks.get(request.task_id)
        if existing is not None:
            if existing.request_hash != request_hash:
                raise ComputeTaskConflictError(f"Task ID is already used by a different request: {request.task_id}")
            return existing.snapshot()
        self._evict_expired()
        self._evict_oldest_terminal_for_capacity()
        if len(self._tasks) >= self._max_tasks:
            raise ComputeWorkerCapacityError("Compute worker in-memory task capacity is full")
        operation = self._registry.resolve(request.operation, request.operation_version)
        if operation.resource_queue != request.resource_queue:
            raise ComputeResourceQueueMismatchError(
                f"Operation {request.operation}@{request.operation_version} requires queue={operation.resource_queue.value}"
            )
        now = datetime.now(UTC)
        state = _TaskState(
            request=request,
            operation=operation,
            request_hash=request_hash,
            status=ComputeTaskStatus.QUEUED,
            created_at=now,
        )
        self._tasks[request.task_id] = state
        if request.execution_scope is not None:
            self._scope_tasks.setdefault(request.execution_scope, set()).add(request.task_id)
        self._publish(state, ComputeQueuedEvent(task_id=request.task_id, at=now))
        if request.execution_scope in self._cancelled_scopes:
            state.cancel_event.set()
            self._mark_cancelled(state)
            return state.snapshot()
        if not request.wait_for_subscriber:
            self._start_execution(state)
        logger.info(
            "Compute task received task_id=%s operation=%s version=%s queue=%s wait_for_subscriber=%s",
            request.task_id,
            request.operation,
            request.operation_version,
            request.resource_queue.value,
            request.wait_for_subscriber,
        )
        return state.snapshot()

    def status(self, task_id: UUID) -> ComputeTaskSnapshot:
        self._require_ready()
        return self._state(task_id).snapshot()

    def cancel(self, task_id: UUID) -> ComputeTaskSnapshot:
        self._require_ready()
        state = self._state(task_id)
        if state.status.is_terminal:
            return state.snapshot()
        state.cancel_event.set()
        logger.info("Compute task cancel requested task_id=%s operation=%s", task_id, state.request.operation)
        if state.status == ComputeTaskStatus.QUEUED:
            if state.execution_task is not None:
                state.execution_task.cancel()
            self._mark_cancelled(state)
        else:
            state.status = ComputeTaskStatus.CANCEL_REQUESTED
            state.message = "Cancellation requested"
            state.cancel_requested_at = datetime.now(UTC)
        return state.snapshot()

    def cancel_scope(self, scope: ExecutionScope) -> list[ComputeTaskSnapshot]:
        """Fence future tasks and cooperatively cancel active tasks owned by one execution."""

        self._require_ready()
        self._cancelled_scopes.add(scope)
        snapshots: list[ComputeTaskSnapshot] = []
        for task_id in tuple(self._scope_tasks.get(scope, ())):
            state = self._tasks.get(task_id)
            if state is None or state.status.is_terminal:
                continue
            snapshots.append(self.cancel(task_id))
        return snapshots

    async def events(self, task_id: UUID) -> AsyncIterator[ComputeEvent]:
        self._require_ready()
        state = self._state(task_id)
        if state.status.is_terminal:
            yield self._terminal_event(state)
            return
        subscription: asyncio.Queue[ComputeEvent] = asyncio.Queue()
        state.subscribers.add(subscription)
        self._start_execution(state)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(subscription.get(), timeout=self._heartbeat_seconds)
                except TimeoutError:
                    yield ComputeHeartbeatEvent(task_id=task_id, at=datetime.now(UTC))
                    continue
                yield event
                if isinstance(event, TERMINAL_EVENT_TYPES):
                    return
        finally:
            state.subscribers.discard(subscription)

    def metrics(self) -> ComputeWorkerMetrics:
        counts = {status: 0 for status in ComputeTaskStatus}
        for state in self._tasks.values():
            counts[state.status] += 1
        return ComputeWorkerMetrics(
            ready=self._ready,
            registered_operations=self._registry.operation_count,
            total_tasks=len(self._tasks),
            queued_tasks=counts[ComputeTaskStatus.QUEUED],
            running_tasks=counts[ComputeTaskStatus.RUNNING] + counts[ComputeTaskStatus.CANCEL_REQUESTED],
            succeeded_tasks=counts[ComputeTaskStatus.SUCCEEDED],
            failed_tasks=counts[ComputeTaskStatus.FAILED],
            cancelled_tasks=counts[ComputeTaskStatus.CANCELLED],
            cancel_timeout_tasks=sum(1 for state in self._tasks.values() if state.cancel_timeout_reported),
        )

    async def _execute(self, state: _TaskState, operation: RegisteredComputeOperation) -> None:
        task_id = state.request.task_id
        started = monotonic()
        loop = self._loop
        if loop is None:
            raise ComputeWorkerNotRunningError("Compute worker event loop is unavailable")
        context = _RuntimeExecutionContext(self, task_id, state.cancel_event, loop)

        def execute() -> JsonObject:
            try:
                loop.call_soon_threadsafe(self._mark_started, state)
                context.raise_if_cancelled()
                result = operation.execute(state.request.input, context)
                context.raise_if_cancelled()
                self._persist_result(task_id, result)
                return result
            finally:
                if operation.release is not None:
                    try:
                        operation.release()
                    except Exception:
                        logger.exception(
                            "Compute operation cleanup failed task_id=%s operation=%s",
                            task_id,
                            operation.name,
                        )

        try:
            result = await self._execution_pool.submit(operation.resource_queue, execute)
            if state.cancel_event.is_set():
                self._mark_cancelled(state)
            else:
                self._mark_succeeded(state, result)
        except asyncio.CancelledError:
            self._mark_cancelled(state)
            raise
        except ComputeExecutionCancelled:
            self._mark_cancelled(state)
        except Exception as error:
            self._mark_failed(state, error)
        finally:
            logger.info(
                "Compute task finished task_id=%s operation=%s status=%s duration_ms=%d",
                task_id,
                state.request.operation,
                state.status.value,
                round((monotonic() - started) * 1000),
            )

    def _start_execution(self, state: _TaskState) -> None:
        if state.execution_task is not None or state.status != ComputeTaskStatus.QUEUED:
            return
        state.execution_task = asyncio.create_task(
            self._execute(state, state.operation),
            name=f"compute-task-{state.request.task_id}",
        )

    def _mark_started(self, state: _TaskState) -> None:
        if state.status != ComputeTaskStatus.QUEUED:
            return
        now = datetime.now(UTC)
        state.status = ComputeTaskStatus.RUNNING
        state.started_at = now
        state.message = None
        self._publish(state, ComputeStartedEvent(task_id=state.request.task_id, at=now))
        logger.info("Compute task started task_id=%s operation=%s", state.request.task_id, state.request.operation)

    def report_progress_from_handler(self, task_id: UUID, progress: float, message: str | None) -> None:
        state = self._tasks.get(task_id)
        if state is None or state.status.is_terminal:
            return
        state.progress = progress
        state.message = message
        self._publish(
            state,
            ComputeProgressEvent(task_id=task_id, at=datetime.now(UTC), progress=progress, message=message),
        )

    def emit_delta_from_handler(self, task_id: UUID, text: str, item_id: str | None = None) -> None:
        state = self._tasks.get(task_id)
        if state is None or state.status.is_terminal:
            return
        self._publish(state, ComputeDeltaEvent(task_id=task_id, at=datetime.now(UTC), text=text, item_id=item_id))

    def _mark_succeeded(self, state: _TaskState, result: JsonObject) -> None:
        now = datetime.now(UTC)
        state.status = ComputeTaskStatus.SUCCEEDED
        state.progress = 1.0
        state.result = result
        state.error = None
        state.finished_at = now
        self._publish(state, ComputeCompletedEvent(task_id=state.request.task_id, at=now, result=result))
        logger.info(
            "Compute task succeeded task_id=%s operation=%s",
            state.request.task_id,
            state.request.operation,
        )

    def _mark_failed(self, state: _TaskState, error: Exception) -> None:
        now = datetime.now(UTC)
        task_error = ComputeTaskError(code="compute_failed", message=str(error) or type(error).__name__, retryable=False)
        state.status = ComputeTaskStatus.FAILED
        state.error = task_error
        state.finished_at = now
        self._publish(state, ComputeFailedEvent(task_id=state.request.task_id, at=now, error=task_error))
        logger.info(
            "Compute task failed task_id=%s operation=%s error_type=%s",
            state.request.task_id,
            state.request.operation,
            type(error).__name__,
            exc_info=True,
        )

    def _mark_cancelled(self, state: _TaskState) -> None:
        if state.status == ComputeTaskStatus.CANCELLED:
            return
        now = datetime.now(UTC)
        state.status = ComputeTaskStatus.CANCELLED
        state.message = "Cancelled"
        state.finished_at = now
        self._publish(state, ComputeCancelledEvent(task_id=state.request.task_id, at=now))
        logger.info(
            "Compute task cancelled task_id=%s operation=%s",
            state.request.task_id,
            state.request.operation,
        )

    @staticmethod
    def _publish(state: _TaskState, event: ComputeEvent) -> None:
        for subscriber in tuple(state.subscribers):
            subscriber.put_nowait(event)

    @staticmethod
    def _terminal_event(state: _TaskState) -> ComputeEvent:
        at = state.finished_at or datetime.now(UTC)
        if state.status == ComputeTaskStatus.SUCCEEDED and state.result is not None:
            return ComputeCompletedEvent(task_id=state.request.task_id, at=at, result=state.result)
        if state.status == ComputeTaskStatus.FAILED and state.error is not None:
            return ComputeFailedEvent(task_id=state.request.task_id, at=at, error=state.error)
        return ComputeCancelledEvent(task_id=state.request.task_id, at=at)

    def _state(self, task_id: UUID) -> _TaskState:
        try:
            return self._tasks[task_id]
        except KeyError as error:
            raise ComputeTaskNotFoundError(f"Compute task not found: {task_id}") from error

    def _require_ready(self) -> None:
        if not self._ready:
            raise ComputeWorkerNotRunningError("Compute worker is not ready")

    async def _cleanup_loop(self) -> None:
        interval = min(5.0, max(1.0, self._cancel_grace.total_seconds() / 2))
        while True:
            await asyncio.sleep(interval)
            self._report_cancel_timeouts()
            self._evict_expired()

    def _report_cancel_timeouts(self) -> None:
        now = datetime.now(UTC)
        for state in self._tasks.values():
            if (
                state.status == ComputeTaskStatus.CANCEL_REQUESTED
                and state.cancel_requested_at is not None
                and not state.cancel_timeout_reported
                and now - state.cancel_requested_at >= self._cancel_grace
            ):
                state.cancel_timeout_reported = True
                logger.error(
                    "Compute task cancel timeout task_id=%s operation=%s grace_seconds=%s",
                    state.request.task_id,
                    state.request.operation,
                    self._cancel_grace.total_seconds(),
                )

    def _evict_expired(self) -> None:
        threshold = datetime.now(UTC) - self._completed_ttl
        expired = [
            task_id for task_id, state in self._tasks.items() if state.status.is_terminal and state.finished_at is not None and state.finished_at < threshold
        ]
        for task_id in expired:
            self._evict(task_id)

    def _evict_oldest_terminal_for_capacity(self) -> None:
        if len(self._tasks) < self._max_tasks:
            return
        terminal = [(state.finished_at or state.created_at, task_id) for task_id, state in self._tasks.items() if state.status.is_terminal]
        if terminal:
            _, task_id = min(terminal)
            self._evict(task_id)

    def _evict(self, task_id: UUID) -> None:
        state = self._tasks.pop(task_id)
        scope = state.request.execution_scope
        if scope is not None:
            task_ids = self._scope_tasks.get(scope)
            if task_ids is not None:
                task_ids.discard(task_id)
                if not task_ids:
                    self._scope_tasks.pop(scope, None)
        self._file_store.delete_file(self._result_key(task_id))
        logger.info("Compute task evicted task_id=%s operation=%s", task_id, state.request.operation)

    def _persist_result(self, task_id: UUID, result: JsonObject) -> None:
        with tempfile.TemporaryDirectory(prefix="compute-result-") as temporary_directory:
            source = Path(temporary_directory) / "result.json"
            source.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")
            self._file_store.put_file(source, key=self._result_key(task_id))

    def _result_key(self, task_id: UUID) -> str:
        suffix = f"{task_id}/result.json"
        return f"{self._result_prefix}/{suffix}" if self._result_prefix else suffix

    @staticmethod
    def _request_hash(request: ComputeTaskRequest) -> str:
        payload = request.model_dump(mode="json", exclude={"task_id"})
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()
