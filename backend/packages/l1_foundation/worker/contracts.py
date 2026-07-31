from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Protocol, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from l1_foundation.task_runtime.resources import ResourceQueue

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


class WorkerExecutionContext(Protocol):
    @property
    def is_cancel_requested(self) -> bool: ...

    def raise_if_cancelled(self) -> None: ...

    def report_progress(self, progress: float, message: str | None = None) -> None: ...

    def emit_delta(self, text: str, item_id: str | None = None) -> None: ...


class WorkerHandler[InputT: BaseModel, ResultT: BaseModel](Protocol):
    def __call__(self, value: InputT, context: WorkerExecutionContext) -> ResultT: ...


class ExecutionScope(BaseModel):
    """The durable upstream execution that owns one compute task."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["generation", "processing", "evaluation", "standalone"]
    id: UUID


_CURRENT_EXECUTION_SCOPE: ContextVar[ExecutionScope | None] = ContextVar("compute_execution_scope", default=None)


def current_execution_scope() -> ExecutionScope | None:
    return _CURRENT_EXECUTION_SCOPE.get()


@contextmanager
def execution_scope(scope: ExecutionScope) -> Generator[None]:
    token: Token[ExecutionScope | None] = _CURRENT_EXECUTION_SCOPE.set(scope)
    try:
        yield
    finally:
        _CURRENT_EXECUTION_SCOPE.reset(token)


class ComputeTaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {self.SUCCEEDED, self.FAILED, self.CANCELLED}


class ComputeTaskError(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=2000)
    retryable: bool = False
    details: JsonObject = Field(default_factory=dict)


class ComputeTaskRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: UUID
    operation: str = Field(min_length=1, max_length=120)
    operation_version: str = Field(min_length=1, max_length=40)
    resource_queue: ResourceQueue
    execution_scope: ExecutionScope | None = None
    input: JsonObject = Field(default_factory=dict)
    wait_for_subscriber: bool = False


@dataclass(frozen=True, slots=True)
class ComputeCommand[InputT: BaseModel]:
    task_id: UUID
    operation: str
    operation_version: str
    resource_queue: ResourceQueue
    input: InputT
    wait_for_subscriber: bool = False
    execution_scope: ExecutionScope | None = None

    def to_request(self) -> ComputeTaskRequest:
        return ComputeTaskRequest(
            task_id=self.task_id,
            operation=self.operation,
            operation_version=self.operation_version,
            resource_queue=self.resource_queue,
            execution_scope=self.execution_scope or current_execution_scope(),
            input=cast(JsonObject, self.input.model_dump(mode="json")),
            wait_for_subscriber=self.wait_for_subscriber,
        )


class ComputeCancelRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: UUID | None = None
    execution_scope: ExecutionScope | None = None
    reason: str = Field(default="requested", min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_target(self) -> ComputeCancelRequest:
        target_count = int(self.task_id is not None) + int(self.execution_scope is not None)
        if target_count != 1:
            raise ValueError("Compute cancellation requires exactly one task_id or execution_scope")
        return self


class ComputeTaskSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: UUID
    operation: str
    operation_version: str
    resource_queue: ResourceQueue
    status: ComputeTaskStatus
    progress: float | None = Field(default=None, ge=0, le=1)
    message: str | None = None
    result: JsonObject | None = None
    error: ComputeTaskError | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ComputeQueuedEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["queued"] = "queued"
    task_id: UUID
    at: datetime


class ComputeStartedEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["started"] = "started"
    task_id: UUID
    at: datetime


class ComputeProgressEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["progress"] = "progress"
    task_id: UUID
    at: datetime
    progress: float = Field(ge=0, le=1)
    message: str | None = None


class ComputeDeltaEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["delta"] = "delta"
    task_id: UUID
    at: datetime
    text: str
    item_id: str | None = None


class ComputeRetryingEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["retrying"] = "retrying"
    task_id: UUID
    at: datetime
    attempt: int = Field(ge=1)
    retry_in_seconds: float = Field(ge=0)
    message: str | None = None


class ComputeCompletedEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["completed"] = "completed"
    task_id: UUID
    at: datetime
    result: JsonObject


class ComputeFailedEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["failed"] = "failed"
    task_id: UUID
    at: datetime
    error: ComputeTaskError


class ComputeCancelledEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["cancelled"] = "cancelled"
    task_id: UUID
    at: datetime


class ComputeHeartbeatEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["heartbeat"] = "heartbeat"
    task_id: UUID
    at: datetime


type ComputeEvent = Annotated[
    ComputeQueuedEvent
    | ComputeStartedEvent
    | ComputeProgressEvent
    | ComputeDeltaEvent
    | ComputeRetryingEvent
    | ComputeCompletedEvent
    | ComputeFailedEvent
    | ComputeCancelledEvent
    | ComputeHeartbeatEvent,
    Field(discriminator="type"),
]

COMPUTE_EVENT_ADAPTER: TypeAdapter[ComputeEvent] = TypeAdapter(ComputeEvent)


def parse_compute_event(value: JsonObject) -> ComputeEvent:
    return COMPUTE_EVENT_ADAPTER.validate_python(value)
