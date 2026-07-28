from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NewType, Protocol
from uuid import UUID

from pydantic import BaseModel, Field

from l1_foundation.task_runtime.resources import ResourceQueue, RetryPolicy

PipelineSubjectId = NewType("PipelineSubjectId", UUID)
PipelineRunId = NewType("PipelineRunId", UUID)
StageRunId = NewType("StageRunId", UUID)


class StageRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    RETRY_WAITING = "retry_waiting"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class ArtifactRef(BaseModel):
    """A versioned reference to a persisted pipeline artifact."""

    artifact_type: str
    artifact_version: str
    uri: str
    producer_stage: str | None = None
    checksum: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactPayload(BaseModel):
    """Data a stage wants the runtime to persist as an artifact."""

    artifact_type: str
    artifact_version: str = "1"
    data: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


class StageProgressReporter(Protocol):
    """A thread-safe, runtime-provided sink for a stage's current progress."""

    def report(self, percent: int, message: str) -> None: ...


@dataclass(frozen=True, slots=True)
class StageContext:
    """Runtime identity supplied to a stage without coupling it to a business aggregate."""

    subject_id: PipelineSubjectId
    pipeline_run_id: PipelineRunId
    stage_run_id: StageRunId
    attempt_count: int
    progress_reporter: StageProgressReporter | None = None

    def report_progress(self, percent: int, message: str) -> None:
        if self.progress_reporter is not None:
            self.progress_reporter.report(percent, message)


@dataclass(frozen=True, slots=True)
class StageResult[OutputT]:
    output: OutputT
    artifacts: tuple[ArtifactPayload, ...] = ()


class Stage[InputT, OutputT](Protocol):
    name: str
    version: str
    resource_queue: ResourceQueue
    retry_policy: RetryPolicy

    async def run(self, context: StageContext, input_payload: InputT) -> StageResult[OutputT]: ...
