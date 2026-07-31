from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

TerminalStatus = Literal["succeeded", "failed", "cancelled", "abandoned"]
InvocationStatus = Literal["running", "succeeded", "failed", "cancelled", "abandoned"]
UsageSource = Literal["provider", "local_tokenizer", "estimated", "unavailable"]


class ObservabilityScope(BaseModel):
    model_config = ConfigDict(frozen=True)

    workspace_id: UUID
    generation_run_id: UUID
    component: str = "rag"


class RagExecutionSpanRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    workspace_id: UUID
    generation_run_id: UUID
    parent_span_id: UUID | None = None
    component: str = "rag"
    operation: str
    operation_version: str = "1"
    attempt: int = Field(default=0, ge=0)
    status: InvocationStatus
    started_at: datetime
    finished_at: datetime | None = None
    elapsed_ms: float | None = Field(default=None, ge=0)
    error_type: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class ModelInvocationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    workspace_id: UUID
    generation_run_id: UUID
    span_id: UUID | None = None
    component: str = "rag"
    operation: str
    operation_version: str = "1"
    attempt: int = Field(default=0, ge=0)
    usage_kind: str = "llm"
    provider: str
    model: str | None = None
    stream: bool = False
    status: InvocationStatus
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    usage_source: UsageSource = "unavailable"
    finish_reason: str | None = None
    provider_request_id: str | None = None
    error_type: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    elapsed_ms: float | None = Field(default=None, ge=0)
    metadata: dict[str, object] = Field(default_factory=dict)
