from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GenerationKind(StrEnum):
    TEXT = "text"


class GenerationPriority(StrEnum):
    INTERACTIVE = "interactive"
    BACKGROUND = "background"


class GenerationStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TextBlock(BaseModel):
    """The first and currently only user-visible content block."""

    type: Literal["text"] = "text"
    value: str = Field(min_length=1)


ContentBlock = TextBlock


def empty_sources() -> list[dict[str, Any]]:
    return []


class GenerationPhase(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=200)


class GenerationEvent(BaseModel):
    """The versioned event envelope shared by database, SSE, and the web SDK."""

    model_config = ConfigDict(populate_by_name=True)

    v: Literal[1] = 1
    run_id: UUID
    seq: int = Field(ge=0)
    type: str
    at: datetime
    data: dict[str, Any] = Field(default_factory=dict)


class GenerationSnapshot(BaseModel):
    id: UUID
    kind: GenerationKind
    priority: GenerationPriority
    status: GenerationStatus
    phase: GenerationPhase | None
    progress_percent: int | None
    blocks: list[ContentBlock]
    sources: list[dict[str, Any]] = Field(default_factory=empty_sources)
    output: dict[str, Any] | None
    last_sequence: int
    cancel_requested: bool
    error_code: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    updated_at: datetime


class GenerationAccessScope(BaseModel):
    """Generic ownership or subject scope used by the authorization layer."""

    owner_user_id: UUID | None = None
    subject_type: str | None = Field(default=None, min_length=1, max_length=80)
    subject_id: UUID | None = None

    @model_validator(mode="after")
    def validate_subject(self) -> GenerationAccessScope:
        if (self.subject_type is None) != (self.subject_id is None):
            raise ValueError("subject_type and subject_id must be provided together")
        return self


class CreateGenerationCommand(BaseModel):
    kind: GenerationKind
    priority: GenerationPriority
    idempotency_key: str = Field(min_length=1, max_length=200)
    parent_type: str | None = Field(default=None, max_length=80)
    parent_id: str | None = Field(default=None, max_length=200)
    access_scope: GenerationAccessScope = Field(default_factory=GenerationAccessScope)
    input: dict[str, Any] = Field(default_factory=dict)


class GenerationNotFoundError(LookupError):
    """Raised when a requested generation run no longer exists."""
