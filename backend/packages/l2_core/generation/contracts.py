from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from l2_core.rag.adjudication.contracts import AdjudicationConfirmationBlock


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

    @property
    def is_terminal(self) -> bool:
        return self in {self.SUCCEEDED, self.FAILED, self.CANCELLED}


class TextBlock(BaseModel):
    """A user-visible text fragment."""

    type: Literal["text"] = "text"
    value: str = Field(min_length=1)


def empty_sources() -> list[dict[str, Any]]:
    return []


def empty_text_blocks() -> list[TextBlock]:
    return []


def empty_sub_messages() -> list[SubMessage]:
    return []


def empty_sub_message_ids() -> list[str]:
    return []


class MessageGroup(BaseModel):
    id: str = Field(min_length=1)
    sub_message_ids: list[str] = Field(default_factory=empty_sub_message_ids, min_length=2)
    primary_sub_message_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_primary(self) -> MessageGroup:
        if len(self.sub_message_ids) != len(set(self.sub_message_ids)):
            raise ValueError("message group sub_message_ids must be unique")
        if self.primary_sub_message_id not in self.sub_message_ids:
            raise ValueError("message group primary_sub_message_id must belong to the group")
        return self


class SubMessage(BaseModel):
    id: str = Field(min_length=1)
    variant: Literal["original", "corrected"]
    title: str = Field(min_length=1)
    status: Literal["pending", "streaming", "completed", "failed", "cancelled"] = "pending"
    blocks: list[TextBlock] = Field(default_factory=empty_text_blocks)
    sources: list[dict[str, Any]] = Field(default_factory=empty_sources)
    error: str | None = None


class AggregateSubMessage(BaseModel):
    message_group: MessageGroup
    sub_message_list: list[SubMessage] = Field(default_factory=empty_sub_messages)

    @model_validator(mode="after")
    def validate_members(self) -> AggregateSubMessage:
        member_ids = [item.id for item in self.sub_message_list]
        if len(member_ids) != len(set(member_ids)):
            raise ValueError("aggregate sub-message ids must be unique")
        if not set(member_ids) <= set(self.message_group.sub_message_ids):
            raise ValueError("aggregate sub-messages must belong to message_group")
        return self


class AggreMessageBlock(BaseModel):
    """One user-visible message containing independently streamed answer variants."""

    type: Literal["AGGRE_MSG"] = "AGGRE_MSG"
    id: str = Field(min_length=1)
    sub_message: AggregateSubMessage


ContentBlock = Annotated[TextBlock | AggreMessageBlock | AdjudicationConfirmationBlock, Field(discriminator="type")]
_CONTENT_BLOCK_ADAPTER: TypeAdapter[ContentBlock] = TypeAdapter(ContentBlock)


def parse_content_block(value: object) -> ContentBlock:
    return _CONTENT_BLOCK_ADAPTER.validate_python(value)


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
