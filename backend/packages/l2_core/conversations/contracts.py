from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

from l2_core.generation.contracts import ContentBlock


class ConversationMessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ConversationMessageStatus(StrEnum):
    PENDING = "pending"
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Conversation(BaseModel):
    id: UUID
    workspace_id: UUID
    owner_user_id: UUID | None
    title: str
    archived_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ConversationMessage(BaseModel):
    id: UUID
    conversation_id: UUID
    role: ConversationMessageRole
    sequence: int = Field(gt=0)
    reply_to_message_id: UUID | None
    content_blocks: list[ContentBlock]
    sources: list[dict[str, object]]
    generation_run_id: UUID | None
    status: ConversationMessageStatus
    client_message_id: UUID | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class ConversationMessagePage(BaseModel):
    items: list[ConversationMessage]
    next_before: int | None
    has_more: bool


class ConversationNotFoundError(LookupError):
    pass


class ConversationBusyError(ValueError):
    pass
