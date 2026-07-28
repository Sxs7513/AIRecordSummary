from __future__ import annotations

import json
from typing import Any, Literal, cast
from uuid import UUID

from sqlalchemy import Engine, RowMapping, text

from l2_core.access.conversations import ConversationAccessDeniedError, ConversationAccessService
from l2_core.auth.contracts import CurrentUser
from l2_core.conversations.contracts import (
    Conversation,
    ConversationBusyError,
    ConversationMessage,
    ConversationMessagePage,
    ConversationMessageStatus,
    ConversationNotFoundError,
)
from l2_core.generation.contracts import ContentBlock, CreateGenerationCommand, GenerationAccessScope, GenerationKind, GenerationPriority, GenerationStatus
from l2_core.generation.service import GenerationService
from l2_core.rag.contracts import RagHistoryMessage, RagHistorySource


class ConversationService:
    """Transactional creation, pagination and Generation projection for chat conversations."""

    def __init__(self, engine: Engine, generation_service: GenerationService) -> None:
        self._engine = engine
        self._generation_service = generation_service
        self._access = ConversationAccessService(engine)

    def create(self, user: CurrentUser, title: str | None = None) -> Conversation:
        with self._engine.begin() as connection:
            row = (
                connection.execute(
                    text(
                        """
                        insert into conversations (workspace_id, owner_user_id, title)
                        select :workspace_id, :user_id, :title
                        where exists (
                            select 1 from workspace_memberships
                            where workspace_id = :workspace_id and user_id = :user_id
                        )
                        returning *
                        """
                    ),
                    {"workspace_id": user.current_workspace_id, "user_id": user.id, "title": title.strip() if title and title.strip() else "新对话"},
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise PermissionError("Current workspace access denied")
        return self._conversation(row)

    def list(self, user: CurrentUser, limit: int = 50) -> list[Conversation]:
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    select conversations.* from conversations
                    join workspace_memberships on workspace_memberships.workspace_id = conversations.workspace_id
                    where workspace_memberships.user_id = :user_id and conversations.archived_at is null
                    order by conversations.updated_at desc, conversations.id desc
                    limit :limit
                    """
                ),
                {"user_id": user.id, "limit": max(1, min(100, limit))},
            ).mappings()
        return [self._conversation(row) for row in rows]

    def delete(self, user: CurrentUser, conversation_id: UUID) -> None:
        self._access.require_view(conversation_id, user)
        with self._engine.begin() as connection:
            conversation = connection.execute(
                text(
                    """
                    select id from conversations
                    where id = :conversation_id and owner_user_id = :user_id
                    for update
                    """
                ),
                {"conversation_id": conversation_id, "user_id": user.id},
            ).scalar_one_or_none()
            if conversation is None:
                raise ConversationAccessDeniedError("Only the conversation owner can delete it")
            active = connection.execute(
                text(
                    """
                    select 1 from conversation_messages
                    where conversation_id = :conversation_id and role = 'assistant' and status in ('pending', 'streaming')
                    """
                ),
                {"conversation_id": conversation_id},
            ).scalar_one_or_none()
            if active is not None:
                raise ConversationBusyError("该对话正在生成回答，完成后才能删除")
            run_ids = (
                connection.execute(
                    text(
                        """
                    select generation_run_id from conversation_messages
                    where conversation_id = :conversation_id and generation_run_id is not null
                    """
                    ),
                    {"conversation_id": conversation_id},
                )
                .scalars()
                .all()
            )
            if run_ids:
                connection.execute(
                    text("delete from generation_runs where id = any(cast(:run_ids as uuid[]))"),
                    {"run_ids": [str(cast(UUID, run_id)) for run_id in run_ids]},
                )
            connection.execute(text("delete from conversations where id = :conversation_id"), {"conversation_id": conversation_id})

    def messages(self, user: CurrentUser, conversation_id: UUID, before: int | None, limit: int = 50) -> ConversationMessagePage:
        self._access.require_view(conversation_id, user)
        bounded_limit = max(1, min(100, limit))
        statement = (
            """
            select * from conversation_messages
            where conversation_id = :conversation_id
            order by sequence desc
            limit :limit
            """
            if before is None
            else """
            select * from conversation_messages
            where conversation_id = :conversation_id and sequence < :before
            order by sequence desc
            limit :limit
            """
        )
        parameters: dict[str, object] = {"conversation_id": conversation_id, "limit": bounded_limit}
        if before is not None:
            parameters["before"] = before
        with self._engine.connect() as connection:
            rows = connection.execute(text(statement), parameters).mappings().all()
        items = [self._message(row) for row in reversed(rows)]
        next_before = items[0].sequence if items else None
        return ConversationMessagePage(items=items, next_before=next_before, has_more=len(rows) >= bounded_limit)

    def create_turn(
        self, user: CurrentUser, conversation_id: UUID, blocks: list[ContentBlock], client_message_id: UUID, limit: int
    ) -> tuple[ConversationMessage, ConversationMessage, list[RagHistoryMessage]]:
        self._access.require_view(conversation_id, user)
        query = "".join(block.value for block in blocks).strip()
        if not query:
            raise ValueError("Message content must not be empty")
        with self._engine.begin() as connection:
            conversation = (
                connection.execute(text("select * from conversations where id = :conversation_id for update"), {"conversation_id": conversation_id})
                .mappings()
                .one_or_none()
            )
            if conversation is None:
                raise ConversationNotFoundError(str(conversation_id))
            existing = (
                connection.execute(
                    text(
                        """
                    select user_message.*, assistant_message.id as assistant_id
                    from conversation_messages user_message
                    join conversation_messages assistant_message on assistant_message.reply_to_message_id = user_message.id
                    where user_message.conversation_id = :conversation_id and user_message.client_message_id = :client_message_id
                    """
                    ),
                    {"conversation_id": conversation_id, "client_message_id": client_message_id},
                )
                .mappings()
                .one_or_none()
            )
            if existing is not None:
                user_message = self._message(existing)
                assistant = (
                    connection.execute(text("select * from conversation_messages where id = :message_id"), {"message_id": existing["assistant_id"]})
                    .mappings()
                    .one()
                )
                return user_message, self._message(assistant), self._history(connection, conversation_id, user_message.sequence)
            active = connection.execute(
                text(
                    """
                    select 1 from conversation_messages
                    where conversation_id = :conversation_id and role = 'assistant' and status in ('pending', 'streaming')
                    """
                ),
                {"conversation_id": conversation_id},
            ).scalar_one_or_none()
            if active is not None:
                raise ConversationBusyError("该对话仍在生成回答，请等待完成后再提问")
            sequence_end = cast(
                int,
                connection.execute(
                    text(
                        """
                        update conversations set next_message_sequence = next_message_sequence + 2, updated_at = now()
                        where id = :conversation_id returning next_message_sequence
                        """
                    ),
                    {"conversation_id": conversation_id},
                ).scalar_one(),
            )
            user_row = (
                connection.execute(
                    text(
                        """
                    insert into conversation_messages (conversation_id, role, sequence, content_blocks, status, client_message_id)
                    values (:conversation_id, 'user', :sequence, cast(:blocks as jsonb), 'completed', :client_message_id)
                    returning *
                    """
                    ),
                    {
                        "conversation_id": conversation_id,
                        "sequence": sequence_end - 1,
                        "blocks": json.dumps([block.model_dump(mode="json") for block in blocks]),
                        "client_message_id": client_message_id,
                    },
                )
                .mappings()
                .one()
            )
            assistant_row = (
                connection.execute(
                    text(
                        """
                    insert into conversation_messages (conversation_id, role, sequence, reply_to_message_id, status)
                    values (:conversation_id, 'assistant', :sequence, :reply_to_message_id, 'pending')
                    returning *
                    """
                    ),
                    {"conversation_id": conversation_id, "sequence": sequence_end, "reply_to_message_id": user_row["id"]},
                )
                .mappings()
                .one()
            )
            generation = self._generation_service.create_in_transaction(
                connection,
                CreateGenerationCommand(
                    kind=GenerationKind.TEXT,
                    priority=GenerationPriority.INTERACTIVE,
                    idempotency_key=f"conversation-message:{assistant_row['id']}",
                    parent_type="conversation_message",
                    parent_id=str(assistant_row["id"]),
                    access_scope=GenerationAccessScope(owner_user_id=user.id),
                    input={"query": query, "limit": limit, "conversation_id": str(conversation_id)},
                ),
            )
            assistant_row = (
                connection.execute(
                    text("update conversation_messages set generation_run_id = :run_id, updated_at = now() where id = :message_id returning *"),
                    {"run_id": generation.id, "message_id": assistant_row["id"]},
                )
                .mappings()
                .one()
            )
            if sequence_end == 2 and cast(str, conversation["title"]) == "新对话":
                connection.execute(
                    text("update conversations set title = :title, updated_at = now() where id = :id"),
                    {"id": conversation_id, "title": query[:80]},
                )
            history = self._history(connection, conversation_id, sequence_end - 1)
        return self._message(user_row), self._message(assistant_row), history

    def mark_streaming(self, generation_run_id: UUID) -> None:
        self._update_from_generation(generation_run_id, ConversationMessageStatus.STREAMING)

    def sync_generation(self, generation_run_id: UUID) -> None:
        snapshot = self._generation_service.get(generation_run_id)
        status = {
            GenerationStatus.SUCCEEDED: ConversationMessageStatus.COMPLETED,
            GenerationStatus.FAILED: ConversationMessageStatus.FAILED,
            GenerationStatus.CANCELLED: ConversationMessageStatus.CANCELLED,
        }.get(snapshot.status, ConversationMessageStatus.STREAMING)
        self._update_from_generation(generation_run_id, status)

    def _update_from_generation(self, generation_run_id: UUID, status: ConversationMessageStatus) -> None:
        snapshot = self._generation_service.get(generation_run_id)
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    update conversation_messages set status = :status, content_blocks = cast(:blocks as jsonb), sources = cast(:sources as jsonb),
                        error_message = :error_message, updated_at = now()
                    where generation_run_id = :generation_run_id
                    """
                ),
                {
                    "generation_run_id": generation_run_id,
                    "status": status.value,
                    "blocks": json.dumps([block.model_dump(mode="json") for block in snapshot.blocks]),
                    "sources": json.dumps(self._persistent_sources(snapshot.sources)),
                    "error_message": snapshot.error_message,
                },
            )

    @staticmethod
    def _persistent_sources(sources: list[dict[str, Any]]) -> list[dict[str, object]]:
        """Persist source metadata without duplicating retrieved transcript text."""
        sanitized: list[dict[str, object]] = []
        for source in sources:
            item: dict[str, object] = dict(source)
            chunk = item.get("chunk")
            if isinstance(chunk, dict):
                chunk_metadata = dict(cast(dict[str, object], chunk))
                chunk_metadata.pop("text", None)
                item["chunk"] = chunk_metadata
            sanitized.append(item)
        return sanitized

    @staticmethod
    def _history(connection: Any, conversation_id: UUID, before_sequence: int) -> list[RagHistoryMessage]:
        rows = (
            connection.execute(
                text(
                    """
                select role, content_blocks, sources from conversation_messages
                where conversation_id = :conversation_id and sequence < :before_sequence and status = 'completed'
                order by sequence desc limit 12
                """
                ),
                {"conversation_id": conversation_id, "before_sequence": before_sequence},
            )
            .mappings()
            .all()
        )
        history: list[RagHistoryMessage] = []
        for row in reversed(rows):
            role = str(row["role"])
            if role not in {"user", "assistant"}:
                continue
            history_role = cast(Literal["user", "assistant"], role)
            history.append(
                RagHistoryMessage(
                    role=history_role,
                    content="".join(str(item.get("value", "")) for item in cast(list[dict[str, object]], row["content_blocks"])),
                    sources=ConversationService._history_sources(row["sources"]),
                )
            )
        return history

    @staticmethod
    def _history_sources(value: object) -> list[RagHistorySource]:
        if not isinstance(value, list):
            return []
        sources: list[RagHistorySource] = []
        for raw_source in cast(list[object], value):
            if not isinstance(raw_source, dict):
                continue
            source = cast(dict[str, object], raw_source)
            recording = source.get("recording")
            if not isinstance(recording, dict):
                continue
            recording_data = cast(dict[str, object], recording)
            recording_id = recording_data.get("id")
            title = recording_data.get("title")
            if not isinstance(recording_id, str) or not isinstance(title, str):
                continue
            chunk = source.get("chunk")
            chunk_data = cast(dict[str, object], chunk) if isinstance(chunk, dict) else {}
            try:
                sources.append(
                    RagHistorySource(
                        recording_id=UUID(recording_id),
                        title=title,
                        start_ms=ConversationService._optional_nonnegative_int(chunk_data.get("startMs")),
                        end_ms=ConversationService._optional_nonnegative_int(chunk_data.get("endMs")),
                    )
                )
            except TypeError, ValueError:
                continue
        return sources

    @staticmethod
    def _optional_nonnegative_int(value: object) -> int | None:
        return value if isinstance(value, int) and value >= 0 else None

    @staticmethod
    def _conversation(row: RowMapping) -> Conversation:
        return Conversation.model_validate(dict(row))

    @staticmethod
    def _message(row: RowMapping) -> ConversationMessage:
        return ConversationMessage.model_validate(dict(row))
