from __future__ import annotations

import json
import logging
from typing import Any, Literal, cast
from uuid import UUID

from sqlalchemy import Connection, Engine, RowMapping, text

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
from l2_core.conversations.history_store import ConversationHistoryStore
from l2_core.generation.contracts import (
    CreateGenerationCommand,
    GenerationAccessScope,
    GenerationKind,
    GenerationPriority,
    GenerationSnapshot,
    GenerationStatus,
    TextBlock,
)
from l2_core.generation.service import GenerationService
from l2_core.rag.adjudication.contracts import AdjudicationConfirmationBlock, ClaimConfirmationDecision
from l2_core.rag.contracts import RagHistoryMessage, RagHistorySource
from l2_core.rag.queue import GenerationCommandPublisher, RagGenerationWorkItem

logger = logging.getLogger(__name__)


class ConversationService:
    """Transactional creation, pagination and Generation projection for chat conversations."""

    def __init__(
        self,
        engine: Engine,
        generation_service: GenerationService,
        history_store: ConversationHistoryStore | None = None,
        command_publisher: GenerationCommandPublisher | None = None,
    ) -> None:
        self._engine = engine
        self._generation_service = generation_service
        self._history_store = history_store
        self._command_publisher = command_publisher
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
                    where conversations.workspace_id = :workspace_id
                        and conversations.owner_user_id = :user_id
                        and conversations.archived_at is null
                    order by conversations.updated_at desc, conversations.id desc
                    limit :limit
                    """
                ),
                {
                    "workspace_id": user.current_workspace_id,
                    "user_id": user.id,
                    "limit": max(1, min(100, limit)),
                },
            ).mappings()
        return [self._conversation(row) for row in rows]

    def delete(self, user: CurrentUser, conversation_id: UUID) -> None:
        self._access.require_view(conversation_id, user)
        generation_run_ids: list[UUID] = []
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
            generation_run_ids = list(
                connection.execute(
                    text(
                        """
                        select distinct generation_run_id
                        from (
                            select generation_run_id
                            from conversation_messages
                            where conversation_id = :conversation_id and generation_run_id is not null
                            union
                            select generation_runs.id
                            from generation_runs
                            join conversation_messages
                                on generation_runs.parent_type = 'conversation_message'
                                and generation_runs.parent_id = conversation_messages.id::text
                            where conversation_messages.conversation_id = :conversation_id
                        ) conversation_generations
                        """
                    ),
                    {"conversation_id": conversation_id},
                )
                .scalars()
                .all()
            )
            connection.execute(
                text(
                    """
                    update conversations
                    set owner_user_id = null, archived_at = now(), updated_at = now()
                    where id = :conversation_id and owner_user_id = :user_id
                    """
                ),
                {"conversation_id": conversation_id, "user_id": user.id},
            )
        self._delete_history_cache(conversation_id)
        self._delete_generation_runtime_data(conversation_id, generation_run_ids)

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
        self,
        user: CurrentUser,
        conversation_id: UUID,
        blocks: list[TextBlock],
        client_message_id: UUID,
        limit: int,
        scope_recording_ids: list[UUID] | None = None,
    ) -> tuple[ConversationMessage, ConversationMessage, list[RagHistoryMessage]]:
        query = "".join(block.value for block in blocks).strip()
        if not query:
            raise ValueError("Message content must not be empty")
        superseded_generation_ids: list[UUID] = []
        with self._engine.begin() as connection:
            conversation = (
                connection.execute(
                    text(
                        """
                        select * from conversations
                        where id = :conversation_id
                            and workspace_id = :workspace_id
                            and owner_user_id = :user_id
                            and archived_at is null
                        for update
                        """
                    ),
                    {
                        "conversation_id": conversation_id,
                        "workspace_id": user.current_workspace_id,
                        "user_id": user.id,
                    },
                )
                .mappings()
                .one_or_none()
            )
            if conversation is None:
                raise ConversationAccessDeniedError(str(conversation_id))
            existing_rows = (
                connection.execute(
                    text(
                        """
                        with existing_turn as (
                            select user_message.id as user_message_id, assistant_message.id as assistant_message_id
                            from conversation_messages user_message
                            join conversation_messages assistant_message on assistant_message.reply_to_message_id = user_message.id
                            where user_message.conversation_id = :conversation_id
                                and user_message.client_message_id = :client_message_id
                        )
                        select message.*
                        from existing_turn
                        join conversation_messages message
                            on message.id in (existing_turn.user_message_id, existing_turn.assistant_message_id)
                        order by message.sequence
                        """
                    ),
                    {"conversation_id": conversation_id, "client_message_id": client_message_id},
                )
                .mappings()
                .all()
            )
            if existing_rows:
                user_message = self._message(existing_rows[0])
                assistant_message = self._message(existing_rows[1])
                # A completed retry may already be present in Redis; history for this
                # request must still end immediately before its user message.
                return user_message, assistant_message, self._history(connection, conversation_id, user_message.sequence, use_cache=False)
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
            superseded_generation_ids = list(
                connection.execute(
                    text(
                        """
                        select generation_run_id from conversation_messages
                        where conversation_id = :conversation_id and role = 'assistant'
                          and status in ('failed', 'cancelled') and generation_run_id is not null
                        """
                    ),
                    {"conversation_id": conversation_id},
                )
                .scalars()
                .all()
            )
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
            command = CreateGenerationCommand(
                kind=GenerationKind.TEXT,
                priority=GenerationPriority.INTERACTIVE,
                idempotency_key=f"conversation-message:{assistant_row['id']}",
                parent_type="conversation_message",
                parent_id=str(assistant_row["id"]),
                access_scope=GenerationAccessScope(owner_user_id=user.id),
                input={"query": query, "limit": limit, "conversation_id": str(conversation_id)},
            )
            generation = self._generation_service.create_in_transaction(connection, command)
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
            self._enqueue_rag(
                connection,
                user,
                generation.id,
                cast(UUID, assistant_row["id"]),
                query,
                limit,
                history,
                scope_recording_ids or [],
                command,
            )
        self._delete_generation_runtime_data(conversation_id, superseded_generation_ids)
        return self._message(user_row), self._message(assistant_row), history

    def create_initial_turn(
        self,
        user: CurrentUser,
        client_creation_id: UUID,
        blocks: list[TextBlock],
        client_message_id: UUID,
        limit: int,
        scope_recording_ids: list[UUID] | None = None,
    ) -> tuple[Conversation, ConversationMessage, ConversationMessage, list[RagHistoryMessage], bool]:
        """Atomically create a conversation and its first generation-backed turn.

        ``client_creation_id`` makes reconnecting the POST SSE request idempotent.
        """
        query = "".join(block.value for block in blocks).strip()
        if not query:
            raise ValueError("Message content must not be empty")
        serialized_blocks = json.dumps([block.model_dump(mode="json") for block in blocks])
        with self._engine.begin() as connection:
            conversation_row = (
                connection.execute(
                    text(
                        """
                        insert into conversations (workspace_id, owner_user_id, client_creation_id, title, next_message_sequence)
                        select :workspace_id, :user_id, :client_creation_id, :title, 2
                        where exists (
                            select 1 from workspace_memberships
                            where workspace_id = :workspace_id and user_id = :user_id
                        )
                        on conflict (owner_user_id, client_creation_id) where client_creation_id is not null do nothing
                        returning *
                        """
                    ),
                    {
                        "workspace_id": user.current_workspace_id,
                        "user_id": user.id,
                        "client_creation_id": client_creation_id,
                        "title": query[:80],
                    },
                )
                .mappings()
                .one_or_none()
            )
            if conversation_row is None:
                conversation_row = (
                    connection.execute(
                        text(
                            """
                            select * from conversations
                            where owner_user_id = :user_id
                                and workspace_id = :workspace_id
                                and client_creation_id = :client_creation_id
                                and archived_at is null
                            """
                        ),
                        {
                            "workspace_id": user.current_workspace_id,
                            "user_id": user.id,
                            "client_creation_id": client_creation_id,
                        },
                    )
                    .mappings()
                    .one_or_none()
                )
                if conversation_row is None:
                    raise PermissionError("Current workspace access denied")
                rows = (
                    connection.execute(
                        text(
                            """
                            select * from conversation_messages
                            where conversation_id = :conversation_id
                            order by sequence
                            limit 2
                            """
                        ),
                        {"conversation_id": conversation_row["id"]},
                    )
                    .mappings()
                    .all()
                )
                if len(rows) != 2:
                    raise RuntimeError("Initial conversation turn is incomplete")
                user_message, assistant_message = self._message(rows[0]), self._message(rows[1])
                return self._conversation(conversation_row), user_message, assistant_message, [], False

            conversation_id = cast(UUID, conversation_row["id"])
            user_row = (
                connection.execute(
                    text(
                        """
                        insert into conversation_messages (conversation_id, role, sequence, content_blocks, status, client_message_id)
                        values (:conversation_id, 'user', 1, cast(:blocks as jsonb), 'completed', :client_message_id)
                        returning *
                        """
                    ),
                    {"conversation_id": conversation_id, "blocks": serialized_blocks, "client_message_id": client_message_id},
                )
                .mappings()
                .one()
            )
            assistant_row = (
                connection.execute(
                    text(
                        """
                        insert into conversation_messages (conversation_id, role, sequence, reply_to_message_id, status)
                        values (:conversation_id, 'assistant', 2, :reply_to_message_id, 'pending')
                        returning *
                        """
                    ),
                    {"conversation_id": conversation_id, "reply_to_message_id": user_row["id"]},
                )
                .mappings()
                .one()
            )
            command = CreateGenerationCommand(
                kind=GenerationKind.TEXT,
                priority=GenerationPriority.INTERACTIVE,
                idempotency_key=f"conversation-message:{assistant_row['id']}",
                parent_type="conversation_message",
                parent_id=str(assistant_row["id"]),
                access_scope=GenerationAccessScope(owner_user_id=user.id),
                input={"query": query, "limit": limit, "conversation_id": str(conversation_id)},
            )
            generation = self._generation_service.create_in_transaction(connection, command)
            assistant_row = (
                connection.execute(
                    text("update conversation_messages set generation_run_id = :run_id, updated_at = now() where id = :message_id returning *"),
                    {"run_id": generation.id, "message_id": assistant_row["id"]},
                )
                .mappings()
                .one()
            )
            self._enqueue_rag(
                connection,
                user,
                generation.id,
                cast(UUID, assistant_row["id"]),
                query,
                limit,
                [],
                scope_recording_ids or [],
                command,
            )
        return self._conversation(conversation_row), self._message(user_row), self._message(assistant_row), [], True

    def mark_streaming(self, generation_run_id: UUID) -> None:
        self._update_from_generation(generation_run_id, ConversationMessageStatus.STREAMING)

    def generation_command(self, generation_run_id: UUID) -> CreateGenerationCommand:
        return self._generation_service.command(generation_run_id)

    def resume_generation(
        self,
        user: CurrentUser,
        conversation_id: UUID,
        source_generation_id: UUID,
        client_request_id: UUID,
        *,
        reuse_checkpoint: bool,
        scope_recording_ids: list[UUID] | None = None,
    ) -> tuple[ConversationMessage, ConversationMessage, list[RagHistoryMessage]]:
        self._access.require_view(conversation_id, user)
        source = self._generation_service.get(source_generation_id)
        if source.status not in {GenerationStatus.CANCELLED, GenerationStatus.FAILED}:
            raise ValueError("Only a cancelled or failed generation can be restarted")
        source_command = self._generation_service.command(source_generation_id)
        with self._engine.begin() as connection:
            assistant_row = (
                connection.execute(
                    text(
                        """
                        select * from conversation_messages
                        where conversation_id = :conversation_id and generation_run_id = :generation_run_id
                          and role = 'assistant'
                          and not exists (
                              select 1 from conversation_messages later_message
                              where later_message.conversation_id = conversation_messages.conversation_id
                                and later_message.role = 'assistant'
                                and later_message.sequence > conversation_messages.sequence
                          )
                        for update
                        """
                    ),
                    {"conversation_id": conversation_id, "generation_run_id": source_generation_id},
                )
                .mappings()
                .one_or_none()
            )
            if assistant_row is None:
                raise ConversationNotFoundError(str(source_generation_id))
            user_row = (
                connection.execute(
                    text("select * from conversation_messages where id = :id"),
                    {"id": assistant_row["reply_to_message_id"]},
                )
                .mappings()
                .one()
            )
            base_input = {key: value for key, value in source_command.input.items() if key not in {"resume_from_generation_id", "resume_content_blocks"}}
            command = CreateGenerationCommand(
                kind=source_command.kind,
                priority=source_command.priority,
                idempotency_key=f"conversation-message:{assistant_row['id']}:{'resume' if reuse_checkpoint else 'regenerate'}:{client_request_id}",
                parent_type=source_command.parent_type,
                parent_id=source_command.parent_id,
                access_scope=source_command.access_scope,
                input=(
                    {
                        **base_input,
                        "resume_from_generation_id": str(source_generation_id),
                        "resume_content_blocks": [block.model_dump(mode="json") for block in source.blocks],
                    }
                    if reuse_checkpoint
                    else base_input
                ),
            )
            generation = self._generation_service.create_in_transaction(connection, command)
            assistant_row = (
                connection.execute(
                    text(
                        """
                        update conversation_messages
                        set generation_run_id = :generation_run_id, status = 'pending',
                            content_blocks = case when :reuse_checkpoint then content_blocks else '[]'::jsonb end,
                            sources = '[]'::jsonb, error_message = null, updated_at = now()
                        where id = :message_id returning *
                        """
                    ),
                    {
                        "generation_run_id": generation.id,
                        "message_id": assistant_row["id"],
                        "reuse_checkpoint": reuse_checkpoint,
                    },
                )
                .mappings()
                .one()
            )
            history = self._history(connection, conversation_id, int(assistant_row["sequence"]) - 1)
            self._enqueue_rag(
                connection,
                user,
                generation.id,
                cast(UUID, assistant_row["id"]),
                str(command.input["query"]),
                int(command.input.get("limit", 10)),
                history,
                scope_recording_ids or [],
                command,
                resume_from_generation_id=source_generation_id if reuse_checkpoint else None,
            )
        return self._message(user_row), self._message(assistant_row), history

    def submit_adjudication_decision(
        self,
        user: CurrentUser,
        conversation_id: UUID,
        source_generation_id: UUID,
        decision: ClaimConfirmationDecision,
        scope_recording_ids: list[UUID] | None = None,
    ) -> tuple[ConversationMessage, ConversationMessage, list[RagHistoryMessage]]:
        self._access.require_view(conversation_id, user)
        source = self._generation_service.get(source_generation_id)
        if source.status != GenerationStatus.SUCCEEDED:
            raise ValueError("Adjudication decisions require a succeeded confirmation generation")
        confirmation = next(
            (block for block in source.blocks if isinstance(block, AdjudicationConfirmationBlock)),
            None,
        )
        if confirmation is None or confirmation.request_id != decision.request_id:
            raise ValueError("Adjudication decision does not match the confirmation request")
        if confirmation.source_generation_id != source_generation_id:
            raise ValueError("Adjudication confirmation source generation mismatch")
        source_command = self._generation_service.command(source_generation_id)
        with self._engine.begin() as connection:
            assistant_row = (
                connection.execute(
                    text(
                        """
                        select * from conversation_messages
                        where conversation_id = :conversation_id and role = 'assistant'
                        order by sequence desc limit 1 for update
                        """
                    ),
                    {"conversation_id": conversation_id},
                )
                .mappings()
                .one_or_none()
            )
            if assistant_row is None:
                raise ConversationNotFoundError(str(source_generation_id))
            current_generation_id = cast(UUID | None, assistant_row["generation_run_id"])
            if current_generation_id != source_generation_id:
                if current_generation_id is not None:
                    current_command = self._generation_service.command(current_generation_id)
                    raw_current_decision = current_command.input.get("adjudication_user_decision")
                    current_decision = cast(dict[str, object], raw_current_decision) if isinstance(raw_current_decision, dict) else None
                    if current_decision is not None and str(current_decision.get("client_request_id")) == str(decision.client_request_id):
                        user_row = (
                            connection.execute(
                                text("select * from conversation_messages where id = :id"),
                                {"id": assistant_row["reply_to_message_id"]},
                            )
                            .mappings()
                            .one()
                        )
                        history = self._history(connection, conversation_id, int(assistant_row["sequence"]) - 1)
                        return self._message(user_row), self._message(assistant_row), history
                raise ValueError("Adjudication confirmation is stale or already consumed")
            user_row = (
                connection.execute(
                    text("select * from conversation_messages where id = :id"),
                    {"id": assistant_row["reply_to_message_id"]},
                )
                .mappings()
                .one()
            )
            base_input = {
                key: value
                for key, value in source_command.input.items()
                if key not in {"resume_from_generation_id", "resume_content_blocks", "adjudication_user_decision"}
            }
            command = CreateGenerationCommand(
                kind=source_command.kind,
                priority=source_command.priority,
                idempotency_key=f"conversation-message:{assistant_row['id']}:adjudication:{decision.client_request_id}",
                parent_type=source_command.parent_type,
                parent_id=source_command.parent_id,
                access_scope=source_command.access_scope,
                input={
                    **base_input,
                    "resume_from_generation_id": str(source_generation_id),
                    "adjudication_user_decision": decision.model_dump(mode="json"),
                },
            )
            generation = self._generation_service.create_in_transaction(connection, command)
            assistant_row = (
                connection.execute(
                    text(
                        """
                        update conversation_messages
                        set generation_run_id = :generation_run_id, status = 'pending',
                            content_blocks = '[]'::jsonb, sources = '[]'::jsonb,
                            error_message = null, updated_at = now()
                        where id = :message_id returning *
                        """
                    ),
                    {"generation_run_id": generation.id, "message_id": assistant_row["id"]},
                )
                .mappings()
                .one()
            )
            history = self._history(connection, conversation_id, int(assistant_row["sequence"]) - 1)
            self._enqueue_rag(
                connection,
                user,
                generation.id,
                cast(UUID, assistant_row["id"]),
                str(command.input["query"]),
                int(command.input.get("limit", 10)),
                history,
                scope_recording_ids or [],
                command,
                resume_from_generation_id=source_generation_id,
                adjudication_user_decision=decision,
            )
        return self._message(user_row), self._message(assistant_row), history

    def _enqueue_rag(
        self,
        connection: Connection,
        user: CurrentUser,
        run_id: UUID,
        conversation_message_id: UUID,
        query: str,
        limit: int,
        history: list[RagHistoryMessage],
        scope_recording_ids: list[UUID],
        command: CreateGenerationCommand,
        *,
        resume_from_generation_id: UUID | None = None,
        adjudication_user_decision: ClaimConfirmationDecision | None = None,
    ) -> None:
        if self._command_publisher is None:
            return
        self._command_publisher.enqueue_rag(
            connection,
            RagGenerationWorkItem(
                run_id=run_id,
                workspace_id=user.current_workspace_id,
                query=query,
                limit=limit,
                scope_recording_ids=scope_recording_ids,
                history=history,
                conversation_message_id=conversation_message_id,
                resume_from_generation_id=resume_from_generation_id,
                adjudication_user_decision=adjudication_user_decision,
                generation=command,
            ),
        )

    def sync_generation(self, generation_run_id: UUID) -> None:
        snapshot = self._generation_service.get(generation_run_id)
        self.sync_generation_snapshot(snapshot)

    def sync_generation_snapshot(self, snapshot: GenerationSnapshot) -> None:
        with self._engine.begin() as connection:
            completed_history = self.sync_generation_in_transaction(connection, snapshot)
        self.apply_generation_history_cache(completed_history)

    def sync_generation_in_transaction(
        self,
        connection: Connection,
        snapshot: GenerationSnapshot,
    ) -> tuple[UUID, list[tuple[RagHistoryMessage, RagHistoryMessage]]] | None:
        status = {
            GenerationStatus.SUCCEEDED: ConversationMessageStatus.COMPLETED,
            GenerationStatus.FAILED: ConversationMessageStatus.FAILED,
            GenerationStatus.CANCELLED: ConversationMessageStatus.CANCELLED,
        }.get(snapshot.status, ConversationMessageStatus.STREAMING)
        completed_history: tuple[UUID, list[tuple[RagHistoryMessage, RagHistoryMessage]]] | None = None
        row = (
            connection.execute(
                text(
                    """
                    update conversation_messages set status = :status, content_blocks = cast(:blocks as jsonb), sources = cast(:sources as jsonb),
                        error_message = :error_message, updated_at = now()
                    where generation_run_id = :generation_run_id
                    returning conversation_id
                    """
                ),
                {
                    "generation_run_id": snapshot.id,
                    "status": status.value,
                    "blocks": json.dumps([block.model_dump(mode="json") for block in snapshot.blocks]),
                    "sources": json.dumps(self._persistent_sources(snapshot.sources)),
                    "error_message": snapshot.error_message,
                },
            )
            .mappings()
            .one_or_none()
        )
        if status == ConversationMessageStatus.COMPLETED and row is not None:
            rows = (
                connection.execute(
                    text(
                        """
                        select role, content_blocks, sources from conversation_messages
                        where conversation_id = :conversation_id and status = 'completed'
                        order by sequence desc limit :limit
                        """
                    ),
                    {"conversation_id": row["conversation_id"], "limit": ConversationHistoryStore.MAX_TURNS * 2},
                )
                .mappings()
                .all()
            )
            history = self._bounded_history(
                [self._history_message(history_row) for history_row in reversed(rows) if str(history_row["role"]) in {"user", "assistant"}]
            )
            completed_history = (cast(UUID, row["conversation_id"]), self._history_turns(history))
        return completed_history

    def apply_generation_history_cache(
        self,
        completed_history: tuple[UUID, list[tuple[RagHistoryMessage, RagHistoryMessage]]] | None,
    ) -> None:
        if completed_history is not None:
            self._put_history_cache(*completed_history)

    def _update_from_generation(self, generation_run_id: UUID, status: ConversationMessageStatus) -> None:
        snapshot = self._generation_service.get(generation_run_id)
        with self._engine.begin() as connection:
            completed_history = self.sync_generation_in_transaction(
                connection,
                snapshot.model_copy(update={"status": self._generation_status(status)}),
            )
        self.apply_generation_history_cache(completed_history)

    @staticmethod
    def _generation_status(status: ConversationMessageStatus) -> GenerationStatus:
        return {
            ConversationMessageStatus.COMPLETED: GenerationStatus.SUCCEEDED,
            ConversationMessageStatus.FAILED: GenerationStatus.FAILED,
            ConversationMessageStatus.CANCELLED: GenerationStatus.CANCELLED,
        }.get(status, GenerationStatus.RUNNING)

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

    def _history(self, connection: Any, conversation_id: UUID, before_sequence: int, *, use_cache: bool = True) -> list[RagHistoryMessage]:
        if use_cache and self._history_store is not None:
            try:
                cached = self._history_store.get(conversation_id)
                if cached is not None:
                    logger.info("Conversation history loaded from Redis conversation_id=%s messages=%d", conversation_id, len(cached))
                    return cached
            except Exception:
                logger.warning("Conversation history Redis read failed; falling back to database conversation_id=%s", conversation_id, exc_info=True)
        rows = (
            connection.execute(
                text(
                    """
                select role, content_blocks, sources from conversation_messages
                where conversation_id = :conversation_id and sequence < :before_sequence and status = 'completed'
                order by sequence desc limit :limit
                """
                ),
                {"conversation_id": conversation_id, "before_sequence": before_sequence, "limit": ConversationHistoryStore.MAX_TURNS * 2},
            )
            .mappings()
            .all()
        )
        history = [self._history_message(row) for row in reversed(rows) if str(row["role"]) in {"user", "assistant"}]
        history = self._bounded_history(history)
        logger.info("Conversation history loaded from database conversation_id=%s messages=%d", conversation_id, len(history))
        if self._history_store is not None and history:
            try:
                self._history_store.put(conversation_id, self._history_turns(history))
            except Exception:
                logger.warning("Conversation history Redis backfill failed conversation_id=%s", conversation_id, exc_info=True)
        return history

    @staticmethod
    def _history_message(row: RowMapping) -> RagHistoryMessage:
        role = cast(Literal["user", "assistant"], str(row["role"]))
        return RagHistoryMessage(
            role=role,
            content=ConversationService._content_text(cast(list[dict[str, object]], row["content_blocks"])),
            sources=ConversationService._history_sources(row["sources"]),
        )

    @staticmethod
    def _content_text(blocks: list[dict[str, object]]) -> str:
        parts: list[str] = []
        for block in blocks:
            if block.get("type") == "text":
                parts.append(str(block.get("value", "")))
                continue
            if block.get("type") != "AGGRE_MSG":
                continue
            aggregate = block.get("sub_message")
            if not isinstance(aggregate, dict):
                continue
            aggregate_data = cast(dict[str, object], aggregate)
            group = aggregate_data.get("message_group")
            sub_messages = aggregate_data.get("sub_message_list")
            if not isinstance(group, dict) or not isinstance(sub_messages, list):
                continue
            group_data = cast(dict[str, object], group)
            primary_id = group_data.get("primary_sub_message_id")
            sub_message_items = cast(list[object], sub_messages)
            primary = next(
                (
                    cast(dict[str, object], item)
                    for item in sub_message_items
                    if isinstance(item, dict) and cast(dict[str, object], item).get("id") == primary_id
                ),
                None,
            )
            if primary is None:
                continue
            primary_blocks = primary.get("blocks")
            if not isinstance(primary_blocks, list):
                continue
            parts.extend(
                str(cast(dict[str, object], item).get("value", ""))
                for item in cast(list[object], primary_blocks)
                if isinstance(item, dict) and cast(dict[str, object], item).get("type") == "text"
            )
        return "".join(parts)

    @staticmethod
    def _history_turns(history: list[RagHistoryMessage]) -> list[tuple[RagHistoryMessage, RagHistoryMessage]]:
        turns: list[tuple[RagHistoryMessage, RagHistoryMessage]] = []
        for index in range(0, len(history) - 1, 2):
            user, assistant = history[index : index + 2]
            if user.role == "user" and assistant.role == "assistant":
                turns.append((user, assistant))
        return turns[-ConversationHistoryStore.MAX_TURNS :]

    @classmethod
    def _bounded_history(cls, history: list[RagHistoryMessage]) -> list[RagHistoryMessage]:
        bounded: list[RagHistoryMessage] = []
        for user, assistant in cls._history_turns(history):
            user_content = user.content[: ConversationHistoryStore.MAX_TURN_CHARS]
            assistant_content = assistant.content[: max(0, ConversationHistoryStore.MAX_TURN_CHARS - len(user_content))]
            bounded.extend((user.model_copy(update={"content": user_content}), assistant.model_copy(update={"content": assistant_content})))
        return bounded

    def _put_history_cache(self, conversation_id: UUID, turns: list[tuple[RagHistoryMessage, RagHistoryMessage]]) -> None:
        if self._history_store is None:
            return
        try:
            self._history_store.put(conversation_id, turns)
        except Exception:
            logger.warning("Conversation history Redis update failed conversation_id=%s", conversation_id, exc_info=True)

    def _delete_history_cache(self, conversation_id: UUID) -> None:
        if self._history_store is None:
            return
        try:
            self._history_store.delete(conversation_id)
        except Exception:
            logger.warning("Conversation history Redis invalidation failed conversation_id=%s", conversation_id, exc_info=True)

    def _delete_generation_runtime_data(self, conversation_id: UUID, generation_run_ids: list[UUID]) -> None:
        for generation_run_id in generation_run_ids:
            try:
                self._generation_service.delete_runtime_data(generation_run_id)
            except Exception:
                logger.warning(
                    "Conversation Generation Redis cleanup failed conversation_id=%s generation_id=%s",
                    conversation_id,
                    generation_run_id,
                    exc_info=True,
                )

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
            if not isinstance(recording_id, str):
                continue
            try:
                sources.append(RagHistorySource(recording_id=UUID(recording_id)))
            except TypeError, ValueError:
                continue
        return sources

    @staticmethod
    def _conversation(row: RowMapping) -> Conversation:
        return Conversation.model_validate(dict(row))

    @staticmethod
    def _message(row: RowMapping) -> ConversationMessage:
        return ConversationMessage.model_validate(dict(row))
