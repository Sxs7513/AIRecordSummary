from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import Engine

from l2_core.auth.contracts import CurrentUser
from l2_core.conversations.service import ConversationService
from l2_core.generation.contracts import (
    CreateGenerationCommand,
    GenerationAccessScope,
    GenerationKind,
    GenerationPriority,
    GenerationSnapshot,
    GenerationStatus,
)
from l2_core.generation.service import GenerationService


def test_failed_generation_can_retry_from_its_checkpoint() -> None:
    conversation_id = uuid4()
    user_id = uuid4()
    source_generation_id = uuid4()
    new_generation_id = uuid4()
    assistant_id = uuid4()
    user_message_id = uuid4()
    now = datetime.now(UTC)
    user = CurrentUser(
        id=user_id,
        email="user@example.com",
        display_name="User",
        current_workspace_id=uuid4(),
        memberships=(),
    )
    source = _snapshot(source_generation_id, GenerationStatus.FAILED, now)
    source_command = CreateGenerationCommand(
        kind=GenerationKind.TEXT,
        priority=GenerationPriority.INTERACTIVE,
        idempotency_key="original",
        parent_type="conversation_message",
        parent_id=str(assistant_id),
        access_scope=GenerationAccessScope(owner_user_id=user_id),
        input={"query": "问题", "limit": 10},
    )
    assistant_row = _message(
        assistant_id,
        conversation_id,
        "assistant",
        2,
        source_generation_id,
        reply_to_message_id=user_message_id,
        status="failed",
        now=now,
    )
    user_row = _message(user_message_id, conversation_id, "user", 1, None, now=now)
    retried_row = {**assistant_row, "generation_run_id": new_generation_id, "status": "pending", "error_message": None}
    assistant_result = MagicMock()
    assistant_result.mappings.return_value.one_or_none.return_value = assistant_row
    user_result = MagicMock()
    user_result.mappings.return_value.one.return_value = user_row
    update_result = MagicMock()
    update_result.mappings.return_value.one.return_value = retried_row
    history_result = MagicMock()
    history_result.mappings.return_value.all.return_value = []
    connection = MagicMock()
    connection.execute.side_effect = [assistant_result, user_result, update_result, history_result]
    engine = MagicMock()
    engine.begin.return_value.__enter__.return_value = connection
    generation_service = MagicMock()
    generation_service.get.return_value = source
    generation_service.command.return_value = source_command
    generation_service.create_in_transaction.return_value = MagicMock(id=new_generation_id)
    service = ConversationService(cast(Engine, engine), cast(GenerationService, generation_service))
    service._access = MagicMock()  # pyright: ignore[reportPrivateUsage]

    _, assistant, _ = service.resume_generation(
        user,
        conversation_id,
        source_generation_id,
        uuid4(),
        reuse_checkpoint=True,
    )

    command = generation_service.create_in_transaction.call_args.args[1]
    assert command.input["resume_from_generation_id"] == str(source_generation_id)
    assert assistant.generation_run_id == new_generation_id
    assert assistant.status.value == "pending"
    assistant_query = str(connection.execute.call_args_list[0].args[0])
    assert "not exists" in assistant_query.lower()
    assert "later_message.sequence > conversation_messages.sequence" in assistant_query


def test_succeeded_generation_cannot_be_restarted() -> None:
    source_generation_id = uuid4()
    generation_service = MagicMock()
    generation_service.get.return_value = _snapshot(source_generation_id, GenerationStatus.SUCCEEDED, datetime.now(UTC))
    service = ConversationService(cast(Engine, MagicMock()), cast(GenerationService, generation_service))
    service._access = MagicMock()  # pyright: ignore[reportPrivateUsage]
    user = CurrentUser(
        id=uuid4(),
        email="user@example.com",
        display_name="User",
        current_workspace_id=uuid4(),
        memberships=(),
    )

    with pytest.raises(ValueError, match="cancelled or failed"):
        service.resume_generation(user, uuid4(), source_generation_id, uuid4(), reuse_checkpoint=True)


def _snapshot(run_id: object, status: GenerationStatus, now: datetime) -> GenerationSnapshot:
    return GenerationSnapshot(
        id=run_id,
        kind=GenerationKind.TEXT,
        priority=GenerationPriority.INTERACTIVE,
        status=status,
        phase=None,
        progress_percent=None,
        blocks=[],
        output=None,
        last_sequence=0,
        cancel_requested=False,
        error_code="test" if status == GenerationStatus.FAILED else None,
        error_message="failed" if status == GenerationStatus.FAILED else None,
        created_at=now,
        started_at=now,
        finished_at=now,
        updated_at=now,
    )


def _message(
    message_id: object,
    conversation_id: object,
    role: str,
    sequence: int,
    generation_run_id: object | None,
    *,
    reply_to_message_id: object | None = None,
    status: str = "completed",
    now: datetime,
) -> dict[str, object]:
    return {
        "id": message_id,
        "conversation_id": conversation_id,
        "role": role,
        "sequence": sequence,
        "reply_to_message_id": reply_to_message_id,
        "content_blocks": [{"type": "text", "value": "问题"}] if role == "user" else [],
        "sources": [],
        "generation_run_id": generation_run_id,
        "status": status,
        "client_message_id": uuid4() if role == "user" else None,
        "error_message": "failed" if status == "failed" else None,
        "created_at": now,
        "updated_at": now,
    }
