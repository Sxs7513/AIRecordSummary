from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

from sqlalchemy import Engine

from l2_core.auth.contracts import CurrentUser
from l2_core.conversations.service import ConversationService
from l2_core.generation.contracts import ContentBlock
from l2_core.generation.service import GenerationService


def test_create_turn_reuses_existing_turn_with_three_queries() -> None:
    conversation_id = uuid4()
    workspace_id = uuid4()
    user_id = uuid4()
    client_message_id = uuid4()
    user_message_id = uuid4()
    assistant_message_id = uuid4()
    now = datetime.now(UTC)
    user = CurrentUser(
        id=user_id,
        email="user@example.com",
        display_name="User",
        current_workspace_id=workspace_id,
        memberships=(),
    )
    conversation = {
        "id": conversation_id,
        "workspace_id": workspace_id,
        "owner_user_id": user_id,
        "title": "已有对话",
        "archived_at": None,
        "created_at": now,
        "updated_at": now,
    }
    user_message = {
        "id": user_message_id,
        "conversation_id": conversation_id,
        "role": "user",
        "sequence": 1,
        "reply_to_message_id": None,
        "content_blocks": [{"type": "text", "value": "原问题"}],
        "sources": [],
        "generation_run_id": None,
        "status": "completed",
        "client_message_id": client_message_id,
        "error_message": None,
        "created_at": now,
        "updated_at": now,
    }
    assistant_message = {
        **user_message,
        "id": assistant_message_id,
        "role": "assistant",
        "sequence": 2,
        "reply_to_message_id": user_message_id,
        "content_blocks": [{"type": "text", "value": "原回答"}],
        "generation_run_id": uuid4(),
        "client_message_id": None,
    }

    locked_conversation_result = MagicMock()
    locked_conversation_result.mappings.return_value.one_or_none.return_value = conversation
    existing_turn_result = MagicMock()
    existing_turn_result.mappings.return_value.all.return_value = [user_message, assistant_message]
    history_result = MagicMock()
    history_result.mappings.return_value.all.return_value = []
    connection = MagicMock()
    connection.execute.side_effect = [locked_conversation_result, existing_turn_result, history_result]
    engine_mock = MagicMock()
    engine_mock.begin.return_value.__enter__.return_value = connection
    service = ConversationService(cast(Engine, engine_mock), cast(GenerationService, MagicMock()))

    returned_user, returned_assistant, history = service.create_turn(
        user,
        conversation_id,
        [ContentBlock(value="原问题")],
        client_message_id,
        8,
    )

    assert returned_user.id == user_message_id
    assert returned_assistant.id == assistant_message_id
    assert history == []
    assert connection.execute.call_count == 3
    engine_mock.connect.assert_not_called()
    lock_sql = str(connection.execute.call_args_list[0].args[0])
    assert "workspace_id = :workspace_id" in lock_sql
    assert "owner_user_id = :user_id" in lock_sql
    assert "archived_at is null" in lock_sql
    assert "for update" in lock_sql
    existing_sql = str(connection.execute.call_args_list[1].args[0])
    assert "with existing_turn" in existing_sql
    assert "select message.*" in existing_sql


def test_create_initial_turn_reuses_client_creation_id() -> None:
    conversation_id = uuid4()
    workspace_id = uuid4()
    user_id = uuid4()
    client_creation_id = uuid4()
    client_message_id = uuid4()
    user_message_id = uuid4()
    now = datetime.now(UTC)
    user = CurrentUser(
        id=user_id,
        email="user@example.com",
        display_name="User",
        current_workspace_id=workspace_id,
        memberships=(),
    )
    conversation = {
        "id": conversation_id,
        "workspace_id": workspace_id,
        "owner_user_id": user_id,
        "title": "原问题",
        "archived_at": None,
        "created_at": now,
        "updated_at": now,
    }
    user_message = {
        "id": user_message_id,
        "conversation_id": conversation_id,
        "role": "user",
        "sequence": 1,
        "reply_to_message_id": None,
        "content_blocks": [{"type": "text", "value": "原问题"}],
        "sources": [],
        "generation_run_id": None,
        "status": "completed",
        "client_message_id": client_message_id,
        "error_message": None,
        "created_at": now,
        "updated_at": now,
    }
    assistant_message = {
        **user_message,
        "id": uuid4(),
        "role": "assistant",
        "sequence": 2,
        "reply_to_message_id": user_message_id,
        "content_blocks": [],
        "generation_run_id": uuid4(),
        "client_message_id": None,
        "status": "pending",
    }

    insert_result = MagicMock()
    insert_result.mappings.return_value.one_or_none.return_value = None
    existing_conversation_result = MagicMock()
    existing_conversation_result.mappings.return_value.one_or_none.return_value = conversation
    existing_messages_result = MagicMock()
    existing_messages_result.mappings.return_value.all.return_value = [user_message, assistant_message]
    connection = MagicMock()
    connection.execute.side_effect = [insert_result, existing_conversation_result, existing_messages_result]
    engine_mock = MagicMock()
    engine_mock.begin.return_value.__enter__.return_value = connection
    generation_service = MagicMock()
    service = ConversationService(cast(Engine, engine_mock), cast(GenerationService, generation_service))

    returned_conversation, returned_user, returned_assistant, history, created = service.create_initial_turn(
        user,
        client_creation_id,
        [ContentBlock(value="原问题")],
        client_message_id,
        8,
    )

    assert returned_conversation.id == conversation_id
    assert returned_user.id == user_message_id
    assert returned_assistant.generation_run_id == assistant_message["generation_run_id"]
    assert history == []
    assert created is False
    generation_service.create_in_transaction.assert_not_called()
    assert "on conflict" in str(connection.execute.call_args_list[0].args[0]).lower()
