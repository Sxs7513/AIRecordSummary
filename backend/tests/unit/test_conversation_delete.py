from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

from sqlalchemy import Engine

from l2_core.auth.contracts import CurrentUser
from l2_core.conversations.history_store import ConversationHistoryStore
from l2_core.conversations.service import ConversationService
from l2_core.generation.service import GenerationService


def test_delete_conversation_removes_all_generation_redis_runtime_data() -> None:
    conversation_id = uuid4()
    user_id = uuid4()
    generation_ids = [uuid4(), uuid4()]
    user = CurrentUser(
        id=user_id,
        email="user@example.com",
        display_name="User",
        current_workspace_id=uuid4(),
        memberships=(),
    )
    owner_result = MagicMock()
    owner_result.scalar_one_or_none.return_value = conversation_id
    active_result = MagicMock()
    active_result.scalar_one_or_none.return_value = None
    generations_result = MagicMock()
    generations_result.scalars.return_value.all.return_value = generation_ids
    archive_result = MagicMock()
    connection = MagicMock()
    connection.execute.side_effect = [owner_result, active_result, generations_result, archive_result]
    engine = MagicMock()
    engine.begin.return_value.__enter__.return_value = connection
    generation_service = MagicMock()
    history_store = MagicMock()
    service = ConversationService(
        cast(Engine, engine),
        cast(GenerationService, generation_service),
        cast(ConversationHistoryStore, history_store),
    )
    service._access = MagicMock()  # pyright: ignore[reportPrivateUsage]

    service.delete(user, conversation_id)

    history_store.delete.assert_called_once_with(conversation_id)
    assert [call.args[0] for call in generation_service.delete_runtime_data.call_args_list] == generation_ids
    generation_query = str(connection.execute.call_args_list[2].args[0])
    assert "generation_runs" in generation_query
    assert "conversation_messages" in generation_query
