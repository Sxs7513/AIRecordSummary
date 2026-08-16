from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

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
from l2_core.rag.adjudication.contracts import (
    AdjudicationConfirmationBlock,
    AdjudicationConfirmationCandidate,
    AdjudicationConfirmationItem,
    ClaimConfirmationDecision,
    ClaimConfirmationItemDecision,
)


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


def test_conversation_history_uses_primary_aggregate_sub_message() -> None:
    blocks: list[dict[str, object]] = [
        {
            "type": "AGGRE_MSG",
            "id": "answer-comparison",
            "sub_message": {
                "message_group": {
                    "id": "answer-comparison",
                    "sub_message_ids": ["original-answer", "corrected-answer"],
                    "primary_sub_message_id": "corrected-answer",
                },
                "sub_message_list": [
                    {
                        "id": "original-answer",
                        "blocks": [{"type": "text", "value": "RF 的时延"}],
                    },
                    {
                        "id": "corrected-answer",
                        "blocks": [{"type": "text", "value": "I²C 的时延"}],
                    },
                ],
            },
        }
    ]

    assert ConversationService._content_text(blocks) == "I²C 的时延"  # pyright: ignore[reportPrivateUsage]


def test_succeeded_confirmation_generation_accepts_structured_adjudication_decision() -> None:
    conversation_id = uuid4()
    user_id = uuid4()
    source_generation_id = uuid4()
    new_generation_id = uuid4()
    assistant_id = uuid4()
    user_message_id = uuid4()
    request_id = uuid4()
    now = datetime.now(UTC)
    user = CurrentUser(
        id=user_id,
        email="user@example.com",
        display_name="User",
        current_workspace_id=uuid4(),
        memberships=(),
    )
    confirmation = AdjudicationConfirmationBlock(
        request_id=request_id,
        source_generation_id=source_generation_id,
        items=[
            AdjudicationConfirmationItem(
                id="p1",
                evidence_index=1,
                recording_id=uuid4(),
                chunk_id=uuid4(),
                start_ms=1_000,
                end_ms=2_000,
                original_expression="RF",
                candidates=[AdjudicationConfirmationCandidate(id="p1", expression="I²C", confidence=0.8)],
            )
        ],
    )
    source = _snapshot(source_generation_id, GenerationStatus.SUCCEEDED, now).model_copy(
        update={"blocks": [confirmation], "output": {"content_blocks": [confirmation.model_dump(mode="json")]}}
    )
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
        status="completed",
        now=now,
    )
    user_row = _message(user_message_id, conversation_id, "user", 1, None, now=now)
    resumed_row: dict[str, object] = dict(assistant_row)
    resumed_row.update({"generation_run_id": new_generation_id, "status": "pending", "content_blocks": []})
    assistant_result = MagicMock()
    assistant_result.mappings.return_value.one_or_none.return_value = assistant_row
    user_result = MagicMock()
    user_result.mappings.return_value.one.return_value = user_row
    update_result = MagicMock()
    update_result.mappings.return_value.one.return_value = resumed_row
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
    decision = ClaimConfirmationDecision(
        request_id=request_id,
        client_request_id=uuid4(),
        decisions=[
            ClaimConfirmationItemDecision(item_id="p1", action="accept_candidate", candidate_id="p1")
        ],
    )

    _user_message, assistant, _history = service.submit_adjudication_decision(
        user,
        conversation_id,
        source_generation_id,
        decision,
    )

    command = generation_service.create_in_transaction.call_args.args[1]
    assert command.input["resume_from_generation_id"] == str(source_generation_id)
    assert command.input["adjudication_user_decision"]["request_id"] == str(request_id)
    assert assistant.generation_run_id == new_generation_id
    assert assistant.status.value == "pending"


def _snapshot(run_id: UUID, status: GenerationStatus, now: datetime) -> GenerationSnapshot:
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
