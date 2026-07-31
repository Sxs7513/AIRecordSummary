from __future__ import annotations

from uuid import UUID

from sqlalchemy import Engine, text

from l2_core.auth.contracts import CurrentUser


class ConversationAccessDeniedError(PermissionError):
    """Raised when a user no longer owns an active conversation."""


class ConversationAccessService:
    """Owner-based access checks for long-lived chat conversations."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def require_view(self, conversation_id: UUID, user: CurrentUser) -> None:
        with self._engine.connect() as connection:
            allowed = connection.execute(
                text(
                    """
                    select 1 from conversations
                    where id = :conversation_id
                        and workspace_id = :workspace_id
                        and owner_user_id = :user_id
                        and archived_at is null
                    """
                ),
                {
                    "conversation_id": conversation_id,
                    "workspace_id": user.current_workspace_id,
                    "user_id": user.id,
                },
            ).scalar_one_or_none()
        if allowed is None:
            raise ConversationAccessDeniedError(str(conversation_id))
