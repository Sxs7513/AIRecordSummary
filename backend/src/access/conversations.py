from __future__ import annotations

from uuid import UUID

from sqlalchemy import Engine, text

from auth.contracts import CurrentUser


class ConversationAccessDeniedError(PermissionError):
    """Raised when a user is not a member of the conversation workspace."""


class ConversationAccessService:
    """Workspace-based access checks for long-lived chat conversations."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def require_view(self, conversation_id: UUID, user: CurrentUser) -> None:
        with self._engine.connect() as connection:
            allowed = connection.execute(
                text(
                    """
                    select 1 from conversations
                    join workspace_memberships on workspace_memberships.workspace_id = conversations.workspace_id
                    where conversations.id = :conversation_id and conversations.archived_at is null
                        and workspace_memberships.user_id = :user_id
                    """
                ),
                {"conversation_id": conversation_id, "user_id": user.id},
            ).scalar_one_or_none()
        if allowed is None:
            raise ConversationAccessDeniedError(str(conversation_id))
