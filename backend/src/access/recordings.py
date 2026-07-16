from __future__ import annotations

from uuid import UUID

from sqlalchemy import Engine, text

from auth.contracts import CurrentUser


class RecordingAccessDeniedError(PermissionError):
    """Raised when the current user has no requested recording capability."""


class RecordingAccessService:
    """Single authority for Workspace and direct-share recording access."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def require_view(self, recording_id: UUID, user: CurrentUser) -> None:
        if not self._has_access(recording_id, user.id, allow_editor=False):
            raise RecordingAccessDeniedError(str(recording_id))

    def require_edit(self, recording_id: UUID, user: CurrentUser) -> None:
        if not self._has_access(recording_id, user.id, allow_editor=True):
            raise RecordingAccessDeniedError(str(recording_id))

    def accessible_predicate(self, *, recording_alias: str = "recordings") -> str:
        """SQL predicate for listing/searching only records visible to :current_user_id."""
        return f"""
            exists (
                select 1 from workspace_memberships workspace_access
                where workspace_access.workspace_id = {recording_alias}.workspace_id
                  and workspace_access.user_id = :current_user_id
            )
            or exists (
                select 1 from recording_memberships direct_access
                where direct_access.recording_id = {recording_alias}.id
                  and direct_access.user_id = :current_user_id
            )
        """

    def accessible_recording_ids(self, user: CurrentUser) -> list[UUID]:
        with self._engine.connect() as connection:
            return [
                UUID(str(value))
                for value in connection.execute(
                    text(f"select recordings.id from recordings where {self.accessible_predicate()}"),
                    {"current_user_id": user.id},
                ).scalars()
            ]

    def _has_access(self, recording_id: UUID, user_id: UUID, *, allow_editor: bool) -> bool:
        direct_role_clause = "and direct_access.role = 'editor'" if allow_editor else ""
        workspace_role_clause = "and workspace_access.role in ('owner', 'admin')" if allow_editor else ""
        with self._engine.connect() as connection:
            result = connection.execute(
                text(
                    f"""
                    select 1
                    from recordings
                    where recordings.id = :recording_id
                      and (
                        exists (
                            select 1 from workspace_memberships workspace_access
                            where workspace_access.workspace_id = recordings.workspace_id
                              and workspace_access.user_id = :user_id
                              {workspace_role_clause}
                        )
                        or recordings.owner_user_id = :user_id
                        or exists (
                            select 1 from recording_memberships direct_access
                            where direct_access.recording_id = recordings.id
                              and direct_access.user_id = :user_id
                              {direct_role_clause}
                        )
                      )
                    """
                ),
                {"recording_id": recording_id, "user_id": user_id},
            ).scalar_one_or_none()
        return result is not None
