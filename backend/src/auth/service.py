from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from secrets import token_urlsafe
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, text

from auth.contracts import CurrentUser, WorkspaceMembership
from auth.passwords import hash_password, verify_password


class AuthenticationError(PermissionError):
    """Raised for missing or invalid login credentials/session."""


class AuthService:
    """Database-backed password login and opaque-session lifecycle."""

    def __init__(self, engine: Engine, session_ttl_days: int) -> None:
        self._engine = engine
        self._session_ttl = timedelta(days=session_ttl_days)

    def login(self, email: str, password: str) -> tuple[CurrentUser, str]:
        normalized_email = email.strip().lower()
        with self._engine.begin() as connection:
            row = (
                connection.execute(text("select id, password_hash, status from users where email = :email"), {"email": normalized_email})
                .mappings()
                .one_or_none()
            )
            if row is None or row["status"] != "active" or not verify_password(password, str(row["password_hash"])):
                raise AuthenticationError("Invalid email or password")
            token = token_urlsafe(32)
            connection.execute(
                text(
                    """
                    insert into user_sessions (user_id, token_hash, expires_at)
                    values (:user_id, :token_hash, :expires_at)
                    """
                ),
                {"user_id": row["id"], "token_hash": _token_hash(token), "expires_at": datetime.now(UTC) + self._session_ttl},
            )
        return self.require_session(token), token

    def require_session(self, token: str | None) -> CurrentUser:
        if not token:
            raise AuthenticationError("Authentication required")
        with self._engine.begin() as connection:
            user_row = (
                connection.execute(
                    text(
                        """
                    select users.id, users.email, users.display_name, users.current_workspace_id
                    from user_sessions
                    join users on users.id = user_sessions.user_id
                    where user_sessions.token_hash = :token_hash
                      and user_sessions.revoked_at is null
                      and user_sessions.expires_at > now()
                      and users.status = 'active'
                    """
                    ),
                    {"token_hash": _token_hash(token)},
                )
                .mappings()
                .one_or_none()
            )
            if user_row is None:
                raise AuthenticationError("Authentication required")
            memberships = _memberships(connection, UUID(str(user_row["id"])))
            workspace_id = user_row["current_workspace_id"]
            if workspace_id is None or UUID(str(workspace_id)) not in {membership.workspace_id for membership in memberships}:
                raise AuthenticationError("Account has no valid default workspace")
            connection.execute(
                text("update user_sessions set last_seen_at = now(), updated_at = now() where token_hash = :token_hash"),
                {"token_hash": _token_hash(token)},
            )
        return CurrentUser(
            id=UUID(str(user_row["id"])),
            email=str(user_row["email"]),
            display_name=str(user_row["display_name"]),
            current_workspace_id=UUID(str(workspace_id)),
            memberships=memberships,
        )

    def logout(self, token: str | None) -> None:
        if not token:
            return
        with self._engine.begin() as connection:
            connection.execute(
                text("update user_sessions set revoked_at = now(), updated_at = now() where token_hash = :token_hash and revoked_at is null"),
                {"token_hash": _token_hash(token)},
            )

    def bootstrap_local_admin(self, email: str, password: str, workspace_name: str) -> None:
        """Create the configured local administrator exactly once during database initialization."""
        normalized_email = email.strip().lower()
        with self._engine.begin() as connection:
            existing = connection.execute(text("select id from users where email = :email"), {"email": normalized_email}).scalar_one_or_none()
            if existing is not None:
                return
            workspace_id = connection.execute(
                text("insert into workspaces (name) values (:name) returning id"), {"name": workspace_name.strip() or "默认工作区"}
            ).scalar_one()
            user_id = connection.execute(
                text(
                    """
                    insert into users (email, display_name, password_hash, current_workspace_id)
                    values (:email, :display_name, :password_hash, :workspace_id)
                    returning id
                    """
                ),
                {
                    "email": normalized_email,
                    "display_name": normalized_email.split("@", maxsplit=1)[0],
                    "password_hash": hash_password(password),
                    "workspace_id": workspace_id,
                },
            ).scalar_one()
            connection.execute(
                text("insert into workspace_memberships (workspace_id, user_id, role) values (:workspace_id, :user_id, 'owner')"),
                {"workspace_id": workspace_id, "user_id": user_id},
            )


def _memberships(connection: Any, user_id: UUID) -> tuple[WorkspaceMembership, ...]:
    return tuple(
        WorkspaceMembership(workspace_id=UUID(str(row["workspace_id"])), workspace_name=str(row["workspace_name"]), role=str(row["role"]))
        for row in connection.execute(
            text(
                """
                select workspace_memberships.workspace_id, workspaces.name as workspace_name, workspace_memberships.role
                from workspace_memberships
                join workspaces on workspaces.id = workspace_memberships.workspace_id
                where workspace_memberships.user_id = :user_id
                order by workspaces.created_at
                """
            ),
            {"user_id": user_id},
        ).mappings()
    )


def _token_hash(token: str) -> str:
    return sha256(token.encode("utf-8")).hexdigest()
