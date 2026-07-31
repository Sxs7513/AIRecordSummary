from __future__ import annotations

from collections.abc import Collection

from l2_core.auth.contracts import CurrentUser


class AuthorizationError(PermissionError):
    """Raised when an authenticated user is not allowed to perform an action."""


class AuthorizationService:
    """Framework-free, minimal authorization policy service.

    Roles remain workspace-scoped. This deliberately avoids introducing a
    permission database or a full RBAC model before the API-specific policies
    are known.
    """

    def require_authenticated(self, user: CurrentUser) -> CurrentUser:
        return user

    def require_current_workspace_role(self, user: CurrentUser, allowed_roles: Collection[str]) -> CurrentUser:
        membership = next(
            (item for item in user.memberships if item.workspace_id == user.current_workspace_id),
            None,
        )
        if membership is None or membership.role not in allowed_roles:
            raise AuthorizationError("Insufficient workspace permission")
        return user
