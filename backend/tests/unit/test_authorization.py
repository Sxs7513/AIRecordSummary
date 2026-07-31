from uuid import UUID

import pytest

from l2_core.auth.authorization import AuthorizationError, AuthorizationService
from l2_core.auth.contracts import CurrentUser, WorkspaceMembership

_WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000001")


def _user(role: str) -> CurrentUser:
    return CurrentUser(
        id=UUID("00000000-0000-0000-0000-000000000002"),
        email="user@example.com",
        display_name="User",
        current_workspace_id=_WORKSPACE_ID,
        memberships=(WorkspaceMembership(_WORKSPACE_ID, "Workspace", role),),
    )


def test_authorization_allows_an_explicit_workspace_role() -> None:
    user = _user("owner")

    assert AuthorizationService().require_current_workspace_role(user, {"owner", "admin"}) is user


def test_authorization_rejects_a_role_outside_the_policy() -> None:
    with pytest.raises(AuthorizationError, match="Insufficient workspace permission"):
        AuthorizationService().require_current_workspace_role(_user("member"), {"owner", "admin"})
