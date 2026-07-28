from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class WorkspaceMembership:
    """One user's role inside a workspace."""

    workspace_id: UUID
    workspace_name: str
    role: str


@dataclass(frozen=True, slots=True)
class CurrentUser:
    """Authenticated request identity and its validated default workspace."""

    id: UUID
    email: str
    display_name: str
    current_workspace_id: UUID
    memberships: tuple[WorkspaceMembership, ...]
