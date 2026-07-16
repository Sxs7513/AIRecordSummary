from __future__ import annotations

from uuid import UUID

from sqlalchemy import Engine

from access.recordings import RecordingAccessService
from auth.contracts import CurrentUser
from generation.contracts import GenerationAccessScope
from generation.store import GenerationEventStore


class GenerationAccessService:
    """Authorize a generic generation through its persisted owner or protected subject."""

    def __init__(self, engine: Engine) -> None:
        self._store = GenerationEventStore(engine)
        self._recordings = RecordingAccessService(engine)

    def require_view(self, generation_id: UUID, user: CurrentUser) -> None:
        scope = self._store.get_access_scope(generation_id)
        if scope.owner_user_id == user.id:
            return
        self._require_subject_view(scope, user)

    def _require_subject_view(self, scope: GenerationAccessScope, user: CurrentUser) -> None:
        if scope.subject_type == "recording" and scope.subject_id is not None:
            self._recordings.require_view(scope.subject_id, user)
            return
        raise PermissionError("Generation access denied")
