from __future__ import annotations

from uuid import UUID

from sqlalchemy import Engine

from l2_core.access.recordings import RecordingAccessService
from l2_core.auth.contracts import CurrentUser
from l2_core.generation.contracts import GenerationAccessScope
from l2_core.generation.redis_runtime import GenerationRedisRuntime
from l2_core.generation.store import GenerationEventStore


class GenerationAccessService:
    """Authorize a generic generation through its persisted owner or protected subject."""

    def __init__(self, engine: Engine, redis_runtime: GenerationRedisRuntime | None = None) -> None:
        self._postgres_store = GenerationEventStore(engine)
        self._redis_runtime = redis_runtime
        self._recordings = RecordingAccessService(engine)

    def require_view(self, generation_id: UUID, user: CurrentUser) -> None:
        command = self._redis_runtime.get_command(generation_id) if self._redis_runtime else None
        scope = command.access_scope if command is not None else self._postgres_store.get_access_scope(generation_id)
        if scope.owner_user_id == user.id:
            return
        self._require_subject_view(scope, user)

    def _require_subject_view(self, scope: GenerationAccessScope, user: CurrentUser) -> None:
        if scope.subject_type == "recording" and scope.subject_id is not None:
            self._recordings.require_view(scope.subject_id, user)
            return
        raise PermissionError("Generation access denied")
