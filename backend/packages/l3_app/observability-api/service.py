from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from repository import ObservabilityRepository


class ObservabilityService:
    def __init__(self, repository: ObservabilityRepository) -> None:
        self._repository = repository

    def overview(self, workspace_id: UUID, start: datetime | None, end: datetime | None) -> dict[str, object]:
        resolved_end = end or datetime.now(UTC)
        resolved_start = start or resolved_end - timedelta(days=7)
        self._validate_range(resolved_start, resolved_end)
        return self._repository.overview(workspace_id, resolved_start, resolved_end)

    def list_runs(
        self,
        workspace_id: UUID,
        user_id: UUID,
        start: datetime | None,
        end: datetime | None,
        limit: int,
        offset: int,
    ) -> list[dict[str, object]]:
        resolved_end = end or datetime.now(UTC)
        resolved_start = start or resolved_end - timedelta(days=7)
        self._validate_range(resolved_start, resolved_end)
        return self._repository.list_runs(workspace_id, user_id, resolved_start, resolved_end, limit, offset)

    def run_detail(self, workspace_id: UUID, run_id: UUID) -> dict[str, object] | None:
        return self._repository.run_detail(workspace_id, run_id)

    def run_conversation(self, workspace_id: UUID, run_id: UUID) -> dict[str, object] | None:
        return self._repository.run_conversation(workspace_id, run_id)

    @staticmethod
    def _validate_range(start: datetime, end: datetime) -> None:
        if start >= end:
            raise ValueError("start must be earlier than end")
        if end - start > timedelta(days=31):
            raise ValueError("time range cannot exceed 31 days")
