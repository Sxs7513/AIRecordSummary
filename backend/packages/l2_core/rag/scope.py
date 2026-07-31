from __future__ import annotations

from datetime import datetime
from uuid import UUID
from zoneinfo import ZoneInfo

from l2_core.rag.contracts import RagRoute, ResolvedFilters

RAG_TIMEZONE = ZoneInfo("Asia/Shanghai")


def resolve_time_range(route: RagRoute) -> tuple[datetime | None, datetime | None]:
    """Normalize the model-resolved half-open interval to the RAG query timezone."""
    time_range = route.time_range
    if time_range is None:
        return None, None
    start = time_range.start.astimezone(RAG_TIMEZONE) if time_range.start is not None else None
    end = time_range.end.astimezone(RAG_TIMEZONE) if time_range.end is not None else None
    return start, end


def make_filters(route: RagRoute, resolved_recording_ids: list[UUID] | None, scope_recording_ids: list[UUID]) -> ResolvedFilters:
    inferred = route.inferred_filters
    route_ids = list(scope_recording_ids)
    if route_ids:
        if inferred.recording_ids:
            route_ids = _intersection(route_ids, list(inferred.recording_ids))
        if resolved_recording_ids is not None:
            route_ids = _intersection(route_ids, resolved_recording_ids)
    created_from, created_to = resolve_time_range(route)
    return ResolvedFilters(
        match_none=not route_ids,
        recording_ids=route_ids,
        speaker_profile_ids=inferred.speaker_profile_ids,
        person_names=inferred.person_names,
        file_names=inferred.file_names,
        locations=inferred.locations,
        target_person_only=inferred.target_person_only,
        created_from=created_from,
        created_to=created_to,
    )


def _intersection(primary: list[UUID], secondary: list[UUID]) -> list[UUID]:
    secondary_set = set(secondary)
    return [item for item in primary if item in secondary_set]
