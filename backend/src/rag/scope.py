from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from uuid import UUID
from zoneinfo import ZoneInfo

from rag.contracts import RagRoute, ResolvedFilters

RAG_TIMEZONE = ZoneInfo("Asia/Shanghai")


def resolve_date_range(route: RagRoute, today: date | None = None) -> tuple[datetime | None, datetime | None]:
    """Resolve relative dates in one place rather than trusting an LLM to calculate calendar dates."""
    time_range = route.time_range
    if time_range is None:
        return None, None
    current = today or datetime.now(RAG_TIMEZONE).date()
    if time_range.kind == "relative_duration":
        return _relative_duration(current, time_range.unit, time_range.value)
    if time_range.kind == "calendar_period":
        return _calendar_period(current, time_range.unit, time_range.offset)
    return _absolute_range(time_range.text, current)


def _relative_duration(current: date, unit: str | None, value: int | None) -> tuple[datetime | None, datetime | None]:
    if unit is None or value is None:
        return None, None
    if unit == "day":
        start = current - timedelta(days=max(0, value - 1))
    elif unit == "week":
        start = current - timedelta(weeks=value)
    elif unit == "month":
        start = _shift_months(current, -value)
    elif unit == "quarter":
        start = _shift_months(current, -(value * 3))
    elif unit == "year":
        start = _shift_months(current, -(value * 12))
    else:
        return None, None
    return _dates(start, current + timedelta(days=1))


def _calendar_period(current: date, unit: str | None, offset: int | None) -> tuple[datetime | None, datetime | None]:
    period_offset = offset or 0
    if unit == "day":
        start = current + timedelta(days=period_offset)
        return _dates(start, start + timedelta(days=1))
    if unit == "week":
        start = current - timedelta(days=current.weekday()) + timedelta(weeks=period_offset)
        return _dates(start, start + timedelta(days=7))
    if unit == "month":
        start = _shift_months(current.replace(day=1), period_offset)
        return _dates(start, _shift_months(start, 1))
    if unit == "quarter":
        start = current.replace(month=((current.month - 1) // 3) * 3 + 1, day=1)
        start = _shift_months(start, period_offset * 3)
        return _dates(start, _shift_months(start, 3))
    if unit == "year":
        start = date(current.year + period_offset, 1, 1)
        return _dates(start, date(start.year + 1, 1, 1))
    return None, None


def _absolute_range(text: str, current: date) -> tuple[datetime | None, datetime | None]:
    iso_values = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
    try:
        values = [date.fromisoformat(value) for value in iso_values]
        if not values:
            current_year = current.year
            for year, month, day in re.findall(r"(?:(\d{4})\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*日", text):
                if year:
                    current_year = int(year)
                values.append(date(current_year, int(month), int(day)))
    except ValueError:
        return None, None
    if not values:
        return None, None
    start = values[0]
    end = values[1] if len(values) > 1 else start
    if end < start:
        return None, None
    return _dates(start, end + timedelta(days=1))


def _shift_months(value: date, months: int) -> date:
    target = value.year * 12 + value.month - 1 + months
    year, month_index = divmod(target, 12)
    month = month_index + 1
    next_month = date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1)
    last_day = (next_month - timedelta(days=1)).day
    return value.replace(year=year, month=month, day=min(value.day, last_day))


def make_filters(route: RagRoute, resolved_recording_ids: list[UUID] | None, scope_recording_ids: list[UUID]) -> ResolvedFilters:
    inferred = route.inferred_filters
    route_ids = list(scope_recording_ids)
    if route_ids:
        if inferred.recording_ids:
            route_ids = _intersection(route_ids, list(inferred.recording_ids))
        if resolved_recording_ids is not None:
            route_ids = _intersection(route_ids, resolved_recording_ids)
    created_from, created_to = resolve_date_range(route)
    return ResolvedFilters(
        match_none=not route_ids,
        recording_ids=route_ids,
        speaker_profile_ids=inferred.speaker_profile_ids,
        person_names=inferred.person_names,
        locations=inferred.locations,
        target_person_only=inferred.target_person_only,
        created_from=created_from,
        created_to=created_to,
    )


def _intersection(primary: list[UUID], secondary: list[UUID]) -> list[UUID]:
    secondary_set = set(secondary)
    return [item for item in primary if item in secondary_set]


def _dates(start: date, end: date) -> tuple[datetime, datetime]:
    return (
        datetime.combine(start, time.min, tzinfo=RAG_TIMEZONE),
        datetime.combine(end, time.min, tzinfo=RAG_TIMEZONE),
    )
