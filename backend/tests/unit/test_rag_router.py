from datetime import date, datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

from l2_core.rag.contracts import InferredFilters, RagRoute, ResolvedFilters, TimeRange
from l2_core.rag.retrieval import RagRetriever
from l2_core.rag.routing import parse_route_response
from l2_core.rag.scope import make_filters, resolve_date_range


def test_relative_date_is_resolved_by_python_not_the_model() -> None:
    route = RagRoute(
        status="resolved",
        strategy="chunk_search",
        topic="硅光耦合",
        time_range=TimeRange(text="昨天", kind="calendar_period", unit="day", offset=-1),
    )

    assert resolve_date_range(route, date(2026, 7, 20)) == (
        datetime(2026, 7, 19, tzinfo=ZoneInfo("Asia/Shanghai")),
        datetime(2026, 7, 20, tzinfo=ZoneInfo("Asia/Shanghai")),
    )


def test_absolute_dates_from_a_validated_route_are_preserved() -> None:
    route = RagRoute.model_validate(
        {
            "status": "resolved",
            "strategy": "scope_summary",
            "time_range": {"text": "2026-07-01 至 2026-07-08", "kind": "absolute_range"},
        }
    )

    assert resolve_date_range(route, date(2026, 7, 20)) == (
        datetime(2026, 7, 1, tzinfo=ZoneInfo("Asia/Shanghai")),
        datetime(2026, 7, 9, tzinfo=ZoneInfo("Asia/Shanghai")),
    )


def test_chinese_absolute_range_is_calculated_by_python() -> None:
    route = RagRoute(
        status="resolved",
        strategy="scope_summary",
        time_range=TimeRange(text="2026 年 7 月 1 日到 7 月 15 日", kind="absolute_range"),
    )

    assert resolve_date_range(route, date(2026, 7, 20)) == (
        datetime(2026, 7, 1, tzinfo=ZoneInfo("Asia/Shanghai")),
        datetime(2026, 7, 16, tzinfo=ZoneInfo("Asia/Shanghai")),
    )


def test_date_filters_bind_timezone_aware_datetimes_without_database_casts() -> None:
    clauses: list[str] = []
    values: dict[str, object] = {}
    timezone = ZoneInfo("Asia/Shanghai")

    RagRetriever._append_recording_filters(  # pyright: ignore[reportPrivateUsage]
        clauses,
        values,
        ResolvedFilters(
            created_from=datetime(2026, 7, 1, tzinfo=timezone),
            created_to=datetime(2026, 7, 2, tzinfo=timezone),
        ),
    )

    rendered = "\n".join(clauses)
    assert "cast(:created_from as timestamptz)" not in rendered
    assert "cast(:created_to as timestamptz)" not in rendered
    assert values["created_from"] == datetime(2026, 7, 1, tzinfo=timezone)
    assert values["created_to"] == datetime(2026, 7, 2, tzinfo=timezone)


def test_ambiguous_route_uses_explicit_status_and_error_code() -> None:
    route = parse_route_response(
        '{"status":"ambiguous","strategy":null,"topic":null,"inferred_filters":{},"error_code":"ambiguous_recording_scope","reason":"存在多个合理范围"}'
    )

    assert route is not None
    assert route.status == "ambiguous"
    assert route.error_code == "ambiguous_recording_scope"


def test_ambiguous_route_with_guessed_scope_is_rejected() -> None:
    route = parse_route_response(
        '{"status":"ambiguous","strategy":null,"topic":null,"recording_rank":1,"inferred_filters":{},"error_code":"ambiguous_recording_scope"}'
    )

    assert route is None


def test_structured_time_range_missing_required_unit_is_rejected() -> None:
    route = parse_route_response(
        '{"status":"resolved","strategy":"scope_summary","topic":null,"time_range":{"text":"上周","kind":"calendar_period","offset":-1},"inferred_filters":{}}'
    )

    assert route is None


def test_legacy_time_expression_is_not_accepted() -> None:
    route = parse_route_response('{"status":"resolved","strategy":"scope_summary","topic":null,"time_expression":"昨天","inferred_filters":{}}')

    assert route is None


def test_ranked_scope_with_no_result_matches_nothing() -> None:
    accessible_ids = [uuid4(), uuid4()]
    route = RagRoute(status="resolved", strategy="scope_summary", recording_limit=1)

    filters = make_filters(route, [], accessible_ids)

    assert filters.match_none
    assert filters.recording_ids == []


def test_absent_ranked_scope_keeps_all_accessible_recordings() -> None:
    accessible_ids = [uuid4(), uuid4()]
    route = RagRoute(status="resolved", strategy="chunk_search", topic="硅光")

    filters = make_filters(route, None, accessible_ids)

    assert not filters.match_none
    assert filters.recording_ids == accessible_ids


def test_inaccessible_explicit_recording_matches_nothing_without_uuid_sentinel() -> None:
    route = RagRoute(
        status="resolved",
        strategy="scope_summary",
        inferred_filters=InferredFilters(recording_ids=[uuid4()]),
    )

    filters = make_filters(route, None, [uuid4()])

    assert filters.match_none
    assert filters.recording_ids == []
