from datetime import datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

from l2_core.rag.contracts import InferredFilters, RagRoute, ResolvedFilters, TimeRange
from l2_core.rag.retrieval import RagRetriever
from l2_core.rag.routing import parse_route_response
from l2_core.rag.scope import make_filters, resolve_time_range


def test_model_resolved_time_range_is_normalized_to_rag_timezone() -> None:
    route = RagRoute(
        status="resolved",
        strategy_id="fact_lookup",
        time_range=TimeRange(
            text="昨天",
            start=datetime.fromisoformat("2026-07-19T00:00:00+08:00"),
            end=datetime.fromisoformat("2026-07-20T00:00:00+08:00"),
        ),
    )

    assert resolve_time_range(route) == (
        datetime(2026, 7, 19, tzinfo=ZoneInfo("Asia/Shanghai")),
        datetime(2026, 7, 20, tzinfo=ZoneInfo("Asia/Shanghai")),
    )


def test_absolute_dates_from_a_validated_route_are_preserved() -> None:
    route = RagRoute.model_validate(
        {
            "status": "resolved",
            "strategy": "scope_summary",
            "time_range": {
                "text": "2026-07-01 至 2026-07-08",
                "start": "2026-07-01T00:00:00+08:00",
                "end": "2026-07-09T00:00:00+08:00",
            },
        }
    )

    assert resolve_time_range(route) == (
        datetime(2026, 7, 1, tzinfo=ZoneInfo("Asia/Shanghai")),
        datetime(2026, 7, 9, tzinfo=ZoneInfo("Asia/Shanghai")),
    )


def test_time_range_accepts_open_boundary_but_rejects_naive_or_reversed_values() -> None:
    open_range = TimeRange(text="从 7 月开始", start=datetime.fromisoformat("2026-07-01T00:00:00+08:00"))

    assert open_range.end is None
    for invalid in (
        {"text": "昨天", "start": "2026-07-19T00:00:00", "end": "2026-07-20T00:00:00"},
        {"text": "错误范围", "start": "2026-07-20T00:00:00+08:00", "end": "2026-07-19T00:00:00+08:00"},
        {"text": "缺少范围"},
    ):
        try:
            TimeRange.model_validate(invalid)
        except ValueError:
            continue
        raise AssertionError(f"Expected invalid time range: {invalid}")


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


def test_route_parses_file_name_as_an_exact_recording_scope_filter() -> None:
    route = parse_route_response(
        '{"status":"resolved","strategy_id":"metadata_lookup",'
        '"inferred_filters":{"file_names":["test3.m4a"]}}'
    )

    assert route is not None
    assert route.inferred_filters.file_names == ["test3.m4a"]

    filters = make_filters(route, None, [uuid4()])
    assert filters.file_names == ["test3.m4a"]


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


def test_unresolved_route_ignores_local_model_zero_placeholders() -> None:
    route = parse_route_response(
        '{"status":"unresolved","strategy":"scope_summary","topic":null,'
        '"standalone_query":"查询文本","recording_limit":0,"recording_rank":0,'
        '"time_range":null,"reason":"没有识别到检索目标"}'
    )

    assert route is not None
    assert route.status == "unresolved"
    assert route.strategy is None
    assert route.recording_limit is None
    assert route.recording_rank is None


def test_resolved_route_normalizes_unused_zero_selectors() -> None:
    route = parse_route_response(
        '{"status":"resolved","strategy":"chunk_search","recording_limit":0,"recording_rank":0}'
    )

    assert route is not None
    assert route.recording_limit is None
    assert route.recording_rank is None


def test_resolved_route_only_requires_strategy_and_scope() -> None:
    route = parse_route_response('{"status":"resolved","strategy":"chunk_search","inferred_filters":{}}')

    assert route is not None
    assert route.strategy_id == "fact_lookup"
    assert route.strategy == "chunk_search"
    assert parse_route_response('{"status":"resolved","inferred_filters":{}}') is None


def test_fact_lookup_preserves_route_content_query() -> None:
    route = parse_route_response(
        '{"status":"resolved","strategy_id":"fact_lookup",'
        '"content_query":"是否讨论了预算审批","inferred_filters":{}}'
    )

    assert route is not None
    assert route.content_query == "是否讨论了预算审批"


def test_route_accepts_new_strategy_ids_and_metadata_lookup() -> None:
    fact = parse_route_response('{"status":"resolved","strategy_id":"fact_lookup","inferred_filters":{}}')
    metadata = parse_route_response(
        '{"status":"resolved","strategy_id":"metadata_lookup","inferred_filters":{}}'
    )

    assert fact is not None and fact.strategy_id == "fact_lookup"
    assert metadata is not None and metadata.strategy_id == "metadata_lookup"
    assert metadata.strategy is None


def test_route_accepts_model_resolved_half_open_time_range() -> None:
    route = parse_route_response(
        '{"status":"resolved","strategy":"scope_summary",'
        '"time_range":{"text":"上周","start":"2026-07-27T00:00:00+08:00",'
        '"end":"2026-08-03T00:00:00+08:00"},"inferred_filters":{}}'
    )

    assert route is not None
    assert resolve_time_range(route) == (
        datetime(2026, 7, 27, tzinfo=ZoneInfo("Asia/Shanghai")),
        datetime(2026, 8, 3, tzinfo=ZoneInfo("Asia/Shanghai")),
    )


def test_legacy_semantic_time_range_is_rejected() -> None:
    route = parse_route_response(
        '{"status":"resolved","strategy":"scope_summary",'
        '"time_range":{"text":"上周","kind":"calendar_period","offset":-1},"inferred_filters":{}}'
    )

    assert route is None


def test_legacy_time_expression_is_not_accepted() -> None:
    route = parse_route_response('{"status":"resolved","strategy":"scope_summary","topic":null,"time_expression":"昨天","inferred_filters":{}}')

    assert route is None


def test_ranked_scope_with_no_result_matches_nothing() -> None:
    accessible_ids = [uuid4(), uuid4()]
    route = RagRoute(status="resolved", strategy_id="scope_summary", recording_limit=1)

    filters = make_filters(route, [], accessible_ids)

    assert filters.match_none
    assert filters.recording_ids == []


def test_absent_ranked_scope_keeps_all_accessible_recordings() -> None:
    accessible_ids = [uuid4(), uuid4()]
    route = RagRoute(status="resolved", strategy_id="fact_lookup")

    filters = make_filters(route, None, accessible_ids)

    assert not filters.match_none
    assert filters.recording_ids == accessible_ids


def test_scope_summary_without_explicit_scope_keeps_all_accessible_recordings() -> None:
    accessible_ids = [uuid4(), uuid4()]
    route = parse_route_response(
        '{"status":"resolved","strategy":"scope_summary","inferred_filters":{}}'
    )

    assert route is not None
    filters = make_filters(route, None, accessible_ids)
    assert not filters.match_none
    assert filters.recording_ids == accessible_ids


def test_inaccessible_explicit_recording_matches_nothing_without_uuid_sentinel() -> None:
    route = RagRoute(
        status="resolved",
        strategy_id="scope_summary",
        inferred_filters=InferredFilters(recording_ids=[uuid4()]),
    )

    filters = make_filters(route, None, [uuid4()])

    assert filters.match_none
    assert filters.recording_ids == []
