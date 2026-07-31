from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from l2_core.rag.contracts import InferredFilters, RagRoute

ROUTE_UNRESOLVED_MESSAGE = "暂时无法理解你的问题。请换一种更明确的说法，例如说明要查找的录音范围或具体话题。"
AMBIGUOUS_RECORDING_SCOPE_MESSAGE = "我还不能确定你指的是对话中此前提到的录音，还是录音库中按上传时间排序的录音。请明确一下录音范围。"


def parse_route_response(raw: str) -> RagRoute | None:
    """Parse a meaningful local-model route; an absent or invalid route must not trigger retrieval."""
    try:
        payload = json.loads(_first_json_object(raw))
        route = RagRoute.model_validate(_normalize(payload))
    except ValueError, TypeError, json.JSONDecodeError:
        return None
    if route.status != "resolved":
        expected_error = "ambiguous_recording_scope" if route.status == "ambiguous" else None
        valid_error = (
            route.error_code == expected_error if expected_error is not None else route.error_code in {"unresolved_query", "unsupported_time_expression"}
        )
        return route if valid_error and not _has_recording_scope(route) else None
    return route if route.error_code is None and route.strategy_id is not None else None


def _first_json_object(value: str) -> str:
    start = value.find("{")
    if start < 0:
        raise ValueError("Router output contains no JSON object")
    depth = 0
    quoted = False
    escaped = False
    for index, char in enumerate(value[start:], start):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return value[start : index + 1]
    raise ValueError("Router JSON object is incomplete")


def _normalize(value: object) -> dict[str, object]:
    raw = _object_mapping(value)
    if not raw:
        raise ValueError("Router output is empty")
    if "time_expression" in raw or "timeExpression" in raw:
        raise ValueError("Legacy time expression is not supported")
    filters = _object_mapping(raw.get("inferred_filters") or raw.get("filters"))
    status = str(raw.get("status") or "")
    reason = str(raw.get("reason") or "")
    if not status:
        status = "ambiguous" if reason.strip().lower() == "ambiguous_recording_scope" else "resolved"
    if status not in {"resolved", "ambiguous", "unresolved"}:
        raise ValueError("Router output has no valid status")
    strategy_value = raw.get("strategy_id", raw.get("strategy"))
    strategy = str(strategy_value or "")
    strategy = {
        "recent_recording_summary": "scope_summary",
        "date_range_summary": "scope_summary",
        "vector_search": "fact_lookup",
        "chunk_search": "fact_lookup",
    }.get(strategy, strategy)
    if status == "resolved" and strategy not in {"fact_lookup", "metadata_lookup", "scope_summary"}:
        raise ValueError("Resolved route has no valid strategy")
    if status != "resolved":
        strategy = None
    error_code = raw.get("error_code", raw.get("errorCode"))
    if status == "ambiguous" and not error_code:
        error_code = "ambiguous_recording_scope"
    elif status == "unresolved" and not error_code:
        error_code = "unresolved_query"
    recording_limit = _optional_positive_int(raw.get("recording_limit", raw.get("recordingLimit")))
    recording_rank = _optional_positive_int(raw.get("recording_rank", raw.get("recordingRank")))
    return {
        "status": status,
        "strategy_id": strategy,
        "content_query": raw.get("content_query", raw.get("contentQuery")),
        "recording_limit": recording_limit,
        "recording_rank": recording_rank,
        "time_range": raw.get("time_range"),
        "inferred_filters": InferredFilters.model_validate(
            {
                "person_names": filters.get("person_names", filters.get("personNames", [])),
                "file_names": filters.get("file_names", filters.get("fileNames", [])),
                "locations": filters.get("locations", filters.get("location", [])),
                "target_person_only": filters.get("target_person_only", filters.get("targetPersonOnly", False)),
                "recording_ids": filters.get("recording_ids", filters.get("recordingIds", [])),
                "speaker_profile_ids": filters.get("speaker_profile_ids", filters.get("speakerProfileIds", [])),
            }
        ),
        "error_code": error_code,
        "reason": reason or "router_output_normalized",
    }


def _object_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    mapping = cast(Mapping[object, object], value)
    return {key: item for key, item in mapping.items() if isinstance(key, str)}


def _optional_positive_int(value: object) -> object:
    """Treat a local model's zero placeholder as an omitted optional selector."""
    return None if value == 0 else value


def _has_recording_scope(route: RagRoute) -> bool:
    inferred = route.inferred_filters
    return any(
        (
            route.recording_limit is not None,
            route.recording_rank is not None,
            route.time_range is not None,
            bool(inferred.recording_ids),
            bool(inferred.speaker_profile_ids),
            bool(inferred.person_names),
            bool(inferred.file_names),
            bool(inferred.locations),
            inferred.target_person_only,
        )
    )
