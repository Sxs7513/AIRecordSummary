from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import cast
from uuid import uuid4

from langchain_core.messages import BaseMessage

from rag.contracts import (
    AnswerPlan,
    AnswerPlanItem,
    Evidence,
    EvidenceChunk,
    EvidenceFacts,
    EvidenceGrade,
    EvidenceRecording,
    InferredFilters,
    RagGraphState,
    RagHistoryMessage,
    RagHistorySource,
    RagRoute,
    ResolvedFilters,
)
from rag.graph import RagGraph
from rag.routing import AMBIGUOUS_RECORDING_SCOPE_MESSAGE, ROUTE_UNRESOLVED_MESSAGE
from rag.streaming import ThinkTagFilter


class FakeModel:
    def __init__(self) -> None:
        self.json_schemas: list[Mapping[str, object] | None] = []
        self._responses = iter(
            [
                '{"status":"resolved","strategy":"chunk_search","topic":"交付风险","inferred_filters":{}}',
                '{"sufficient":true,"planning_required":true,"planning_reason":"需要组织风险结论","reason":"enough"}',
                '{"items":[{"statement":"存在交付风险","evidence_indexes":[1]}]}',
            ]
        )

    async def complete(
        self,
        messages: Sequence[BaseMessage],
        max_tokens: int,
        temperature: float = 0.0,
        json_schema: Mapping[str, object] | None = None,
    ) -> str:
        if not hasattr(self, "json_schemas"):
            self.json_schemas = []
        self.json_schemas.append(json_schema)
        return next(self._responses)

    async def _stream(self) -> AsyncIterator[str]:
        yield "交付风险是供应延期。"

    def stream(self, messages: Sequence[BaseMessage], max_tokens: int, temperature: float = 0.1) -> AsyncIterator[str]:
        return self._stream()


class FakeRetriever:
    hybrid_search_enabled = False

    def resolve_recording_scope(self, filters: object, *args: object) -> list[object]:
        return list(getattr(filters, "recording_ids", []))

    def retrieve_chunks(self, topic: str, filters: object, limit: int, run_id: str = "standalone") -> list[Evidence]:
        recording_id = uuid4()
        return [
            Evidence(
                index=1,
                recording=EvidenceRecording(id=recording_id, title="项目周会", file_name="week.mp3"),
                chunk=EvidenceChunk(id=uuid4(), text="供应延期会影响交付。", start_ms=1_000, end_ms=2_000),
                score=0.92,
                match_type="vector",
                url=f"/recordings/{recording_id}?t=1000",
            )
        ]

    def retrieve_scope(self, *args: object) -> list[Evidence]:
        raise AssertionError("topic route must use chunk retrieval")


class JsonEventHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, object]] = []

    def emit(self, record: logging.LogRecord) -> None:
        value = json.loads(record.getMessage())
        if isinstance(value, dict):
            self.events.append(cast(dict[str, object], value))


def test_langgraph_routes_retrieves_validates_and_streams_only_final_answer() -> None:
    model = FakeModel()
    graph = RagGraph(FakeRetriever(), model)  # type: ignore[arg-type]
    deltas: list[str] = []
    phases: list[str] = []

    answer, sources, not_enough_evidence, message = asyncio.run(
        graph.run("交付风险是什么", 10, [uuid4()], lambda name, _label, _progress: phases.append(name), deltas.append)
    )

    assert answer == "交付风险是供应延期。"
    assert "routing" in phases and "generating" in phases
    assert "".join(deltas) == answer
    assert not not_enough_evidence
    assert message is None
    assert model.json_schemas[0] == RagRoute.model_json_schema()
    recording = cast(dict[str, object], sources[0]["recording"])
    assert recording == {
        "id": str(recording["id"]),
        "title": "项目周会",
        "fileName": "week.mp3",
        "location": None,
        "durationSeconds": None,
    }
    assert len(model.json_schemas) == 3


def test_planned_answer_prompt_and_sources_only_use_plan_selected_evidence() -> None:
    first_recording_id = uuid4()
    second_recording_id = uuid4()

    class TwoEvidenceRetriever(FakeRetriever):
        def retrieve_chunks(
            self,
            topic: str,
            filters: object,
            limit: int,
            run_id: str = "standalone",
        ) -> list[Evidence]:
            return [
                Evidence(
                    index=1,
                    recording=EvidenceRecording(id=first_recording_id, title="未选择录音", file_name="first.mp3"),
                    chunk=EvidenceChunk(id=uuid4(), text="未选择的独特内容", start_ms=100, end_ms=200),
                    score=0.9,
                    match_type="vector",
                    url=f"/recordings/{first_recording_id}?t=100",
                ),
                Evidence(
                    index=2,
                    recording=EvidenceRecording(id=second_recording_id, title="已选择录音", file_name="second.mp3"),
                    chunk=EvidenceChunk(id=uuid4(), text="计划选择的独特内容", start_ms=300, end_ms=400),
                    score=0.8,
                    match_type="hybrid",
                    url=f"/recordings/{second_recording_id}?t=300",
                ),
            ]

    class SelectingPlanModel(FakeModel):
        def __init__(self) -> None:
            self.answer_messages: Sequence[BaseMessage] = []
            self._responses = iter(
                [
                    '{"status":"resolved","strategy":"chunk_search","topic":"比较风险","inferred_filters":{}}',
                    '{"sufficient":true,"planning_required":true,"planning_reason":"需要比较","reason":"enough"}',
                    '{"items":[{"statement":"仅回答第二条","evidence_indexes":[2]}]}',
                ]
            )

        def stream(
            self,
            messages: Sequence[BaseMessage],
            max_tokens: int,
            temperature: float = 0.1,
        ) -> AsyncIterator[str]:
            self.answer_messages = messages
            return super().stream(messages, max_tokens, temperature)

    model = SelectingPlanModel()
    answer, sources, not_enough_evidence, message = asyncio.run(
        RagGraph(TwoEvidenceRetriever(), model).run(  # type: ignore[arg-type]
            "比较风险",
            10,
            [first_recording_id, second_recording_id],
            lambda _name, _label, _progress: None,
            lambda _delta: None,
        )
    )

    answer_prompt_text = "\n".join(str(item.content) for item in model.answer_messages)
    assert answer == "交付风险是供应延期。"
    assert not not_enough_evidence
    assert message is None
    assert len(sources) == 1
    assert sources[0]["index"] == 2
    assert cast(dict[str, object], sources[0]["recording"])["id"] == str(second_recording_id)
    assert "计划选择的独特内容" in answer_prompt_text
    assert "未选择的独特内容" not in answer_prompt_text


def test_structured_logs_cover_every_planned_graph_node_and_transition() -> None:
    run_id = uuid4()
    handler = JsonEventHandler()
    rag_logger = logging.getLogger("rag")
    previous_level = rag_logger.level
    rag_logger.setLevel(logging.INFO)
    rag_logger.addHandler(handler)
    try:
        asyncio.run(
            RagGraph(FakeRetriever(), FakeModel()).run(  # type: ignore[arg-type]
                "比较交付风险",
                10,
                [uuid4()],
                lambda _name, _label, _progress: None,
                lambda _delta: None,
                run_id=run_id,
            )
        )
    finally:
        rag_logger.removeHandler(handler)
        rag_logger.setLevel(previous_level)

    node_starts = [event["node"] for event in handler.events if event["event"] == "node_started"]
    node_completions = [event["node"] for event in handler.events if event["event"] == "node_completed"]
    transitions = [
        (event["source"], event["target"]) for event in handler.events if event["event"] == "graph_transition"
    ]

    assert node_starts == [
        "route",
        "retrieve",
        "grade",
        "decide_plan",
        "plan",
        "validate_plan",
        "select_planned_evidence",
        "answer",
    ]
    assert node_completions == node_starts
    assert transitions == [
        ("route", "retrieve"),
        ("grade", "decide_plan"),
        ("decide_plan", "plan"),
    ]
    assert all(event["run_id"] == str(run_id) for event in handler.events)
    route_completed = next(
        event for event in handler.events if event["event"] == "node_completed" and event["node"] == "route"
    )
    assert '"strategy":"chunk_search"' in cast(str, route_completed["raw"])
    assert cast(dict[str, object], route_completed["resolved_filters"])["recording_scope_resolved"] is True
    assert any(event["event"] == "answer_first_token" for event in handler.events)
    assert any(event["event"] == "graph_completed" and event["status"] == "succeeded" for event in handler.events)


def test_simple_question_skips_answer_plan_and_generates_directly() -> None:
    class DirectAnswerModel(FakeModel):
        def __init__(self) -> None:
            self.json_schemas: list[Mapping[str, object] | None] = []
            self._responses = iter(
                [
                    '{"status":"resolved","strategy":"chunk_search","topic":"发布日期","inferred_filters":{}}',
                    '{"sufficient":true,"planning_required":false,"planning_reason":"单一事实","reason":"enough"}',
                ]
            )

    model = DirectAnswerModel()
    graph = RagGraph(FakeRetriever(), model)  # type: ignore[arg-type]

    answer, sources, not_enough_evidence, message = asyncio.run(
        graph.run("发布日期是什么", 10, [uuid4()], lambda _name, _label, _progress: None, lambda _delta: None)
    )

    assert answer == "交付风险是供应延期。"
    assert len(sources) == 1
    assert not not_enough_evidence
    assert message is None
    assert len(model.json_schemas) == 2


def test_hybrid_retrieval_uses_lexical_candidates_when_vector_branch_fails() -> None:
    recording_id = uuid4()
    chunk_id = uuid4()

    class DegradedHybridRetriever:
        hybrid_search_enabled = True

        def generate_query_embedding(self, query: str) -> list[float]:
            raise RuntimeError("embedding unavailable")

        def retrieve_lexical_candidates(self, query: str, filters: ResolvedFilters) -> list[dict[str, object]]:
            return [{"chunk_id": chunk_id, "recording_id": recording_id, "score": 0.8}]

        def retrieve_vector_candidates(
            self,
            embedding: list[float],
            filters: ResolvedFilters,
        ) -> list[dict[str, object]]:
            raise AssertionError("vector SQL must not run without an embedding")

        def fuse_candidates(
            self,
            vector_rows: list[dict[str, object]],
            lexical_rows: list[dict[str, object]],
            limit: int,
        ) -> list[dict[str, object]]:
            assert vector_rows == []
            assert lexical_rows[0]["chunk_id"] == chunk_id
            return [{**lexical_rows[0], "match_type": "lexical"}]

        def expand_candidates(self, rows: list[dict[str, object]]) -> list[Evidence]:
            return [
                Evidence(
                    index=1,
                    recording=EvidenceRecording(id=recording_id, title="发布会", file_name="release.mp3"),
                    chunk=EvidenceChunk(id=chunk_id, text="API v2 已发布", start_ms=100, end_ms=200),
                    score=0.8,
                    match_type="lexical",
                    url=f"/recordings/{recording_id}?t=100",
                )
            ]

    graph = RagGraph(cast(object, DegradedHybridRetriever()), FakeModel())  # type: ignore[arg-type]

    evidence = asyncio.run(
        graph._retrieve_chunks(  # pyright: ignore[reportPrivateUsage]
            "API v2",
            ResolvedFilters(recording_scope_resolved=True, recording_ids=[recording_id]),
            10,
            str(uuid4()),
        )
    )

    assert len(evidence) == 1
    assert evidence[0].match_type == "lexical"


def test_think_tag_filter_hides_split_thinking_blocks_without_leaking_them() -> None:
    visible: list[str] = []
    stream = ThinkTagFilter(visible.append)

    stream.feed("回答前<thi")
    stream.feed("nk>内部推理")
    stream.feed("不应出现</think>回答后")
    stream.feed("</think>\n最终答案")
    stream.finish()

    assert "".join(visible) == "回答前回答后\n最终答案"


def test_unresolved_route_stops_before_retrieval_and_returns_a_rephrase_message() -> None:
    class UnresolvedRouteModel(FakeModel):
        def __init__(self) -> None:
            self._responses = iter(["这不是 JSON"])

    graph = RagGraph(FakeRetriever(), UnresolvedRouteModel())  # type: ignore[arg-type]
    phases: list[str] = []

    answer, sources, not_enough_evidence, message = asyncio.run(
        graph.run("嗯", 10, [], lambda name, _label, _progress: phases.append(name), lambda _delta: None)
    )

    assert answer == ROUTE_UNRESOLVED_MESSAGE
    assert sources == []
    assert not_enough_evidence
    assert message == "route_unresolved"
    assert phases == ["routing"]


def test_repetitive_non_topic_route_stops_before_retrieval() -> None:
    class RepetitiveTopicModel(FakeModel):
        def __init__(self) -> None:
            self._responses = iter(['{"status":"resolved","strategy":"chunk_search","topic":"哈哈哈","inferred_filters":{}}'])

    graph = RagGraph(FakeRetriever(), RepetitiveTopicModel())  # type: ignore[arg-type]
    answer, sources, not_enough_evidence, message = asyncio.run(graph.run("哈哈哈", 10, [], lambda _name, _label, _progress: None, lambda _delta: None))

    assert answer == ROUTE_UNRESOLVED_MESSAGE
    assert sources == []
    assert not_enough_evidence
    assert message == "route_unresolved"


def test_scope_summary_without_a_recording_scope_stops_before_retrieval() -> None:
    class EmptyScopeModel(FakeModel):
        def __init__(self) -> None:
            self._responses = iter(['{"status":"resolved","strategy":"scope_summary","topic":null,"inferred_filters":{}}'])

    graph = RagGraph(FakeRetriever(), EmptyScopeModel())  # type: ignore[arg-type]
    answer, sources, not_enough_evidence, message = asyncio.run(graph.run("讲了什么", 10, [], lambda _name, _label, _progress: None, lambda _delta: None))

    assert answer == ROUTE_UNRESOLVED_MESSAGE
    assert sources == []
    assert not_enough_evidence
    assert message == "route_unresolved"


def test_route_history_keeps_sources_in_message_order_but_answer_history_excludes_them() -> None:
    recording_id = uuid4()
    history = [
        RagHistoryMessage(role="user", content="前一条录音是什么？"),
        RagHistoryMessage(
            role="assistant",
            content="是项目周会。",
            sources=[RagHistorySource(recording_id=recording_id, title="项目周会", start_ms=1_000, end_ms=2_000)],
        ),
    ]

    assert str(recording_id) in RagGraph._route_history_sources(history)  # pyright: ignore[reportPrivateUsage]
    assert "项目周会" in RagGraph._route_history_messages(history)  # pyright: ignore[reportPrivateUsage]
    assert str(recording_id) not in RagGraph._history_text(history)  # pyright: ignore[reportPrivateUsage]


def test_ambiguous_recording_scope_stops_before_retrieval() -> None:
    class AmbiguousRouteModel(FakeModel):
        def __init__(self) -> None:
            self._responses = iter(
                [
                    '{"status":"ambiguous","strategy":null,"topic":null,"inferred_filters":{},'
                    '"error_code":"ambiguous_recording_scope","reason":"存在多个合理范围"}'
                ]
            )

    graph = RagGraph(FakeRetriever(), AmbiguousRouteModel())  # type: ignore[arg-type]
    phases: list[str] = []
    answer, sources, not_enough_evidence, message = asyncio.run(
        graph.run("最近的录音讲了什么", 10, [], lambda name, _label, _progress: phases.append(name), lambda _delta: None)
    )

    assert answer == AMBIGUOUS_RECORDING_SCOPE_MESSAGE
    assert sources == []
    assert not_enough_evidence
    assert message == "ambiguous_recording_scope"
    assert phases == ["routing"]


def test_selected_recording_ids_must_be_currently_accessible() -> None:
    recording_id = uuid4()
    route = RagRoute(
        status="resolved",
        strategy="scope_summary",
        inferred_filters=InferredFilters(recording_ids=[recording_id]),
    )
    assert RagGraph._validate_selected_recording_ids(route, [recording_id]) is None  # pyright: ignore[reportPrivateUsage]
    assert RagGraph._validate_selected_recording_ids(route, []) == "referenced_recording_unavailable"  # pyright: ignore[reportPrivateUsage]


def test_scope_summary_with_full_text_rewrites_without_retrieval_or_rerouting() -> None:
    recording_id = uuid4()
    evidence = [
        Evidence(
            index=1,
            recording=EvidenceRecording(id=recording_id, title="项目周会", file_name="week.mp3"),
            chunk=EvidenceChunk(id=uuid4(), text="完整录音内容", start_ms=0, end_ms=1_000),
            score=1.0,
            match_type="scope",
            url=f"/recordings/{recording_id}",
        )
    ]
    filters = ResolvedFilters(recording_ids=[recording_id])
    state = cast(
        RagGraphState,
        {
            "query": "最近一条录音里讨论了什么",
            "route": RagRoute(status="resolved", strategy="scope_summary", recording_limit=1),
            "filters": filters,
            "grade": EvidenceGrade(sufficient=False, rewrite_query="硅光前景", reason="insufficient"),
            "retrieval_attempt": 0,
            "evidence": evidence,
        },
    )

    assert RagGraph._after_grade(state) == "rewrite"  # pyright: ignore[reportPrivateUsage]
    update = asyncio.run(RagGraph._rewrite(state))  # pyright: ignore[reportPrivateUsage]
    rewritten_state = cast(RagGraphState, {**state, **update})

    assert "filters" not in update
    assert "evidence" not in update
    assert rewritten_state["filters"] is filters
    assert rewritten_state["evidence"] is evidence
    assert RagGraph._after_rewrite(rewritten_state) == "grade"  # pyright: ignore[reportPrivateUsage]


def test_chunk_search_rewrite_preserves_route_and_filters() -> None:
    recording_id = uuid4()
    filters = ResolvedFilters(recording_ids=[recording_id])
    state = cast(
        RagGraphState,
        {
            "query": "原始问题",
            "route": RagRoute(status="resolved", strategy="chunk_search", topic="旧检索词"),
            "filters": filters,
            "grade": EvidenceGrade(sufficient=False, rewrite_query="新检索词", reason="insufficient"),
            "retrieval_attempt": 0,
            "evidence": [],
        },
    )

    update = asyncio.run(RagGraph._rewrite(state))  # pyright: ignore[reportPrivateUsage]
    rewritten_route = cast(RagRoute, update["route"])

    assert rewritten_route.topic == "新检索词"
    assert "filters" not in update


def test_failed_grade_stops_after_retrieval_retry_instead_of_entering_answer_plan() -> None:
    state = cast(
        RagGraphState,
        {
            "route": RagRoute(status="resolved", strategy="chunk_search", topic="硅光"),
            "grade": EvidenceGrade(sufficient=False, reason="insufficient"),
            "retrieval_attempt": 1,
        },
    )

    assert RagGraph._after_grade(state) == "done"  # pyright: ignore[reportPrivateUsage]


def test_grade_success_enters_plan_decision_before_optional_plan() -> None:
    state = cast(
        RagGraphState,
        {
            "route": RagRoute(status="resolved", strategy="chunk_search", topic="发布日期"),
            "grade": EvidenceGrade(sufficient=True, planning_required=False, planning_reason="单一事实"),
            "retrieval_attempt": 0,
        },
    )

    assert RagGraph._after_grade(state) == "decide_plan"  # pyright: ignore[reportPrivateUsage]


def test_multiple_scope_recordings_force_plan_even_when_grader_marks_direct() -> None:
    evidence = [
        Evidence(
            index=index,
            recording=EvidenceRecording(id=uuid4(), title=f"录音 {index}", file_name=f"{index}.mp3"),
            chunk=EvidenceChunk(id=uuid4(), text=f"内容 {index}", start_ms=0, end_ms=1_000),
            score=1.0,
            match_type="scope",
            url=f"/recordings/{index}",
        )
        for index in (1, 2)
    ]
    state = cast(
        RagGraphState,
        {
            "route": RagRoute(status="resolved", strategy="scope_summary", recording_limit=2),
            "grade": EvidenceGrade(sufficient=True, planning_required=False, planning_reason="模型判断为简单问题"),
            "evidence": evidence,
        },
    )

    update = asyncio.run(RagGraph._decide_plan(state))  # pyright: ignore[reportPrivateUsage]
    decided_state = cast(RagGraphState, {**state, **update})

    assert update["planning_required"] is True
    assert RagGraph._after_plan_decision(decided_state) == "plan"  # pyright: ignore[reportPrivateUsage]


def test_single_recording_direct_grade_skips_plan() -> None:
    recording_id = uuid4()
    state = cast(
        RagGraphState,
        {
            "route": RagRoute(status="resolved", strategy="chunk_search", topic="发布日期"),
            "grade": EvidenceGrade(sufficient=True, planning_required=False, planning_reason="单一事实"),
            "evidence": [
                Evidence(
                    index=1,
                    recording=EvidenceRecording(id=recording_id, title="项目周会", file_name="week.mp3"),
                    chunk=EvidenceChunk(id=uuid4(), text="发布日期是 8 月 1 日。", start_ms=0, end_ms=1_000),
                    score=0.9,
                    match_type="vector",
                    url=f"/recordings/{recording_id}",
                )
            ],
        },
    )

    update = asyncio.run(RagGraph._decide_plan(state))  # pyright: ignore[reportPrivateUsage]
    decided_state = cast(RagGraphState, {**state, **update})

    assert update["planning_required"] is False
    assert RagGraph._after_plan_decision(decided_state) == "select_direct_evidence"  # pyright: ignore[reportPrivateUsage]


def test_invalid_answer_plan_evidence_indexes_use_a_valid_fallback() -> None:
    recording_id = uuid4()
    evidence = Evidence(
        index=3,
        recording=EvidenceRecording(id=recording_id, title="项目周会", file_name="week.mp3"),
        chunk=EvidenceChunk(id=uuid4(), text="硅光方案包含集成和封装路径。", start_ms=0, end_ms=1_000),
        score=0.9,
        match_type="vector",
        url=f"/recordings/{recording_id}",
    )
    state = cast(
        RagGraphState,
        {
            "evidence": [evidence],
            "answer_plan": AnswerPlan(items=[AnswerPlanItem(statement="无效引用", evidence_indexes=[99])]),
        },
    )

    update = asyncio.run(RagGraph._validate_plan(state))  # pyright: ignore[reportPrivateUsage]
    plan = cast(AnswerPlan, update["answer_plan"])

    assert plan.items[0].evidence_indexes == [3]
    assert "硅光方案" in plan.items[0].statement


def test_answer_plan_validation_removes_invalid_and_duplicate_indexes_without_discarding_valid_items() -> None:
    recording_id = uuid4()
    evidence = Evidence(
        index=3,
        recording=EvidenceRecording(id=recording_id, title="项目周会", file_name="week.mp3"),
        chunk=EvidenceChunk(id=uuid4(), text="有效证据", start_ms=0, end_ms=1_000),
        score=0.9,
        match_type="vector",
        url=f"/recordings/{recording_id}",
    )
    state = cast(
        RagGraphState,
        {
            "evidence": [evidence],
            "answer_plan": AnswerPlan(
                items=[
                    AnswerPlanItem(statement="保留该计划项", evidence_indexes=[3, 99, 3]),
                    AnswerPlanItem(statement="删除该计划项", evidence_indexes=[98]),
                ]
            ),
        },
    )

    update = asyncio.run(RagGraph._validate_plan(state))  # pyright: ignore[reportPrivateUsage]
    plan = cast(AnswerPlan, update["answer_plan"])

    assert len(plan.items) == 1
    assert plan.items[0].statement == "保留该计划项"
    assert plan.items[0].evidence_indexes == [3]


def test_chunk_search_grades_original_query_without_replacing_it_with_retrieval_topic() -> None:
    class CapturingGradeModel(FakeModel):
        def __init__(self) -> None:
            self.messages: list[Sequence[BaseMessage]] = []

        async def complete(
            self,
            messages: Sequence[BaseMessage],
            max_tokens: int,
            temperature: float = 0.0,
            json_schema: Mapping[str, object] | None = None,
        ) -> str:
            self.messages.append(messages)
            return '{"sufficient":true,"reason":"enough"}'

    recording_id = uuid4()
    evidence = Evidence(
        index=1,
        recording=EvidenceRecording(id=recording_id, title="最近录音", file_name="latest.mp3"),
        chunk=EvidenceChunk(id=uuid4(), text="硅光具有较好的发展前景。", start_ms=0, end_ms=1_000),
        score=0.9,
        match_type="vector",
        url=f"/recordings/{recording_id}",
    )
    state = cast(
        RagGraphState,
        {
            "query": "最近的一个录音里关于硅光的前景都讨论了什么",
            "retrieval_query": "最近的一个录音里关于硅光的前景都讨论了什么",
            "route": RagRoute(status="resolved", strategy="chunk_search", topic="硅光的前景", recording_limit=1),
            "evidence": [evidence],
        },
    )
    model = CapturingGradeModel()
    graph = RagGraph(FakeRetriever(), model)  # type: ignore[arg-type]

    result = asyncio.run(graph._grade(state))  # pyright: ignore[reportPrivateUsage]

    rendered = "\n".join(str(message.content) for message in model.messages[0])
    assert cast(EvidenceGrade, result["grade"]).sufficient
    assert "问题：最近的一个录音里关于硅光的前景都讨论了什么" in rendered


def test_empty_chunk_evidence_retries_with_retrieval_topic_not_original_scope_expression() -> None:
    state = cast(
        RagGraphState,
        {
            "query": "最近的一个录音里关于硅光的前景都讨论了什么",
            "retrieval_query": "最近的一个录音里关于硅光的前景都讨论了什么",
            "route": RagRoute(status="resolved", strategy="chunk_search", topic="硅光的前景", recording_limit=1),
            "evidence": [],
        },
    )
    graph = RagGraph(FakeRetriever(), FakeModel())  # type: ignore[arg-type]

    result = asyncio.run(graph._grade(state))  # pyright: ignore[reportPrivateUsage]
    grade = cast(EvidenceGrade, result["grade"])

    assert not grade.sufficient
    assert grade.rewrite_query == "硅光的前景"


def test_scope_evidence_text_exposes_verified_speaker_facts() -> None:
    recording_id = uuid4()
    evidence = Evidence(
        index=1,
        recording=EvidenceRecording(id=recording_id, title="最近录音", file_name="latest.mp3"),
        chunk=EvidenceChunk(
            id=recording_id,
            text="Speaker A: 你好\nSpeaker B: 你好",
            start_ms=0,
            end_ms=2_000,
            speaker_labels=["Speaker A", "Speaker B"],
        ),
        score=1.0,
        match_type="scope",
        facts=EvidenceFacts(scope_verified=True, speaker_count=2, utterance_count=8, transcript_truncated=True),
        url=f"/recordings/{recording_id}",
    )

    rendered = RagGraph._evidence_text([evidence])  # pyright: ignore[reportPrivateUsage]

    assert "录音范围：已由 route 和权限 filters 验证" in rendered
    assert "结构化说话人标签数量：2" in rendered
    assert "结构化说话人标签：Speaker A、Speaker B" in rendered
    assert "发言段总数：8" in rendered
    assert "提供给模型的正文是否截断：是" in rendered

    source = evidence.source_payload()
    chunk = cast(dict[str, object], source["chunk"])
    assert "text" not in chunk
    assert chunk["startMs"] == 0
