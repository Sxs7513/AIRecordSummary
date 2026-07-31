from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Literal, cast
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from l1_foundation.llm import LlmGenerateInput, LlmGenerateResult, LlmProvider
from l1_foundation.observability import (
    InstrumentedModelClient,
    ModelInvocationRecord,
    ObservabilityClient,
    ObservabilityScope,
    RagExecutionSpanRecord,
    observation_scope,
)
from l1_foundation.worker import ComputeCommand
from l2_core.rag.contracts import (
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
from l2_core.rag.graph import RagGraph
from l2_core.rag.hooks import RagNodeCompleted, RagOperationCompleted
from l2_core.rag.retrieval import RagRetriever
from l2_core.rag.routing import AMBIGUOUS_RECORDING_SCOPE_MESSAGE, ROUTE_UNRESOLVED_MESSAGE
from l2_core.rag.streaming import ThinkTagFilter
from l2_core.rag.worker_tasks import RerankResult
from l2_core.rag.workflows.chunk_evidence import _retain_protected_evidence  # pyright: ignore[reportPrivateUsage]


class FakeModel:
    def __init__(self) -> None:
        self.json_schemas: list[Mapping[str, object] | None] = []
        self.model_profiles: list[str] = []
        self.providers: list[LlmProvider] = []
        self._responses = iter(
            [
                '{"status":"resolved","strategy":"chunk_search","inferred_filters":{}}',
                '{"verdict":"direct_answer","reason":"enough"}',
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
        yield "交付风险是供应延期[1]。"

    def stream(self, messages: Sequence[BaseMessage], max_tokens: int, temperature: float = 0.1) -> AsyncIterator[str]:
        return self._stream()

    def set_responses(self, responses: Sequence[str]) -> None:
        self._responses = iter(responses)


class FakeWorkerClient:
    def __init__(self, model: FakeModel) -> None:
        self._model = model

    async def execute(self, command: ComputeCommand[LlmGenerateInput], *, result_type: type[LlmGenerateResult]) -> LlmGenerateResult:
        value = command.input
        model_profiles = cast(list[str] | None, getattr(self._model, "model_profiles", None))
        if model_profiles is None:
            model_profiles = []
            self._model.model_profiles = model_profiles
        model_profiles.append(value.model_profile)
        providers = cast(list[LlmProvider] | None, getattr(self._model, "providers", None))
        if providers is None:
            providers = []
            self._model.providers = providers
        providers.append(value.provider)
        messages = [_base_message(message.role.value, message.content) for message in value.messages]
        options = value.options
        text = await self._model.complete(
            messages,
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            json_schema=options.response_format.json_schema,
        )
        return result_type.model_validate({"text": text, "provider": "gemini", "model": "gemini-test"})

    async def execute_streaming(
        self,
        command: ComputeCommand[LlmGenerateInput],
        *,
        result_type: type[LlmGenerateResult],
        on_delta: Callable[[str], None] | None = None,
    ) -> LlmGenerateResult:
        value = command.input
        providers = cast(list[LlmProvider] | None, getattr(self._model, "providers", None))
        if providers is None:
            providers = []
            self._model.providers = providers
        providers.append(value.provider)
        messages = [_base_message(message.role.value, message.content) for message in value.messages]
        options = value.options
        chunks: list[str] = []
        async for text in self._model.stream(messages, options.max_tokens, options.temperature):
            chunks.append(text)
            if on_delta is not None:
                on_delta(text)
        return result_type(text="".join(chunks), provider=LlmProvider.GEMINI, model="gemini-test")


def _base_message(role: str, content: str) -> BaseMessage:
    if role == "assistant":
        return AIMessage(content)
    if role == "system":
        return SystemMessage(content)
    return HumanMessage(content)


def _graph(
    retriever: object,
    model: FakeModel,
    *,
    plan_local_input_tokens: int = 4_000,
    route_model_profile: Literal["default", "rag"] = "default",
    node_model_profile: Literal["default", "rag"] = "default",
    query_term_expansion_enabled: bool = False,
) -> RagGraph:
    return RagGraph(
        cast(RagRetriever, retriever),
        cast(InstrumentedModelClient, FakeWorkerClient(model)),
        online_provider=LlmProvider.GEMINI,
        context_size=16_384,
        plan_local_input_tokens=plan_local_input_tokens,
        route_model_profile=route_model_profile,
        node_model_profile=node_model_profile,
        query_term_expansion_enabled=query_term_expansion_enabled,
    )


class FakeRetriever:
    hybrid_search_enabled = False
    rerank_enabled = False

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

    def retrieve_candidates(self, topic: str, filters: object, limit: int) -> list[dict[str, object]]:
        return [
            {"chunk_id": item.chunk.id, "recording_id": item.recording.id, "score": item.score, "evidence": item}
            for item in self.retrieve_chunks(topic, filters, limit)
        ]

    def expand_candidates(self, rows: list[dict[str, object]]) -> list[Evidence]:
        return [cast(Evidence, row["evidence"]) for row in rows]

    def retrieve_scope(self, *args: object) -> list[Evidence]:
        raise AssertionError("topic route must use chunk retrieval")


class MetadataRetriever(FakeRetriever):
    def __init__(self, recording_id: object) -> None:
        self.recording_id = recording_id
        self.chunk_search_called = False

    def retrieve_candidates(self, topic: str, filters: object, limit: int) -> list[dict[str, object]]:
        del topic, filters, limit
        self.chunk_search_called = True
        raise AssertionError("metadata strategy must not use chunk retrieval")

    def retrieve_metadata(self, *args: object) -> list[dict[str, object]]:
        del args
        return [
            {
                "id": self.recording_id,
                "title": "产品周会",
                "file_name": "weekly.mp3",
                "location": "",
                "duration_seconds": 125,
                "created_at": datetime(2026, 8, 8, 9, 30, tzinfo=UTC),
                "speakers": [
                    {"name": "张三", "speaking_duration_seconds": 90.5},
                    {"name": "李四", "speaking_duration_seconds": 34.2},
                ],
            }
        ]


def test_query_term_expansion_keeps_the_original_question_and_only_adds_anchors() -> None:
    model = FakeModel()
    model.set_responses(
        ['{"content_query":"王总说 API v2 的上线时间定了吗？","terms":["王总","API v2"],"phrases":["上线时间"]}']
    )
    graph = _graph(FakeRetriever(), model, query_term_expansion_enabled=True)
    state = RagGraph._initial_state(  # pyright: ignore[reportPrivateUsage]
        "test", "answer", "最近的录音里，王总说 API v2 的上线时间定了吗？", 10, [], None
    )
    state["route"] = RagRoute(status="resolved", strategy_id="fact_lookup")
    state["content_query"] = "王总说 API v2 的上线时间定了吗？"

    update = asyncio.run(graph._expand_retrieval_terms(state))  # pyright: ignore[reportPrivateUsage]

    assert state["query"] == "最近的录音里，王总说 API v2 的上线时间定了吗？"
    assert update["content_query"] == "王总说 API v2 的上线时间定了吗？"
    assert update["retrieval_expanded_query"] == "上线时间 王总 API v2"
    assert update["retrieval_lexical_queries"] == ["上线时间", "王总", "API v2"]


def test_route_removes_recording_time_scope_from_content_query() -> None:
    model = FakeModel()
    model.set_responses(
        [
            '{"status":"resolved","strategy_id":"fact_lookup",'
            '"content_query":"是否讨论了预算审批",'
            '"time_range":{"text":"本月","start":"2026-08-01T00:00:00+08:00",'
            '"end":"2026-08-13T00:00:00+08:00"},"inferred_filters":{}}'
        ]
    )
    graph = _graph(FakeRetriever(), model)
    state = RagGraph._initial_state(  # pyright: ignore[reportPrivateUsage]
        "test", "answer", "本月的录音中是否讨论了预算审批", 10, [], None
    )

    update = asyncio.run(graph._route(state))  # pyright: ignore[reportPrivateUsage]

    assert update["route_error"] is None
    assert update["content_query"] == "是否讨论了预算审批"
    route = cast(RagRoute, update["route"])
    assert route.time_range is not None
    assert route.time_range.text == "本月"


def test_extracted_terms_are_each_sent_to_lexical_retrieval() -> None:
    recording_id = uuid4()
    lexical_queries: list[str] = []

    class CapturingHybridRetriever:
        hybrid_search_enabled = True

        def generate_query_embedding(self, _query: str) -> list[float]:
            return [0.1]

        def retrieve_vector_candidates(
            self, _embedding: list[float], _filters: ResolvedFilters
        ) -> list[dict[str, object]]:
            return []

        def retrieve_lexical_candidates(
            self, query: str, _filters: ResolvedFilters
        ) -> list[dict[str, object]]:
            lexical_queries.append(query)
            return []

        def fuse_candidate_lists(
            self,
            _vector_lists: list[list[dict[str, object]]],
            _lexical_lists: list[list[dict[str, object]]],
            _limit: int,
        ) -> list[dict[str, object]]:
            return []

    graph = _graph(CapturingHybridRetriever(), FakeModel())
    asyncio.run(
        graph._retrieve_candidates(  # pyright: ignore[reportPrivateUsage]
            "最近是不是有个项目答辩的路演",
            ResolvedFilters(recording_scope_resolved=True, recording_ids=[recording_id]),
            10,
            str(uuid4()),
            lexical_queries=["项目答辩", "路演"],
        )
    )

    assert set(lexical_queries) == {"最近是不是有个项目答辩的路演", "项目答辩", "路演"}


def test_exact_lexical_term_hit_is_retained_when_rrf_drops_it() -> None:
    recording_id = uuid4()
    protected_chunk_id = uuid4()

    class ExactMatchRetriever:
        hybrid_search_enabled = True

        def generate_query_embedding(self, _query: str) -> list[float]:
            return [0.1]

        def retrieve_vector_candidates(
            self, _embedding: list[float], _filters: ResolvedFilters
        ) -> list[dict[str, object]]:
            return []

        def retrieve_lexical_candidates(
            self, query: str, _filters: ResolvedFilters
        ) -> list[dict[str, object]]:
            if query != "路演":
                return []
            return [
                {
                    "chunk_id": protected_chunk_id,
                    "recording_id": recording_id,
                    "score": 1.0,
                    "exact_match": True,
                }
            ]

        def fuse_candidate_lists(
            self,
            _vector_lists: list[list[dict[str, object]]],
            _lexical_lists: list[list[dict[str, object]]],
            _limit: int,
        ) -> list[dict[str, object]]:
            return []

    graph = _graph(ExactMatchRetriever(), FakeModel())
    candidates = asyncio.run(
        graph._retrieve_candidates(  # pyright: ignore[reportPrivateUsage]
            "项目答辩路演",
            ResolvedFilters(recording_scope_resolved=True, recording_ids=[recording_id]),
            10,
            str(uuid4()),
            lexical_queries=["路演"],
            protected_lexical_queries=["路演"],
        )
    )

    assert candidates[0]["chunk_id"] == protected_chunk_id
    assert candidates[0]["protected_lexical_terms"] == ["路演"]


def test_rerank_preserves_protected_exact_lexical_evidence() -> None:
    recording_id = uuid4()
    protected = Evidence(
        index=1,
        recording=EvidenceRecording(id=recording_id, title="路演", file_name="roadshow.mp3"),
        chunk=EvidenceChunk(id=uuid4(), text="项目答辩路演", start_ms=0, end_ms=1_000),
        score=1.0,
        match_type="lexical",
        url=f"/recordings/{recording_id}",
    )
    reranked = Evidence(
        index=2,
        recording=EvidenceRecording(id=recording_id, title="其他", file_name="other.mp3"),
        chunk=EvidenceChunk(id=uuid4(), text="其他候选", start_ms=1_000, end_ms=2_000),
        score=0.9,
        match_type="vector",
        url=f"/recordings/{recording_id}",
    )

    retained = _retain_protected_evidence([reranked], [protected, reranked], [str(protected.chunk.id)])

    assert retained == [protected, reranked]


class JsonEventHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, object]] = []

    def emit(self, record: logging.LogRecord) -> None:
        value = json.loads(record.getMessage())
        if isinstance(value, dict):
            self.events.append(cast(dict[str, object], value))


class CapturingObservabilityClient:
    def __init__(self) -> None:
        self.model_invocations: list[ModelInvocationRecord] = []
        self.spans: list[RagExecutionSpanRecord] = []

    def publish_model_invocation(self, record: ModelInvocationRecord) -> None:
        self.model_invocations.append(record)

    def publish_span(self, record: RagExecutionSpanRecord) -> None:
        self.spans.append(record)


class CapturingRagHook:
    def __init__(self) -> None:
        self.nodes: list[RagNodeCompleted] = []
        self.operations: list[RagOperationCompleted] = []

    def on_node_completed(self, event: RagNodeCompleted) -> None:
        self.nodes.append(event)

    def on_operation_completed(self, event: RagOperationCompleted) -> None:
        self.operations.append(event)


def test_langgraph_routes_retrieves_validates_and_streams_only_final_answer() -> None:
    model = FakeModel()
    graph = _graph(FakeRetriever(), model)
    deltas: list[str] = []
    phases: list[str] = []

    answer, sources, not_enough_evidence, message = asyncio.run(
        graph.run("交付风险是什么", 10, [uuid4()], lambda name, _label, _progress: phases.append(name), deltas.append)
    )

    assert answer == "交付风险是供应延期[1]。"
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
    assert model.model_profiles and set(model.model_profiles) == {"default"}


def test_metadata_strategy_skips_chunk_retrieval_and_returns_recording_source() -> None:
    recording_id = uuid4()
    retriever = MetadataRetriever(recording_id)
    model = FakeModel()
    model.set_responses(['{"status":"resolved","strategy_id":"metadata_lookup","inferred_filters":{}}'])

    answer, sources, not_enough, message = asyncio.run(
        _graph(retriever, model).run(
            "这条录音多长？",
            10,
            [recording_id],
            lambda _name, _label, _progress: None,
            lambda _delta: None,
        )
    )

    assert answer == "交付风险是供应延期[1]。"
    assert not not_enough
    assert message is None
    assert not retriever.chunk_search_called
    recording = cast(dict[str, object], sources[0]["recording"])
    assert recording["id"] == str(recording_id)
    assert recording["durationSeconds"] == 125
    assert recording["location"] is None
    chunk = cast(dict[str, object], sources[0]["chunk"])
    assert chunk["speakerLabels"] == ["张三", "李四"]
    assert len(model.json_schemas) == 1
    assert model.providers == [LlmProvider.GEMINI, LlmProvider.LOCAL]


def test_retrieval_run_uses_production_nodes_and_stops_before_grade() -> None:
    model = FakeModel()
    hook = CapturingRagHook()

    state = asyncio.run(
        _graph(FakeRetriever(), model).run_retrieval(
            "交付风险是什么",
            10,
            [uuid4()],
            hook=hook,
        )
    )

    assert state["evidence"]
    assert [event.node for event in hook.nodes] == ["route", "retrieve", "expand_context"]
    assert [event.operation for event in hook.operations] == ["retrieve.vector", "retrieve.expand"]
    assert len(model.json_schemas) == 1
    assert model.json_schemas[0] == RagRoute.model_json_schema()


def test_retrieved_evidence_can_be_graded_without_running_plan_or_answer() -> None:
    model = FakeModel()
    model.set_responses(
        [
            '{"status":"resolved","strategy":"chunk_search","inferred_filters":{}}',
            '{"verdict":"direct_answer","reason":"enough"}',
        ]
    )
    hook = CapturingRagHook()
    graph = _graph(FakeRetriever(), model)
    state = asyncio.run(
        graph.run_retrieval(
            "交付风险是什么",
            10,
            [uuid4()],
            hook=hook,
        )
    )

    graded = asyncio.run(graph.grade_retrieval(state, hook=hook))

    assert graded["grade"] is not None
    assert graded["grade"].verdict == "direct_answer"
    assert [event.node for event in hook.nodes][-1] == "grade"
    assert "plan" not in [event.node for event in hook.nodes]
    assert model.json_schemas[-1] == EvidenceGrade.model_json_schema()
    assert model.providers[-1] == LlmProvider.LOCAL


def test_retrieval_run_hook_captures_independent_rerank_node() -> None:
    class RerankingRetriever(FakeRetriever):
        rerank_enabled = True

        def rerank_evidence(
            self, query: str, evidence: list[Evidence]
        ) -> tuple[list[Evidence], RerankResult]:
            return evidence, RerankResult(
                model_name="Qwen/Qwen3-Reranker-0.6B",
                scores=[],
                input_tokens=42,
                skipped_candidates=0,
            )

    hook = CapturingRagHook()
    asyncio.run(
        _graph(RerankingRetriever(), FakeModel()).run_retrieval(
            "交付风险是什么",
            10,
            [uuid4()],
            hook=hook,
        )
    )

    assert [event.node for event in hook.nodes] == ["route", "retrieve", "expand_context", "rerank"]
    assert [event.operation for event in hook.operations] == [
        "retrieve.vector",
        "retrieve.expand",
        "retrieve.rerank",
    ]
    assert hook.operations[-1].details["input_tokens"] == 42


def test_route_uses_the_online_answer_provider() -> None:
    model = FakeModel()
    graph = _graph(FakeRetriever(), model, route_model_profile="rag")
    state = cast(
        RagGraphState,
        {
            "run_id": "route-profile-test",
            "query": "公司目前营收多少",
            "content_query": "公司目前营收多少",
            "history": [],
            "scope_recording_ids": [],
            "retrieval_attempt": 0,
            "token_usage": 0,
        },
    )

    asyncio.run(graph._route(state))  # pyright: ignore[reportPrivateUsage]

    assert model.model_profiles == ["default"]
    assert model.providers == [LlmProvider.GEMINI]


def test_local_non_route_rag_nodes_can_switch_back_to_rag_4b() -> None:
    model = FakeModel()
    graph = _graph(FakeRetriever(), model, node_model_profile="rag")

    asyncio.run(
        graph.run(
            "交付风险是什么",
            10,
            [uuid4()],
            lambda _name, _label, _progress: None,
            lambda _delta: None,
        )
    )

    assert model.model_profiles[:1] == ["default"]  # route
    assert model.model_profiles[1:] and set(model.model_profiles[1:]) == {"rag"}


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
                    '{"status":"resolved","strategy":"chunk_search","inferred_filters":{}}',
                    '{"verdict":"direct_answer","reason":"enough"}',
                    '{"items":[{"statement":"仅回答第二条","evidence_indexes":[2]}]}',
                ]
            )

        async def _stream(self) -> AsyncIterator[str]:
            yield "交付风险是供应延期[2]。"

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
        _graph(TwoEvidenceRetriever(), model).run(
            "比较风险",
            10,
            [first_recording_id, second_recording_id],
            lambda _name, _label, _progress: None,
            lambda _delta: None,
        )
    )

    answer_prompt_text = "\n".join(str(item.content) for item in model.answer_messages)
    assert answer == "交付风险是供应延期[1]。"
    assert not not_enough_evidence
    assert message is None
    assert len(sources) == 1
    assert sources[0]["index"] == 1
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
            _graph(FakeRetriever(), FakeModel()).run(
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
        "expand_context",
        "grade",
        "plan",
        "validate_plan",
        "select_planned_evidence",
        "answer",
    ]
    assert node_completions == node_starts
    completions_by_node = {
        cast(str, event["node"]): event
        for event in handler.events
        if event["event"] == "node_completed"
    }
    assert {node: event["model_execution"] for node, event in completions_by_node.items()} == {
        "route": "online",
        "retrieve": "local",
        "expand_context": "none",
        "grade": "local",
        "plan": "local",
        "validate_plan": "none",
        "select_planned_evidence": "none",
        "answer": "online",
    }
    assert transitions == [
        ("route", "fact_lookup"),
        ("retrieve", "expand_context"),
        ("expand_context", "grade"),
        ("grade", "plan"),
    ]
    assert all(event["run_id"] == str(run_id) for event in handler.events)
    route_completed = next(
        event for event in handler.events if event["event"] == "node_completed" and event["node"] == "route"
    )
    assert route_completed["query"] == "比较交付风险"
    assert '"strategy":"chunk_search"' in cast(str, route_completed["raw"])
    assert cast(dict[str, object], route_completed["resolved_filters"])["recording_scope_resolved"] is True
    grade_completed = next(
        event for event in handler.events if event["event"] == "node_completed" and event["node"] == "grade"
    )
    plan_completed = next(
        event for event in handler.events if event["event"] == "node_completed" and event["node"] == "plan"
    )
    assert grade_completed["model_execution"] == "local"
    assert grade_completed["provider"] == "local"
    evidence_refs = cast(list[dict[str, str]], grade_completed["evidence_refs"])
    assert len(evidence_refs) == 1
    assert set(evidence_refs[0]) == {"recording_id", "chunk_id"}
    assert plan_completed["model_execution"] == "local"
    assert plan_completed["provider"] == "local"
    answer_mode = next(event for event in handler.events if event["event"] == "answer_mode_selected")
    answer_evidence = cast(list[dict[str, object]], answer_mode["answer_evidence_text"])
    assert answer_evidence[0]["index"] == 1
    assert "text" not in answer_evidence[0]
    assert {"recording_id", "chunk_id", "start_ms", "end_ms"} <= set(answer_evidence[0])
    assert any(event["event"] == "answer_first_token" for event in handler.events)
    assert any(event["event"] == "graph_completed" and event["status"] == "succeeded" for event in handler.events)


def test_enabled_rerank_executes_as_an_independent_langgraph_node() -> None:
    class RerankingRetriever(FakeRetriever):
        rerank_enabled = True

        def rerank_evidence(
            self, query: str, evidence: list[Evidence]
        ) -> tuple[list[Evidence], RerankResult]:
            assert query == "交付风险是什么"
            return evidence, RerankResult(
                model_name="Qwen/Qwen3-Reranker-0.6B",
                scores=[],
                input_tokens=42,
                skipped_candidates=0,
            )

    handler = JsonEventHandler()
    telemetry = CapturingObservabilityClient()
    rag_logger = logging.getLogger("rag")
    previous_level = rag_logger.level
    rag_logger.setLevel(logging.INFO)
    rag_logger.addHandler(handler)
    try:
        async def scenario() -> None:
            with observation_scope(
                cast(ObservabilityClient, telemetry),
                ObservabilityScope(workspace_id=uuid4(), generation_run_id=uuid4()),
            ):
                await _graph(RerankingRetriever(), FakeModel()).run(
                    "交付风险是什么",
                    10,
                    [uuid4()],
                    lambda _name, _label, _progress: None,
                    lambda _delta: None,
                )

        asyncio.run(scenario())
    finally:
        rag_logger.removeHandler(handler)
        rag_logger.setLevel(previous_level)

    node_starts = [event["node"] for event in handler.events if event["event"] == "node_started"]
    assert node_starts[:5] == ["route", "retrieve", "expand_context", "rerank", "grade"]
    rerank_completed = next(
        event for event in handler.events if event["event"] == "node_completed" and event["node"] == "rerank"
    )
    assert rerank_completed["input_tokens"] == 42
    reranked_evidence = cast(list[dict[str, object]], rerank_completed["reranked_evidence_text"])
    assert "text" not in reranked_evidence[0]
    assert {"recording_id", "chunk_id", "start_ms", "end_ms"} <= set(reranked_evidence[0])
    assert any(
        event["event"] == "graph_transition" and event["source"] == "rerank" and event["target"] == "grade"
        for event in handler.events
    )
    assert [record.status for record in telemetry.model_invocations] == ["running", "succeeded"]
    terminal_invocation = telemetry.model_invocations[-1]
    assert terminal_invocation.operation == "rerank"
    assert terminal_invocation.usage_kind == "rerank"
    assert terminal_invocation.provider == "local"
    assert terminal_invocation.model == "Qwen/Qwen3-Reranker-0.6B"
    assert terminal_invocation.prompt_tokens == 42
    assert terminal_invocation.completion_tokens == 0
    assert terminal_invocation.usage_source == "local_tokenizer"
    rerank_span = next(
        record
        for record in telemetry.spans
        if record.operation == "rerank" and record.status == "succeeded"
    )
    assert rerank_span.metadata["model_execution"] == "local"


def test_grade_stays_local_while_plan_uses_online_provider_when_its_threshold_is_exceeded() -> None:
    handler = JsonEventHandler()
    rag_logger = logging.getLogger("rag")
    previous_level = rag_logger.level
    rag_logger.setLevel(logging.INFO)
    rag_logger.addHandler(handler)
    try:
        asyncio.run(
            _graph(FakeRetriever(), FakeModel(), plan_local_input_tokens=1).run(
                "比较交付风险",
                10,
                [uuid4()],
                lambda _name, _label, _progress: None,
                lambda _delta: None,
            )
        )
    finally:
        rag_logger.removeHandler(handler)
        rag_logger.setLevel(previous_level)

    completions = {
        cast(str, event["node"]): event
        for event in handler.events
        if event["event"] == "node_completed" and event["node"] in {"grade", "plan"}
    }
    assert completions["grade"]["model_execution"] == "local"
    assert completions["grade"]["provider"] == "local"
    assert completions["plan"]["model_execution"] == "online"
    assert completions["plan"]["provider"] == "gemini"


def test_simple_question_also_enters_answer_plan() -> None:
    class DirectAnswerModel(FakeModel):
        def __init__(self) -> None:
            self.json_schemas: list[Mapping[str, object] | None] = []
            self._responses = iter(
                [
                    '{"status":"resolved","strategy":"chunk_search","inferred_filters":{}}',
                    '{"verdict":"direct_answer","reason":"enough"}',
                    '{"items":[{"statement":"回答发布日期","evidence_indexes":[1]}]}',
                ]
            )

    model = DirectAnswerModel()
    graph = _graph(FakeRetriever(), model)

    answer, sources, not_enough_evidence, message = asyncio.run(
        graph.run("发布日期是什么", 10, [uuid4()], lambda _name, _label, _progress: None, lambda _delta: None)
    )

    assert answer == "交付风险是供应延期[1]。"
    assert len(sources) == 1
    assert not not_enough_evidence
    assert message is None
    assert len(model.json_schemas) == 3


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

    graph = _graph(DegradedHybridRetriever(), FakeModel())

    candidates = asyncio.run(
        graph._retrieve_candidates(  # pyright: ignore[reportPrivateUsage]
            "API v2",
            ResolvedFilters(recording_scope_resolved=True, recording_ids=[recording_id]),
            10,
            str(uuid4()),
        )
    )
    update = asyncio.run(
        graph._expand_context(  # pyright: ignore[reportPrivateUsage]
            cast(
                RagGraphState,
                {
                    "run_id": "standalone",
                    "retrieval_attempt": 0,
                    "retrieval_candidates": candidates,
                },
            )
        )
    )
    evidence = cast(list[Evidence], update["evidence"])

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

    graph = _graph(FakeRetriever(), UnresolvedRouteModel())
    phases: list[str] = []

    answer, sources, not_enough_evidence, message = asyncio.run(
        graph.run("嗯", 10, [], lambda name, _label, _progress: phases.append(name), lambda _delta: None)
    )

    assert answer == ROUTE_UNRESOLVED_MESSAGE
    assert sources == []
    assert not_enough_evidence
    assert message == "route_unresolved"
    assert phases == ["routing"]


def test_route_does_not_validate_query_topic() -> None:
    class RepetitiveTopicModel(FakeModel):
        def __init__(self) -> None:
            self._responses = iter(['{"status":"resolved","strategy":"chunk_search","inferred_filters":{}}'])

    graph = _graph(FakeRetriever(), RepetitiveTopicModel())
    state = cast(
        RagGraphState,
        {
            "run_id": "test",
            "query": "哈哈哈",
            "content_query": "哈哈哈",
            "history": [],
            "scope_recording_ids": [],
            "retrieval_attempt": 0,
            "token_usage": 0,
        },
    )

    result = asyncio.run(graph._route(state))  # pyright: ignore[reportPrivateUsage]

    assert result["route_error"] is None
    assert cast(RagRoute, result["route"]).strategy == "chunk_search"


def test_vague_scope_question_stays_unresolved_because_the_query_object_is_missing() -> None:
    class VagueScopeModel(FakeModel):
        def __init__(self) -> None:
            self._responses = iter(
                [
                    '{"status":"unresolved","strategy":null,'
                    '"inferred_filters":{},"error_code":"unresolved_query"}'
                ]
            )

    graph = _graph(FakeRetriever(), VagueScopeModel())
    answer, sources, not_enough_evidence, message = asyncio.run(graph.run("讲了什么", 10, [], lambda _name, _label, _progress: None, lambda _delta: None))

    assert answer == ROUTE_UNRESOLVED_MESSAGE
    assert sources == []
    assert not_enough_evidence
    assert message == "unresolved_query"


def test_route_receives_turn_bound_history_context_while_answer_history_excludes_sources() -> None:
    recording_id = uuid4()
    history = [
        RagHistoryMessage(role="user", content="前一条录音是什么？"),
        RagHistoryMessage(
            role="assistant",
            content="是项目周会。",
            sources=[RagHistorySource(recording_id=recording_id)],
        ),
    ]

    context = RagGraph._route_history_context(history)  # pyright: ignore[reportPrivateUsage]
    assert str(recording_id) in context
    assert "是项目周会。" in context
    assert '"sources":[{"recording_id"' in context
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

    graph = _graph(FakeRetriever(), AmbiguousRouteModel())
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
        strategy_id="scope_summary",
        inferred_filters=InferredFilters(recording_ids=[recording_id]),
    )
    assert RagGraph._validate_selected_recording_ids(route, [recording_id]) is None  # pyright: ignore[reportPrivateUsage]
    assert RagGraph._validate_selected_recording_ids(route, []) == "referenced_recording_unavailable"  # pyright: ignore[reportPrivateUsage]


def test_grade_contract_rejects_legacy_fields() -> None:
    with pytest.raises(ValueError):
        EvidenceGrade.model_validate({"sufficient": True})
    with pytest.raises(ValueError):
        EvidenceGrade.model_validate({"verdict": "direct_answer", "decision": "answer"})

    assert "verdict" in EvidenceGrade.model_json_schema()["properties"]
    assert "sufficient" not in EvidenceGrade.model_json_schema()["properties"]


def test_qualified_answer_grade_enters_plan() -> None:
    grade = EvidenceGrade(verdict="qualified_answer")
    state = cast(
        RagGraphState,
        {
            "grade": grade,
            "route": RagRoute(status="resolved", strategy_id="fact_lookup"),
            "retrieval_attempt": 0,
        },
    )

    assert RagGraph._after_grade(state) == "plan"  # pyright: ignore[reportPrivateUsage]


def test_abstain_grade_does_not_trigger_a_pointless_retrieval_retry() -> None:
    state = cast(
        RagGraphState,
        {
            "grade": EvidenceGrade(
                verdict="abstain",
                reason="cannot_infer_internal_intent",
            ),
            "route": RagRoute(status="resolved", strategy_id="fact_lookup"),
            "retrieval_attempt": 0,
        },
    )

    assert RagGraph._after_grade(state) == "done"  # pyright: ignore[reportPrivateUsage]


def test_direct_answer_grade_enters_plan() -> None:
    state = cast(
        RagGraphState,
        {
            "route": RagRoute(status="resolved", strategy_id="fact_lookup"),
            "grade": EvidenceGrade(verdict="direct_answer"),
            "retrieval_attempt": 0,
        },
    )

    assert RagGraph._after_grade(state) == "plan"  # pyright: ignore[reportPrivateUsage]


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
            "route": RagRoute(status="resolved", strategy_id="scope_summary", recording_limit=2),
            "grade": EvidenceGrade(verdict="direct_answer"),
            "evidence": evidence,
        },
    )

    update = asyncio.run(RagGraph._decide_plan(state))  # pyright: ignore[reportPrivateUsage]
    decided_state = cast(RagGraphState, {**state, **update})

    assert update["planning_required"] is True
    assert RagGraph._after_plan_decision(decided_state) == "plan"  # pyright: ignore[reportPrivateUsage]


def test_single_scope_recording_skips_plan() -> None:
    recording_id = uuid4()
    state = cast(
        RagGraphState,
        {
            "route": RagRoute(status="resolved", strategy_id="scope_summary"),
            "grade": EvidenceGrade(verdict="direct_answer"),
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


def test_grade_evidence_includes_recording_context_and_omits_empty_location() -> None:
    occurred_at = datetime(2026, 8, 8, 9, 30, tzinfo=UTC)
    first = Evidence(
        index=1,
        recording=EvidenceRecording(
            id=uuid4(),
            title="项目周会",
            file_name="week.mp3",
            location="上海会议室",
            created_at=occurred_at,
        ),
        chunk=EvidenceChunk(
            id=uuid4(),
            text="第一条证据。",
            start_ms=0,
            end_ms=1_000,
            topic="路演答辩策略",
            terms=["路演", "评委问答"],
            search_context="讨论汇报顺序和答辩准备。",
        ),
        score=0.9,
        match_type="vector",
        url="/recordings/first",
    )
    second = Evidence(
        index=2,
        recording=EvidenceRecording(id=uuid4(), title="线上沟通", file_name="online.mp3", created_at=occurred_at),
        chunk=EvidenceChunk(id=uuid4(), text="第二条证据。", start_ms=0, end_ms=1_000),
        score=0.8,
        match_type="lexical",
        url="/recordings/second",
    )

    rendered = RagGraph._grade_evidence_text([first, second])  # pyright: ignore[reportPrivateUsage]

    assert "证据 1：" in rendered
    assert "录音名字：项目周会" in rendered
    assert "录音发生时间：2026-08-08T09:30:00+00:00" in rendered
    assert "录音发生地点：上海会议室" in rendered
    assert "主题：路演答辩策略" in rendered
    assert "标准术语：路演、评委问答" in rendered
    assert "语义上下文：讨论汇报顺序和答辩准备。" in rendered
    assert "正文：第一条证据。" in rendered
    assert "录音正文：\n主题：路演答辩策略" in rendered
    assert "录音名字：线上沟通" in rendered
    assert "录音正文：\n第二条证据。" in rendered
    assert rendered.count("录音发生地点：") == 1


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


def test_chunk_search_grades_the_scope_free_content_query() -> None:
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
            return '{"verdict":"direct_answer","reason":"enough"}'

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
            "content_query": "关于硅光的前景都讨论了什么",
            "route": RagRoute(
                status="resolved",
                strategy_id="fact_lookup",
                recording_limit=1,
            ),
            "evidence": [evidence],
        },
    )
    model = CapturingGradeModel()
    graph = _graph(FakeRetriever(), model)

    result = asyncio.run(graph._grade(state))  # pyright: ignore[reportPrivateUsage]

    rendered = "\n".join(str(message.content) for message in model.messages[0])
    assert cast(EvidenceGrade, result["grade"]).verdict == "direct_answer"
    assert "问题：关于硅光的前景都讨论了什么" in rendered
    assert "问题：最近的一个录音里" not in rendered
    assert "硅光具有较好的发展前景。" in rendered
    assert "recording_id" not in rendered
    assert "录音名字：最近录音" in rendered


def test_empty_chunk_evidence_abstains() -> None:
    state = cast(
        RagGraphState,
        {
            "query": "最近的一个录音里关于硅光的前景都讨论了什么",
            "content_query": "关于硅光的前景都讨论了什么",
            "route": RagRoute(
                status="resolved",
                strategy_id="fact_lookup",
                recording_limit=1,
            ),
            "evidence": [],
        },
    )
    graph = _graph(FakeRetriever(), FakeModel())

    result = asyncio.run(graph._grade(state))  # pyright: ignore[reportPrivateUsage]
    grade = cast(EvidenceGrade, result["grade"])

    assert grade.verdict == "abstain"


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
