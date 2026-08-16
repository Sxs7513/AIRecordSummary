from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import pytest

from l2_core.rag.checkpoint import (
    RagCheckpointSession,
    RagCheckpointStore,
    _serialize_state_without_evidence_text,  # pyright: ignore[reportPrivateUsage]
)
from l2_core.rag.contracts import Evidence, EvidenceChunk, EvidenceRecording, RagGraphState
from l2_core.rag.execution_middleware import (
    RagExecutionCancelled,
    rag_cancellation_scope,
    rag_checkpoint_scope,
    rag_execution_middleware,
    raise_if_rag_cancelled,
)
from l2_core.rag.strategies.base import StrategyResult


def test_rag_cancellation_is_observed_cooperatively_at_boundaries() -> None:
    cancelled = False

    with rag_cancellation_scope(lambda: cancelled):
        raise_if_rag_cancelled()
        cancelled = True
        with pytest.raises(RagExecutionCancelled):
            raise_if_rag_cancelled()

    raise_if_rag_cancelled()


def test_rag_cancellation_middleware_checks_after_node_execution() -> None:
    cancelled = False

    async def node() -> str:
        nonlocal cancelled
        cancelled = True
        return "ignored"

    wrapped = rag_execution_middleware.wrap_node(node, graph_name="test_graph", node_name="node")
    with rag_cancellation_scope(lambda: cancelled), pytest.raises(RagExecutionCancelled):
        asyncio.run(wrapped())


def test_rag_middleware_reuses_checkpoint_without_running_node() -> None:
    generation_id = uuid4()
    new_generation_id = uuid4()
    cached = cast(RagGraphState, {"run_id": "old", "token_usage": 7, "message": "cached"})
    saved: list[tuple[object, str, object]] = []

    class Store:
        def load_all(self, source_generation_id: object, input_hash: str) -> list[tuple[str, str, object]]:
            assert source_generation_id == generation_id
            assert input_hash == "input"
            return [("2026-08-13T10:00:00+00:00", "test_graph-node", cached)]

        def save(self, target_generation_id: object, node: str, _input_hash: str, state: object) -> None:
            saved.append((target_generation_id, node, state))

    called = False

    async def node(_state: RagGraphState) -> dict[str, str]:
        nonlocal called
        called = True
        return {"value": "new"}

    session = RagCheckpointSession(
        store=cast(RagCheckpointStore, Store()),
        generation_id=new_generation_id,
        source_generation_id=generation_id,
        input_hash="input",
        hydrate_state=lambda state: state,
    )
    restored = session.prepare()
    wrapped = rag_execution_middleware.wrap_node(node, graph_name="test_graph", node_name="node")
    with rag_checkpoint_scope(session):
        result = asyncio.run(wrapped(cast(RagGraphState, restored)))

    assert restored is not None
    assert restored["run_id"] == str(new_generation_id)
    assert restored["token_usage"] == 7
    assert restored["message"] == "cached"
    assert result == {}
    assert saved == [(new_generation_id, "test_graph-node", cached)]
    assert not called


def test_checkpoint_node_identity_includes_graph_name() -> None:
    source_generation_id = uuid4()
    generation_id = uuid4()
    cached = cast(RagGraphState, {"run_id": "old", "token_usage": 0, "message": "cached"})
    saved_nodes: list[str] = []

    class Store:
        def load_all(self, _source_generation_id: object, _input_hash: str) -> list[tuple[str, str, object]]:
            return [("2026-08-13T10:00:00+00:00", "first_graph-shared", cached)]

        def save(self, _generation_id: object, node: str, _input_hash: str, _state: object) -> None:
            saved_nodes.append(node)

    calls = 0

    async def shared_node(_state: RagGraphState) -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"message": "executed"}

    session = RagCheckpointSession(
        store=cast(RagCheckpointStore, Store()),
        generation_id=generation_id,
        source_generation_id=source_generation_id,
        input_hash="input",
        hydrate_state=lambda state: state,
    )
    restored = session.prepare()
    first = rag_execution_middleware.wrap_node(shared_node, graph_name="first_graph", node_name="shared")
    second = rag_execution_middleware.wrap_node(shared_node, graph_name="second_graph", node_name="shared")

    assert restored is not None
    with rag_checkpoint_scope(session):
        assert asyncio.run(first(restored)) == {}
        assert asyncio.run(second(restored)) == {"message": "executed"}

    assert calls == 1
    assert saved_nodes == ["first_graph-shared", "second_graph-shared"]


def test_checkpoint_session_reruns_only_selected_completed_nodes() -> None:
    source_generation_id = uuid4()
    generation_id = uuid4()
    cached = cast(RagGraphState, {"run_id": "old", "token_usage": 0, "message": "cached"})

    class Store:
        def load_all(self, _source_generation_id: object, _input_hash: str) -> list[tuple[str, str, object]]:
            return [
                ("2026-08-13T10:00:00+00:00", "graph-retrieve", cached),
                ("2026-08-13T10:01:00+00:00", "graph-finalize", cached),
            ]

        def save(self, _generation_id: object, _node: str, _input_hash: str, _state: object) -> None:
            pass

    calls: list[str] = []

    async def retrieve(_state: RagGraphState) -> dict[str, str]:
        calls.append("retrieve")
        return {"message": "retrieved"}

    async def finalize(_state: RagGraphState) -> dict[str, str]:
        calls.append("finalize")
        return {"message": "finalized"}

    session = RagCheckpointSession(
        store=cast(RagCheckpointStore, Store()),
        generation_id=generation_id,
        source_generation_id=source_generation_id,
        input_hash="input",
        hydrate_state=lambda state: state,
        rerun_nodes={"graph-finalize"},
    )
    restored = session.prepare()
    assert restored is not None

    with rag_checkpoint_scope(session):
        assert asyncio.run(rag_execution_middleware.wrap_node(retrieve, graph_name="graph", node_name="retrieve")(restored)) == {}
        assert asyncio.run(rag_execution_middleware.wrap_node(finalize, graph_name="graph", node_name="finalize")(restored)) == {
            "message": "finalized"
        }

    assert calls == ["finalize"]


def test_checkpoint_repeatable_node_can_run_multiple_agent_iterations() -> None:
    generation_id = uuid4()
    saved_nodes: list[str] = []

    class Store:
        def load_all(self, _source_generation_id: object, _input_hash: str) -> list[tuple[str, str, object]]:
            return []

        def save(self, _generation_id: object, node: str, _input_hash: str, _state: object) -> None:
            saved_nodes.append(node)

    calls = 0

    async def inspect(_state: RagGraphState) -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"token_usage": 1}

    session = RagCheckpointSession(
        store=cast(RagCheckpointStore, Store()),
        generation_id=generation_id,
        source_generation_id=None,
        input_hash="input",
        hydrate_state=lambda state: state,
        repeatable_nodes={"graph-inspect"},
    )
    wrapped = rag_execution_middleware.wrap_node(inspect, graph_name="graph", node_name="inspect")
    state = cast(RagGraphState, {"token_usage": 0})

    with rag_checkpoint_scope(session):
        assert asyncio.run(wrapped(state)) == {"token_usage": 1}
        assert asyncio.run(wrapped(state)) == {"token_usage": 1}

    assert calls == 2
    assert saved_nodes == ["graph-inspect", "graph-inspect"]


def test_checkpoint_state_omits_candidate_and_evidence_text() -> None:
    recording_id = uuid4()
    evidence = Evidence(
        index=1,
        recording=EvidenceRecording(id=recording_id, title="录音", file_name="recording.m4a"),
        chunk=EvidenceChunk(id=uuid4(), text="敏感正文", start_ms=10, end_ms=20),
        score=1,
        match_type="vector",
        url=f"/recordings/{recording_id}",
    )
    state = cast(
        RagGraphState,
        {
            "retrieval_candidates": [
                {
                    "chunk_id": str(evidence.chunk.id),
                    "text": "候选正文",
                    "created_at": datetime(2026, 8, 13, 10, 30, tzinfo=UTC),
                }
            ],
            "evidence": [evidence],
            "answer_evidence": [evidence],
            "strategy_result": StrategyResult(
                status="ready",
                answer_context="派生正文",
                evidence=[evidence],
            ),
        },
    )

    serialized = _serialize_state_without_evidence_text(state)

    payload = str(serialized)
    assert "敏感正文" not in payload
    assert "候选正文" not in payload
    assert "派生正文" not in payload
    assert serialized["retrieval_candidates"][0]["created_at"] == "2026-08-13T10:30:00+00:00"
    json.dumps(serialized)
