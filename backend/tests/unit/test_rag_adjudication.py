from __future__ import annotations

import asyncio
import json
from typing import Any, cast
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import ValidationError

from l1_foundation.llm import LlmGenerateResult, LlmProvider
from l2_core.rag.adjudication.agent import AdjudicationCaseContext, EvidenceAdjudicationAgent
from l2_core.rag.adjudication.contracts import (
    AdjudicationAgentState,
    AdjudicationProposal,
    AdjudicationReview,
    CandidateDecision,
    CandidateDecisionBatch,
    EvidenceAdjudicationCaseState,
    ExpressionAudit,
    ExpressionAuditItem,
    GroundedResearchFinding,
    GroundedSource,
)
from l2_core.rag.adjudication.prompts import (
    adjudication_agent_prompt,
    correction_risk_prompt,
    evidence_review_prompt,
    expression_audit_prompt,
)
from l2_core.rag.adjudication.web_research import (
    ChromeAiOverviewSearchClient,
    ChromeAiOverviewSnapshot,
    GeminiGroundedSearchClient,
)
from l2_core.rag.contracts import AnswerPlan, AnswerPlanItem, Evidence, EvidenceChunk, EvidenceRecording


def test_adjudication_proposal_requires_a_search_query() -> None:
    with pytest.raises(ValidationError):
        AdjudicationProposal.model_validate(
            {
                "id": "p1",
                "audit_item_id": "audit-1",
                "evidence_index": 1,
                "chunk_id": "chunk-1",
                "original_expression": "内部指标是 5 秒",
                "proposed_expression": "内部指标是 5 微秒",
                "expression_type": "number",
            }
        )


def test_candidate_decision_migrates_old_checkpoint_score_but_schema_requires_new_score() -> None:
    decision = CandidateDecision.model_validate(
        {
            "proposal_id": "proposal-1",
            "action": "accept",
            "confidence": 0.82,
            "reason": "旧 checkpoint",
            "reconstruct_focus": "",
        }
    )

    assert decision.candidate_score == 0.82
    assert "candidate_score" in CandidateDecision.model_json_schema()["required"]


def test_reject_decision_may_include_reconstruction_focus() -> None:
    decision = CandidateDecision(
        proposal_id="proposal-1",
        action="reject",
        confidence=0.85,
        candidate_score=0.4,
        reason="当前候选不成立",
        reconstruct_focus="根据口语上下文重新生成候选",
    )

    assert decision.reconstruct_focus == "根据口语上下文重新生成候选"


def test_adjudication_validation_error_is_json_serializable_for_logging() -> None:
    with pytest.raises(ValidationError) as captured:
        AdjudicationReview.model_validate(
            {
                "proposals": [
                    {
                        "id": "proposal-1",
                        "audit_item_id": "audit-1",
                        "evidence_index": 1,
                        "chunk_id": "chunk-1",
                        "original_expression": "I2C",
                        "proposed_expression": "I2C",
                        "expression_type": "proper_noun",
                        "search_query": "I2C specification",
                    }
                ]
            }
        )

    payload = EvidenceAdjudicationAgent._error_for_log(captured.value)  # pyright: ignore[reportPrivateUsage]
    serialized = json.dumps(payload, ensure_ascii=False)

    assert "adjudication proposal must change the expression" in serialized
    assert '"type": "ValueError"' in serialized


def test_adjudication_review_filters_one_noop_proposal_without_discarding_valid_siblings() -> None:
    review, rejected = EvidenceAdjudicationAgent._parse_review(  # pyright: ignore[reportPrivateUsage]
        json.dumps(
            {
                "proposals": [
                    {
                        "id": "proposal-1",
                        "audit_item_id": "audit-1",
                        "evidence_index": 1,
                        "chunk_id": "chunk-1",
                        "original_expression": "五微秒",
                        "proposed_expression": "五纳秒",
                        "expression_type": "number",
                        "search_query": "signal latency nanoseconds",
                    },
                    {
                        "id": "proposal-2",
                        "audit_item_id": "audit-2",
                        "evidence_index": 1,
                        "chunk_id": "chunk-1",
                        "original_expression": "保持原文",
                        "proposed_expression": "保持原文",
                        "expression_type": "compound",
                        "search_query": "original expression",
                    },
                ]
            },
            ensure_ascii=False,
        )
    )

    assert [item.id for item in review.proposals] == ["proposal-1"]
    assert len(rejected) == 1
    assert rejected[0]["index"] == 1
    assert "adjudication proposal must change the expression" in json.dumps(rejected, ensure_ascii=False)


def test_agent_state_contains_no_full_evidence_body_field() -> None:
    schema = AdjudicationAgentState.model_json_schema()
    serialized = json.dumps(schema, ensure_ascii=False)

    assert "search_chunk_text" not in serialized
    assert "adjacent_context" not in serialized
    assert "evidence_text" not in serialized


def test_expression_audit_allows_an_empty_result() -> None:
    audit = ExpressionAudit(items=[])
    schema = ExpressionAudit.model_json_schema()

    assert audit.items == []
    serialized = json.dumps(schema)
    assert "contradicting_evidence_indexes" not in serialized
    assert "supporting_evidence_indexes" not in serialized
    audit_index_schema = schema["$defs"]["ExpressionAuditItem"]["properties"]["supporting_evidence_index"]
    assert {item["type"] for item in audit_index_schema["anyOf"]} == {"integer", "null"}

    review_schema = AdjudicationReview.model_json_schema()
    serialized_review = json.dumps(review_schema)
    assert "supporting_evidence_indexes" not in serialized_review
    proposal_index_schema = review_schema["$defs"]["AdjudicationProposal"]["properties"]["supporting_evidence_index"]
    assert {item["type"] for item in proposal_index_schema["anyOf"]} == {"integer", "null"}


def test_valid_audit_keeps_only_valid_reference_indexes() -> None:
    recording_id = uuid4()
    target = Evidence(
        index=1,
        recording=EvidenceRecording(id=recording_id, title="测试", file_name="test.mp3"),
        chunk=EvidenceChunk(id=uuid4(), text="目标表达", start_ms=0, end_ms=1_000),
        score=1,
        match_type="vector",
        url="/recordings/test",
    )
    reference = target.model_copy(update={"index": 2, "chunk": target.chunk.model_copy(update={"id": uuid4()})})
    context = AdjudicationCaseContext(
        query="问题",
        plan=AnswerPlan(items=[AnswerPlanItem(statement="计划", evidence_indexes=[1])]),
        evidence=target,
        reference_evidence=[reference],
        run_id="test-run",
    )
    audit = ExpressionAudit(
        items=[
            ExpressionAuditItem(
                id="invalid-reference",
                expression="目标表达",
                context_quote="目标表达",
                supporting_evidence_index=1,
            ),
            ExpressionAuditItem(
                id="valid-reference",
                expression="目标表达",
                context_quote="目标表达",
                supporting_evidence_index=2,
            ),
        ]
    )

    validated = EvidenceAdjudicationAgent._valid_audit(context, audit)  # pyright: ignore[reportPrivateUsage]

    assert validated.items[0].supporting_evidence_index is None
    assert validated.items[1].supporting_evidence_index == 2


def test_audit_context_quote_resolves_all_matching_spans() -> None:
    text = "RF有规定。别的。RF有规定。RF无关"
    target = Evidence(
        index=1,
        recording=EvidenceRecording(id=uuid4(), title="测试", file_name="test.mp3"),
        chunk=EvidenceChunk(id=uuid4(), text=text, start_ms=0, end_ms=1_000),
        score=1,
        match_type="vector",
        url="/recordings/test",
    )
    context = AdjudicationCaseContext(
        query="问题",
        plan=AnswerPlan(items=[AnswerPlanItem(statement="计划", evidence_indexes=[1])]),
        evidence=target,
        reference_evidence=[],
        run_id="test-run",
    )
    audit = ExpressionAudit(
        items=[
            ExpressionAuditItem(
                id="audit-1",
                expression="RF",
                context_quote="RF有规定",
            ),
            ExpressionAuditItem(
                id="invalid-context",
                expression="RF",
                context_quote="不存在的RF上下文",
            ),
        ]
    )

    validated = EvidenceAdjudicationAgent._valid_audit(context, audit)  # pyright: ignore[reportPrivateUsage]
    spans = EvidenceAdjudicationAgent._target_spans(  # pyright: ignore[reportPrivateUsage]
        text,
        validated.items[0],
    )

    assert [item.id for item in validated.items] == ["audit-1", f"audit-1-occ-{text.rindex('RF')}"]
    assert validated.items[0].context_quote == "RF有规定"
    assert len(spans) == 2
    assert [text[span.start_char : span.end_char] for span in spans] == ["RF", "RF"]
    assert [span.start_char for span in spans] == [0, text.index("RF", 1)]
    assert "status" not in ExpressionAuditItem.model_json_schema()["properties"]
    assert "occurrence_scope" not in ExpressionAuditItem.model_json_schema()["properties"]


def test_audit_item_migrates_removed_scope_and_status_fields() -> None:
    item = ExpressionAuditItem.model_validate(
        {
            "id": "audit-1",
            "expression": "RF",
            "context_quote": "RF有规定",
            "status": "needs_research",
            "occurrence_scope": "all_occurrences",
        }
    )

    assert item.expression == "RF"


def test_valid_proposals_must_use_the_bound_audit_expression() -> None:
    target = Evidence(
        index=1,
        recording=EvidenceRecording(id=uuid4(), title="测试", file_name="test.mp3"),
        chunk=EvidenceChunk(id=uuid4(), text="RF有规定。", start_ms=0, end_ms=1_000),
        score=1,
        match_type="vector",
        url="/recordings/test",
    )
    context = AdjudicationCaseContext(
        query="问题",
        plan=AnswerPlan(items=[AnswerPlanItem(statement="计划", evidence_indexes=[1])]),
        evidence=target,
        reference_evidence=[],
        run_id="test-run",
    )
    case = EvidenceAdjudicationCaseState(
        evidence_index=1,
        chunk_id=target.chunk.id,
        expression_audit=ExpressionAudit(
            items=[
                ExpressionAuditItem(
                    id="audit-1",
                    expression="RF",
                    context_quote="RF有规定",
                )
            ]
        ),
    )
    valid = AdjudicationProposal(
        id="valid",
        audit_item_id="audit-1",
        evidence_index=1,
        chunk_id=str(target.chunk.id),
        original_expression="RF",
        proposed_expression="I2C",
        expression_type="proper_noun",
        search_query="I2C protocol",
    )
    mismatched = valid.model_copy(update={"id": "mismatched", "original_expression": "RF有规定"})

    proposals = EvidenceAdjudicationAgent._valid_proposals(  # pyright: ignore[reportPrivateUsage]
        case,
        context,
        [mismatched, valid],
    )

    assert [proposal.id for proposal in proposals] == ["valid"]


def test_valid_proposals_limits_each_audit_item_to_two_candidates() -> None:
    target = Evidence(
        index=1,
        recording=EvidenceRecording(id=uuid4(), title="测试", file_name="test.mp3"),
        chunk=EvidenceChunk(id=uuid4(), text="RF有规定。", start_ms=0, end_ms=1_000),
        score=1,
        match_type="vector",
        url="/recordings/test",
    )
    context = AdjudicationCaseContext(
        query="问题",
        plan=AnswerPlan(items=[AnswerPlanItem(statement="计划", evidence_indexes=[1])]),
        evidence=target,
        reference_evidence=[],
        run_id="test-run",
    )
    case = EvidenceAdjudicationCaseState(
        evidence_index=1,
        chunk_id=target.chunk.id,
        expression_audit=ExpressionAudit(
            items=[ExpressionAuditItem(id="audit-1", expression="RF", context_quote="RF有规定")]
        ),
    )
    proposals = [
        AdjudicationProposal(
            id=f"proposal-{index}",
            audit_item_id="audit-1",
            evidence_index=1,
            chunk_id=str(target.chunk.id),
            original_expression="RF",
            proposed_expression=replacement,
            expression_type="proper_noun",
            search_query=f"{replacement} protocol",
        )
        for index, replacement in enumerate(["I2C", "AUX", "SPI"], start=1)
    ]

    valid = EvidenceAdjudicationAgent._valid_proposals(case, context, proposals)  # pyright: ignore[reportPrivateUsage]

    assert [proposal.id for proposal in valid] == ["proposal-1", "proposal-2"]


def test_candidate_decision_postprocess_discards_extra_duplicate_ids(caplog: pytest.LogCaptureFixture) -> None:
    chunk_id = uuid4()

    def proposal(proposal_id: str) -> AdjudicationProposal:
        return AdjudicationProposal(
            id=proposal_id,
            audit_item_id=f"audit-{proposal_id}",
            evidence_index=1,
            chunk_id=str(chunk_id),
            original_expression=proposal_id,
            proposed_expression=f"修正{proposal_id}",
            expression_type="proper_noun",
            search_query=proposal_id,
        )

    case = EvidenceAdjudicationCaseState(evidence_index=1, chunk_id=chunk_id, proposals=[proposal("proposal-1"), proposal("proposal-2")])
    decisions = CandidateDecisionBatch(
        decisions=[
            CandidateDecision(
                proposal_id="proposal-1", action="accept", confidence=0.9, candidate_score=0.9, reason="首条", reconstruct_focus=""
            ),
            CandidateDecision(
                proposal_id="proposal-2", action="reject", confidence=0.5, candidate_score=0.5, reason="第二条", reconstruct_focus=""
            ),
            CandidateDecision(
                proposal_id="proposal-1", action="reject", confidence=0.1, candidate_score=0.1, reason="重复条", reconstruct_focus=""
            ),
        ]
    )
    target = Evidence(
        index=1,
        recording=EvidenceRecording(id=uuid4(), title="测试", file_name="test.mp3"),
        chunk=EvidenceChunk(id=chunk_id, text="测试", start_ms=0, end_ms=1_000),
        score=1,
        match_type="vector",
        url="/recordings/test",
    )
    context = AdjudicationCaseContext(
        query="问题",
        plan=AnswerPlan(items=[AnswerPlanItem(statement="计划", evidence_indexes=[1])]),
        evidence=target,
        reference_evidence=[],
        run_id="test-run",
    )

    with caplog.at_level("WARNING", logger="rag"):
        normalized, missing_ids = EvidenceAdjudicationAgent._normalize_candidate_decisions(  # pyright: ignore[reportPrivateUsage]
            case, decisions, context
        )

    assert [item.proposal_id for item in normalized.decisions] == ["proposal-1", "proposal-2"]
    assert missing_ids == []
    assert "discarded_duplicate_proposal_ids=['proposal-1']" in caplog.text


def test_candidate_decision_retries_once_for_missing_proposal_ids(caplog: pytest.LogCaptureFixture) -> None:
    chunk_id = uuid4()
    target = Evidence(
        index=1,
        recording=EvidenceRecording(id=uuid4(), title="测试", file_name="test.mp3"),
        chunk=EvidenceChunk(id=chunk_id, text="术语甲和术语乙", start_ms=0, end_ms=1_000),
        score=1,
        match_type="vector",
        url="/recordings/test",
    )

    def proposal(proposal_id: str, expression: str) -> AdjudicationProposal:
        return AdjudicationProposal(
            id=proposal_id,
            audit_item_id=f"audit-{proposal_id}",
            evidence_index=1,
            chunk_id=str(chunk_id),
            original_expression=expression,
            proposed_expression=f"修正{expression}",
            expression_type="proper_noun",
            search_query=expression,
        )

    case = EvidenceAdjudicationCaseState(
        evidence_index=1,
        chunk_id=chunk_id,
        expression_audit=ExpressionAudit(
            items=[
                ExpressionAuditItem(id="audit-proposal-1", expression="术语甲", context_quote="术语甲"),
                ExpressionAuditItem(id="audit-proposal-2", expression="术语乙", context_quote="术语乙"),
            ]
        ),
        proposals=[proposal("proposal-1", "术语甲"), proposal("proposal-2", "术语乙")],
        initial_reconstruction_completed=True,
    )
    state = AdjudicationAgentState(risk=True, cases=[case])
    context = AdjudicationCaseContext(
        query="问题",
        plan=AnswerPlan(items=[AnswerPlanItem(statement="计划", evidence_indexes=[1])]),
        evidence=target,
        reference_evidence=[],
        run_id="test-run",
    )
    calls = 0

    def decision(proposal_id: str) -> dict[str, object]:
        return {
            "proposal_id": proposal_id,
            "action": "reject",
            "confidence": 0.5,
            "candidate_score": 0.5,
            "reason": "不成立",
            "reconstruct_focus": "",
        }

    async def complete(*_: object) -> LlmGenerateResult:
        nonlocal calls
        calls += 1
        payload = {"decisions": [decision("proposal-1"), decision("unknown")]} if calls == 1 else {"decisions": [decision("proposal-2")]}
        return LlmGenerateResult(text=json.dumps(payload), provider=LlmProvider.GEMINI, model="gemini-test")

    agent = object.__new__(EvidenceAdjudicationAgent)
    agent._max_iterations = 4  # pyright: ignore[reportPrivateUsage]
    agent._max_searches = 3  # pyright: ignore[reportPrivateUsage]
    agent._grounded_search_client = None  # pyright: ignore[reportPrivateUsage]
    with caplog.at_level("WARNING", logger="rag"):
        transition = asyncio.run(agent.decide_candidate_actions(state, context, complete=complete))

    assert calls == 2
    assert transition.outcome == "candidate_actions"
    assert [item.proposal_id for item in transition.state.cases[0].pending_decisions.decisions] == [  # type: ignore[union-attr]
        "proposal-1",
        "proposal-2",
    ]
    assert "Candidate 决策缺失，开始单次补全" in caplog.text


def test_candidate_actions_keep_highest_scoring_non_overlapping_accept(caplog: pytest.LogCaptureFixture) -> None:
    chunk_id = uuid4()
    target = Evidence(
        index=1,
        recording=EvidenceRecording(id=uuid4(), title="测试", file_name="test.mp3"),
        chunk=EvidenceChunk(id=chunk_id, text="USB20", start_ms=0, end_ms=1_000),
        score=1,
        match_type="vector",
        url="/recordings/test",
    )
    audit = ExpressionAudit(
        items=[
            ExpressionAuditItem(id="audit-usb", expression="USB20", context_quote="USB20"),
            ExpressionAuditItem(id="audit-version", expression="20", context_quote="USB20"),
        ]
    )
    proposals = [
        AdjudicationProposal(
            id="proposal-usb",
            audit_item_id="audit-usb",
            evidence_index=1,
            chunk_id=str(chunk_id),
            original_expression="USB20",
            proposed_expression="USB 2.0",
            expression_type="proper_noun",
            search_query="USB 2.0",
        ),
        AdjudicationProposal(
            id="proposal-version",
            audit_item_id="audit-version",
            evidence_index=1,
            chunk_id=str(chunk_id),
            original_expression="20",
            proposed_expression="2.0",
            expression_type="proper_noun",
            search_query="USB 2.0",
        ),
    ]
    decisions = CandidateDecisionBatch(
        decisions=[
            CandidateDecision(
                proposal_id="proposal-usb", action="accept", confidence=0.9, candidate_score=0.9, reason="可接受", reconstruct_focus=""
            ),
            CandidateDecision(
                proposal_id="proposal-version", action="accept", confidence=0.95, candidate_score=0.95, reason="更高分", reconstruct_focus=""
            ),
        ]
    )
    case = EvidenceAdjudicationCaseState(
        evidence_index=1,
        chunk_id=chunk_id,
        expression_audit=audit,
        proposals=proposals,
        initial_reconstruction_completed=True,
        pending_decisions=decisions,
    )
    state = AdjudicationAgentState(risk=True, cases=[case])
    context = AdjudicationCaseContext(
        query="问题",
        plan=AnswerPlan(items=[AnswerPlanItem(statement="计划", evidence_indexes=[1])]),
        evidence=target,
        reference_evidence=[],
        run_id="test-run",
    )
    agent = object.__new__(EvidenceAdjudicationAgent)
    agent._max_searches = 3  # pyright: ignore[reportPrivateUsage]
    agent._grounded_search_client = None  # pyright: ignore[reportPrivateUsage]

    async def reconstruct(*_: object) -> LlmGenerateResult:
        raise AssertionError("an overlap-discarded accept must not trigger reconstruction")

    with caplog.at_level("WARNING", logger="rag"):
        transition = asyncio.run(agent.execute_candidate_actions(state, context, reconstruct=reconstruct))

    assert [overlay.proposal_id for overlay in transition.state.overlays] == ["proposal-version"]
    assert transition.state.cases[0].rejected_proposal_ids == ["proposal-usb"]
    assert "接受候选重叠，保留更高 candidate_score" in caplog.text


def test_candidate_actions_accept_reject_and_run_search_with_reconstruction_in_parallel() -> None:
    text = "术语甲，参数乙，名称丙，表达丁。"
    target = Evidence(
        index=1,
        recording=EvidenceRecording(id=uuid4(), title="测试", file_name="test.mp3"),
        chunk=EvidenceChunk(id=uuid4(), text=text, start_ms=0, end_ms=1_000),
        score=1,
        match_type="vector",
        url="/recordings/test",
    )
    audit = ExpressionAudit(
        items=[
            ExpressionAuditItem(
                id=f"audit-{index}",
                expression=expression,
                context_quote=expression,
            )
            for index, expression in enumerate(["术语甲", "参数乙", "名称丙", "表达丁"], start=1)
        ]
    )

    def proposal(index: int, expression: str) -> AdjudicationProposal:
        return AdjudicationProposal(
            id=f"proposal-{index}",
            audit_item_id=f"audit-{index}",
            evidence_index=1,
            chunk_id=str(target.chunk.id),
            original_expression=expression,
            proposed_expression=f"修正{index}",
            expression_type="proper_noun",
            search_query=f"{expression} 技术资料",
        )

    proposals = [
        proposal(index, expression)
        for index, expression in enumerate(["术语甲", "参数乙", "名称丙", "表达丁"], start=1)
    ]
    decisions = CandidateDecisionBatch(
        decisions=[
            CandidateDecision(
                proposal_id="proposal-1",
                action="accept",
                confidence=0.96,
                candidate_score=0.96,
                reason="上下文支持",
                reconstruct_focus="",
            ),
            CandidateDecision(
                proposal_id="proposal-2",
                action="web_search",
                confidence=0.6,
                candidate_score=0.7,
                reason="需要外部证据",
                reconstruct_focus="",
            ),
            CandidateDecision(
                proposal_id="proposal-3",
                action="reconstruct",
                confidence=0.2,
                candidate_score=0.3,
                reason="当前候选不成立",
                reconstruct_focus="重新判断名称丙",
            ),
            CandidateDecision(
                proposal_id="proposal-4",
                action="reject",
                confidence=0.1,
                candidate_score=0.1,
                reason="当前候选会破坏原有语义角色",
                reconstruct_focus="",
            ),
        ]
    )
    case = EvidenceAdjudicationCaseState(
        evidence_index=1,
        chunk_id=target.chunk.id,
        expression_audit=audit,
        proposals=proposals,
        initial_reconstruction_completed=True,
        pending_decisions=decisions,
    )
    state = AdjudicationAgentState(risk=True, cases=[case], web_search_enabled=True)
    context = AdjudicationCaseContext(
        query="问题",
        plan=AnswerPlan(items=[AnswerPlanItem(statement="计划", evidence_indexes=[1])]),
        evidence=target,
        reference_evidence=[],
        run_id="test-run",
    )
    search_started = asyncio.Event()
    reconstruction_started = asyncio.Event()

    class SearchClient:
        async def search(self, proposal_id: str, query: str) -> GroundedResearchFinding:
            search_started.set()
            await reconstruction_started.wait()
            return GroundedResearchFinding(proposal_id=proposal_id, query=query, summary="搜索结果")

        async def close(self) -> None:
            return None

    reconstruction_inputs: list[str] = []

    async def reconstruct(messages: list[BaseMessage], *_: object) -> LlmGenerateResult:
        reconstruction_started.set()
        await search_started.wait()
        reconstruction_inputs.extend(str(getattr(message, "content", message)) for message in messages)
        return LlmGenerateResult(
            text=json.dumps(
                {
                    "proposals": [
                        {
                            **proposals[2].model_dump(mode="json"),
                            "id": "proposal-3-rebuilt",
                            "proposed_expression": "重建名称",
                        },
                        {
                            **proposals[3].model_dump(mode="json"),
                            "id": "proposal-4-rebuilt",
                            "proposed_expression": "重建表达",
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            provider=LlmProvider.GEMINI,
            model="gemini-test",
        )

    agent = object.__new__(EvidenceAdjudicationAgent)
    agent._grounded_search_client = SearchClient()  # type: ignore[assignment]  # pyright: ignore[reportPrivateUsage]
    agent._max_searches = 2  # pyright: ignore[reportPrivateUsage]

    transition = asyncio.run(agent.execute_candidate_actions(state, context, reconstruct=reconstruct))
    updated = transition.state.cases[0]

    assert [overlay.proposal_id for overlay in transition.state.overlays] == ["proposal-1"]
    assert {proposal.id for proposal in updated.proposals} == {
        "proposal-2",
        "proposal-3-rebuilt",
        "proposal-4-rebuilt",
    }
    assert [finding.proposal_id for finding in updated.findings] == ["proposal-2"]
    assert updated.accepted_proposal_ids == ["proposal-1"]
    assert updated.rejected_proposal_ids == ["proposal-4"]
    assert "当前候选会破坏原有语义角色" in "\n".join(reconstruction_inputs)
    assert updated.search_count == 1
    assert updated.pending_decisions is None
    assert len(updated.decision_history) == 1


def test_candidate_actions_choose_by_audit_group_priority_and_score() -> None:
    target = Evidence(
        index=1,
        recording=EvidenceRecording(id=uuid4(), title="测试", file_name="test.mp3"),
        chunk=EvidenceChunk(id=uuid4(), text="RF有规定，五秒不合理，名称异常。", start_ms=0, end_ms=1_000),
        score=1,
        match_type="vector",
        url="/recordings/test",
    )
    audit = ExpressionAudit(
        items=[
            ExpressionAuditItem(id="audit-rf", expression="RF", context_quote="RF有规定"),
            ExpressionAuditItem(id="audit-time", expression="五秒", context_quote="五秒不合理"),
            ExpressionAuditItem(id="audit-name", expression="名称", context_quote="名称异常"),
        ]
    )

    def proposal(proposal_id: str, audit_item_id: str, original: str, replacement: str) -> AdjudicationProposal:
        return AdjudicationProposal(
            id=proposal_id,
            audit_item_id=audit_item_id,
            evidence_index=1,
            chunk_id=str(target.chunk.id),
            original_expression=original,
            proposed_expression=replacement,
            expression_type="number" if audit_item_id == "audit-time" else "proper_noun",
            search_query=f"{replacement} specification",
        )

    proposals = [
        proposal("rf-low", "audit-rf", "RF", "AUX"),
        proposal("rf-high", "audit-rf", "RF", "I2C"),
        proposal("time-reject", "audit-time", "五秒", "五毫秒"),
        proposal("time-search", "audit-time", "五秒", "五微秒"),
        proposal("name-search-1", "audit-name", "名称", "MIPI"),
        proposal("name-search-2", "audit-name", "名称", "LVDS"),
    ]
    decisions = CandidateDecisionBatch(
        decisions=[
            CandidateDecision(
                proposal_id="rf-low", action="accept", confidence=0.8, candidate_score=0.4,
                reason="可解释", reconstruct_focus="",
            ),
            CandidateDecision(
                proposal_id="rf-high", action="accept", confidence=0.9, candidate_score=0.95,
                reason="整体最自洽", reconstruct_focus="",
            ),
            CandidateDecision(
                proposal_id="time-reject", action="reject", confidence=0.8, candidate_score=0.2,
                reason="数量级仍错误", reconstruct_focus="",
            ),
            CandidateDecision(
                proposal_id="time-search", action="web_search", confidence=0.7, candidate_score=0.85,
                reason="需要规范确认", reconstruct_focus="",
            ),
            CandidateDecision(
                proposal_id="name-search-1", action="web_search", confidence=0.6, candidate_score=0.6,
                reason="两个候选都需要检索", reconstruct_focus="",
            ),
            CandidateDecision(
                proposal_id="name-search-2", action="web_search", confidence=0.7, candidate_score=0.7,
                reason="两个候选都需要检索", reconstruct_focus="",
            ),
        ]
    )
    case = EvidenceAdjudicationCaseState(
        evidence_index=1,
        chunk_id=target.chunk.id,
        expression_audit=audit,
        proposals=proposals,
        initial_reconstruction_completed=True,
        pending_decisions=decisions,
    )
    state = AdjudicationAgentState(risk=True, cases=[case], web_search_enabled=True)
    context = AdjudicationCaseContext(
        query="问题",
        plan=AnswerPlan(items=[AnswerPlanItem(statement="计划", evidence_indexes=[1])]),
        evidence=target,
        reference_evidence=[],
        run_id="test-run",
    )
    searched: list[str] = []

    class SearchClient:
        async def search(self, proposal_id: str, query: str) -> GroundedResearchFinding:
            searched.append(proposal_id)
            return GroundedResearchFinding(proposal_id=proposal_id, query=query, summary="搜索结果")

        async def close(self) -> None:
            return None

    async def reconstruct(*_: object) -> LlmGenerateResult:
        raise AssertionError("accept/web_search groups must not reconstruct")

    agent = object.__new__(EvidenceAdjudicationAgent)
    agent._grounded_search_client = SearchClient()  # type: ignore[assignment]  # pyright: ignore[reportPrivateUsage]
    agent._max_searches = 3  # pyright: ignore[reportPrivateUsage]

    transition = asyncio.run(agent.execute_candidate_actions(state, context, reconstruct=reconstruct))
    updated = transition.state.cases[0]

    assert [overlay.proposal_id for overlay in transition.state.overlays] == ["rf-high"]
    assert [proposal.id for proposal in updated.proposals] == ["time-search", "name-search-1", "name-search-2"]
    assert searched == ["time-search", "name-search-1", "name-search-2"]
    assert [finding.proposal_id for finding in updated.findings] == searched
    assert updated.accepted_proposal_ids == ["rf-high"]
    assert updated.rejected_proposal_ids == ["time-reject"]
    assert updated.decision_history == [decisions]


def test_empty_expression_audit_ends_case_without_reconstruction(caplog: pytest.LogCaptureFixture) -> None:
    case = EvidenceAdjudicationCaseState(evidence_index=1, chunk_id=uuid4(), pending_setup_phase="audit")
    state = AdjudicationAgentState(risk=True, cases=[case])
    evidence = Evidence(
        index=1,
        recording=EvidenceRecording(id=uuid4(), title="测试", file_name="test.mp3"),
        chunk=EvidenceChunk(id=case.chunk_id, text="普通表达", start_ms=0, end_ms=1_000),
        score=1,
        match_type="vector",
        url="/recordings/test",
    )
    context = AdjudicationCaseContext(
        query="问题",
        plan=AnswerPlan(items=[AnswerPlanItem(statement="计划", evidence_indexes=[1])]),
        evidence=evidence,
        reference_evidence=[],
        run_id="test-run",
    )
    agent = object.__new__(EvidenceAdjudicationAgent)
    agent._audit_prompt_variant = "relation_rules"  # pyright: ignore[reportPrivateUsage]

    async def reconstruct(*_: object) -> LlmGenerateResult:
        return LlmGenerateResult(text='{"items":[]}', provider=LlmProvider.GEMINI, model="gemini-test")

    with caplog.at_level("INFO", logger="rag"):
        transition = asyncio.run(
            agent.execute_setup_phase(state, context, complete_audit=reconstruct, reconstruct=reconstruct)
        )

    assert transition.outcome == "audit_empty"
    assert transition.terminal
    assert transition.state.status == "completed"
    assert transition.state.cases[0].status == "rejected"
    assert transition.state.cases[0].pending_setup_phase is None
    assert "Evidence 裁决 Agent 最终审计 run_id=test-run case=0 item_count=0 items=[]" in caplog.text


def test_initial_reconstruction_retries_once_for_missing_audit_items(caplog: pytest.LogCaptureFixture) -> None:
    target = Evidence(
        index=1,
        recording=EvidenceRecording(id=uuid4(), title="测试", file_name="test.mp3"),
        chunk=EvidenceChunk(id=uuid4(), text="RF 与 USB 都需要核验。", start_ms=0, end_ms=1_000),
        score=1,
        match_type="vector",
        url="/recordings/test",
    )
    audit = ExpressionAudit(
        items=[
            ExpressionAuditItem(id="audit-rf", expression="RF", context_quote="RF 与 USB"),
            ExpressionAuditItem(id="audit-usb", expression="USB", context_quote="RF 与 USB"),
        ]
    )
    case = EvidenceAdjudicationCaseState(
        evidence_index=target.index,
        chunk_id=target.chunk.id,
        expression_audit=audit,
        pending_setup_phase="initial_reconstruct",
    )
    state = AdjudicationAgentState(risk=True, cases=[case])
    context = AdjudicationCaseContext(
        query="问题",
        plan=AnswerPlan(items=[AnswerPlanItem(statement="计划", evidence_indexes=[1])]),
        evidence=target,
        reference_evidence=[],
        run_id="test-run",
    )
    calls = 0

    def proposal(audit_id: str, original: str, replacement: str) -> dict[str, object]:
        return {
            "id": f"proposal-{audit_id}",
            "audit_item_id": audit_id,
            "evidence_index": target.index,
            "chunk_id": str(target.chunk.id),
            "original_expression": original,
            "proposed_expression": replacement,
            "expression_type": "proper_noun",
            "search_query": f"{replacement} protocol",
        }

    async def reconstruct(*_: object) -> LlmGenerateResult:
        nonlocal calls
        calls += 1
        proposals = [proposal("audit-rf", "RF", "I2C")] if calls == 1 else [proposal("audit-usb", "USB", "SPI")]
        return LlmGenerateResult(
            text=json.dumps({"proposals": proposals}), provider=LlmProvider.GEMINI, model="gemini-test"
        )

    agent = object.__new__(EvidenceAdjudicationAgent)
    with caplog.at_level("WARNING", logger="rag"):
        transition = asyncio.run(
            agent.execute_setup_phase(state, context, complete_audit=reconstruct, reconstruct=reconstruct)
        )

    assert calls == 2
    assert {proposal.audit_item_id for proposal in transition.state.cases[0].proposals} == {"audit-rf", "audit-usb"}
    assert "首次候选重建缺少审计项，开始单次重试" in caplog.text


def test_audit_and_candidate_reconstruction_use_independent_completion_policies() -> None:
    commands: list[Any] = []
    budget_nodes: list[str] = []

    class CaptureClient:
        async def execute(self, command: object, *, result_type: object) -> LlmGenerateResult:
            commands.append(command)
            return LlmGenerateResult(text="{}", provider=LlmProvider.GEMINI, model="gemini-test")

    class CaptureBudget:
        def before_model(self, _used_tokens: int, node: str) -> None:
            budget_nodes.append(node)

    agent = object.__new__(EvidenceAdjudicationAgent)
    untyped_agent = cast(Any, agent)
    untyped_agent._model_client = CaptureClient()
    untyped_agent._online_provider = LlmProvider.GEMINI
    untyped_agent._context_size = 16_384
    untyped_agent._token_budget = CaptureBudget()
    untyped_agent._audit_model = "gemini-3.6-flash"
    untyped_agent._audit_min_request_interval_seconds = 15.0
    state = {"token_usage": 0}
    messages = [HumanMessage(content="测试")]
    schema = {"type": "object"}

    asyncio.run(agent._complete_expression_audit(state, messages, schema))  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage]
    asyncio.run(agent._complete_candidate_reconstruction(state, messages, schema))  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage]
    asyncio.run(agent._complete_candidate_decisions(state, messages, schema))  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage]

    audit_options = commands[0].input.options
    reconstruction_options = commands[1].input.options
    decision_options = commands[2].input.options
    assert audit_options.model == "gemini-3.6-flash"
    assert audit_options.max_tokens == 5_000
    assert audit_options.min_request_interval_seconds == 15
    assert reconstruction_options.model is None
    assert reconstruction_options.max_tokens == 8_000
    assert reconstruction_options.min_request_interval_seconds is None
    assert decision_options.max_tokens == 8_000
    assert budget_nodes == [
        "adjudication_audit_expressions",
        "adjudication_reconstruct_candidates",
        "adjudication_candidate_decisions",
    ]


def test_new_setup_flow_is_explicitly_audit_then_initial_reconstruct() -> None:
    case = EvidenceAdjudicationCaseState(evidence_index=1, chunk_id=uuid4())

    assert EvidenceAdjudicationAgent._required_setup_phase(case) == "audit"  # pyright: ignore[reportPrivateUsage]

    audited = case.model_copy(
        update={
            "expression_audit": ExpressionAudit(
                items=[
                    ExpressionAuditItem(
                        id="audit-1",
                        expression="RF",
                        context_quote="RF 有规定",
                        semantic_role="协议名称",
                        reason="上下文不支持射频含义",
                    )
                ]
            )
        }
    )
    assert EvidenceAdjudicationAgent._required_setup_phase(audited) == "initial_reconstruct"  # pyright: ignore[reportPrivateUsage]

    reconstructed = audited.model_copy(update={"initial_reconstruction_completed": True})
    assert EvidenceAdjudicationAgent._required_setup_phase(reconstructed) is None  # pyright: ignore[reportPrivateUsage]


def test_adjudication_initialization_honors_configured_case_limit() -> None:
    evidence = [
        Evidence(
            index=index,
            recording=EvidenceRecording(id=uuid4(), title=f"录音 {index}", file_name=f"{index}.mp3"),
            chunk=EvidenceChunk(id=uuid4(), text=f"证据 {index}", start_ms=0, end_ms=1_000),
            score=1,
            match_type="vector",
            url=f"/recordings/{index}",
        )
        for index in range(1, 7)
    ]
    agent = object.__new__(EvidenceAdjudicationAgent)
    agent._max_cases = 5  # pyright: ignore[reportPrivateUsage]

    state = agent.initialize(True, evidence, web_search_enabled=False)

    assert [case.evidence_index for case in state.cases] == [1, 2, 3, 4, 5]


def test_adjudication_context_uses_up_to_five_cross_recording_references() -> None:
    evidence = [
        Evidence(
            index=index,
            recording=EvidenceRecording(id=uuid4(), title=f"录音 {index}", file_name=f"{index}.mp3"),
            chunk=EvidenceChunk(id=uuid4(), text=f"证据 {index}", start_ms=index * 1_000, end_ms=(index + 1) * 1_000),
            score=1,
            match_type="vector",
            url=f"/recordings/{index}",
        )
        for index in range(1, 8)
    ]
    target = evidence[2]
    agent = object.__new__(EvidenceAdjudicationAgent)
    agent_state = AdjudicationAgentState(
        risk=True,
        cases=[EvidenceAdjudicationCaseState(evidence_index=target.index, chunk_id=target.chunk.id)],
    )

    context = agent._context(  # pyright: ignore[reportPrivateUsage]
        cast(
            Any,
            {
                "query": "问题",
                "answer_plan": None,
                "answer_evidence": evidence,
                "run_id": "run-1",
            },
        ),
        agent_state,
    )

    assert [item.index for item in context.reference_evidence] == [1, 2, 4, 5, 6]
    assert all(item.recording.id != target.recording.id for item in context.reference_evidence)


def test_adjudication_static_prompts_are_bounded_and_contain_no_example_answers() -> None:
    recording_id = uuid4()
    evidence = Evidence(
        index=1,
        recording=EvidenceRecording(id=recording_id, title="测试", file_name="test.mp3"),
        chunk=EvidenceChunk(id=uuid4(), text="运行时证据", start_ms=0, end_ms=1_000),
        score=1,
        match_type="vector",
        url=f"/recordings/{recording_id}",
    )
    prompts = [
        correction_risk_prompt("问题"),
        expression_audit_prompt(
            "问题",
            AnswerPlan(items=[AnswerPlanItem(statement="计划", evidence_indexes=[1])]),
            evidence,
            [],
        ),
        evidence_review_prompt(
            "问题",
            AnswerPlan(items=[AnswerPlanItem(statement="计划", evidence_indexes=[1])]),
            evidence,
            expression_audit=ExpressionAudit(
                items=[
                    ExpressionAuditItem(
                        id="audit-1",
                        expression="运行时证据",
                        context_quote="运行时证据",
                    )
                ]
            ),
        ),
        adjudication_agent_prompt(
            "问题",
            True,
            AnswerPlan(items=[AnswerPlanItem(statement="计划", evidence_indexes=[1])]),
            evidence,
            [],
            EvidenceAdjudicationCaseState(evidence_index=1, chunk_id=evidence.chunk.id),
            3,
            3,
        ),
    ]

    for prompt, values in prompts:
        system_content = prompt.invoke(values).to_messages()[0].content
        assert isinstance(system_content, str)
        system_text = system_content
        assert len(system_text) <= 1_100
        assert "I²C" not in system_text
        assert "RF" not in system_text


def test_evidence_review_prompt_declares_semiconductor_asr_context_and_contextual_review() -> None:
    recording_id = uuid4()
    evidence = Evidence(
        index=1,
        recording=EvidenceRecording(id=recording_id, title="测试", file_name="test.mp3"),
        chunk=EvidenceChunk(id=uuid4(), text="运行时证据", start_ms=0, end_ms=1_000),
        score=1,
        match_type="vector",
        url=f"/recordings/{recording_id}",
    )

    prompt, values = evidence_review_prompt(
        "问题",
        AnswerPlan(items=[AnswerPlanItem(statement="计划", evidence_indexes=[1])]),
        evidence,
        expression_audit=ExpressionAudit(
            items=[
                ExpressionAuditItem(
                    id="audit-1",
                    expression="运行时证据",
                    context_quote="运行时证据",
                    reason="疑似应替换为不应传给 Review 的候选",
                )
            ]
        ),
    )
    system_content = prompt.invoke(values).to_messages()[0].content
    assert isinstance(system_content, str)
    system_text = system_content

    assert "把 Expression Audit 中的表达重建为 ASR 修正候选" in system_text
    assert "Target Evidence 是唯一修改目标" in system_text
    assert "Audit 只定义疑点和范围，不提供候选" in system_text
    assert "忽略其自由文本中意外出现的替换方向" in system_text
    assert "必须独立根据 Target、Reference 和 Findings 推导并代回验证" in system_text
    assert "独立得到相同结果可用" in system_text
    audit_payload = json.loads(values["expression_audit"])
    assert audit_payload["items"][0]["expression"] == "运行时证据"
    assert "reason" not in audit_payload["items"][0]
    assert "不应传给 Review 的候选" not in values["expression_audit"]
    assert json.loads(values["required_audit_item_ids"]) == ["audit-1"]
    assert "audit_item_id" in system_text
    assert "候选与原表达不要求同音、近音或字形相似" in system_text
    assert "把候选代入该 item 的 context_quote" in system_text
    assert "技术对象、信号、动作、属性、数字、单位、条件和因果链" in system_text
    assert "局部修正后，必须重新检查同一因果链" in system_text
    assert "补充原文不存在的隐含术语" in system_text
    assert "相互一致的关联候选" in system_text
    assert "不得为 Expression Audit 中不存在的表达编造 audit_item_id" in system_text
    assert "Target 原文是待审查 ASR，不是事实" in system_text
    assert "不能因为它能被讲成一个合理故事" in system_text
    assert "模型知识只能提出候选和 search_query" in system_text
    assert "supporting_evidence_index" in system_text
    assert "proposed_expression 去除首尾空白后必须与 original_expression 不同" in system_text
    assert "你没有否定或省略 Audit items 的权限" in system_text
    assert "必须逐项覆盖 items 里的每一项，每一项输出1至2个 proposal" in system_text
    assert "依据不足时也要给出最可能且可检验的候选" in system_text
    assert "禁止静默省略" in system_text
    assert "禁止用 no-op proposal 表示保留原文" in system_text
    assert "一、证据边界" in system_text
    assert "二、逐项重建" in system_text
    assert "三、生成职责" in system_text
    assert "四、输出规则" in system_text


def test_expression_audit_prompt_requires_exhaustive_systematic_error_review() -> None:
    recording_id = uuid4()
    evidence = Evidence(
        index=1,
        recording=EvidenceRecording(id=recording_id, title="测试", file_name="test.mp3"),
        chunk=EvidenceChunk(id=uuid4(), text="目标表达重复出现。", start_ms=0, end_ms=1_000),
        score=1,
        match_type="vector",
        url=f"/recordings/{recording_id}",
    )
    prompt, values = expression_audit_prompt(
        "问题",
        AnswerPlan(items=[AnswerPlanItem(statement="计划", evidence_indexes=[1])]),
        evidence,
        [],
    )
    content = prompt.invoke(values).to_messages()[0].content
    assert isinstance(content, str)
    assert "高召回疑点审计" in content
    assert "所有值得重建或核验的原文" in content
    assert "Target 是唯一审计对象" in content
    assert "Reference 可能错误、换题或无关" in content
    assert "只能辅助判断 Target" in content
    assert "不审计其自身问题" in content
    assert "先独立理解 Target 的整体技术叙述" in content
    assert "这些类别具有相同的审计优先级" in content
    assert "以完整语义关系为单位审计" in content
    assert "通常技术身份是否适合该角色" in content
    assert "都不能单独证明它正确" in content
    assert "稳定重复同一种合法但角色错误的表达" in content
    assert "暂时把已发现的异常视为未知" in content
    assert "不得把同一关系中的其他表达视为已确认事实" in content
    assert "不得只选择最显眼或最容易解释的一项" in content
    assert "可在内部比较解释以定位疑点" in content
    assert "不得输出替换方向" in content
    assert "reason、semantic_role 等任何字段" in content
    assert "均不得出现具体替换词、缩写、名称、数值或示例候选" in content
    assert "只描述原表达的角色及可疑性" in content
    assert "expression 和 context_quote 必须逐字存在于 Target" in content
    assert "其全部匹配位置均属该疑点" in content
    assert "不输出位置或序号" in content
    assert "expression 按独立疑点拆分" in content
    assert "可单独成 item" in content
    assert "独立疑点分别输出" in content
    assert "不可分割的才合并" in content
    assert "不得遗漏或无故扩大范围" in content
    assert "reason 先说明表达在 Target 中的语义角色" in content
    assert "reason 不得评价或修正 Reference" in content
    assert "reason 不需要确定正确答案" in content
    assert "不得提出替换文本" in content
    assert "不输出 supported 表达" in content
    assert "supporting_evidence_index 填最相关的 Reference index，无则为 null" in content
    assert "没有疑点时输出空 items" in content
    human_content = prompt.invoke(values).to_messages()[1].content
    assert isinstance(human_content, str)
    assert human_content.index("Reference Evidence（仅辅助判断，不审计）") < human_content.index(
        "Target Evidence（唯一审计对象）"
    )


def test_free_discovery_audit_prompt_is_concise_and_preserves_output_boundaries() -> None:
    recording_id = uuid4()
    evidence = Evidence(
        index=1,
        recording=EvidenceRecording(id=recording_id, title="测试", file_name="test.mp3"),
        chunk=EvidenceChunk(id=uuid4(), text="目标表达。", start_ms=0, end_ms=1_000),
        score=1,
        match_type="vector",
        url=f"/recordings/{recording_id}",
    )
    prompt, values = expression_audit_prompt(
        "问题",
        AnswerPlan(items=[AnswerPlanItem(statement="计划", evidence_indexes=[1])]),
        evidence,
        [],
        variant="free_discovery",
    )
    content = prompt.invoke(values).to_messages()[0].content
    assert isinstance(content, str)
    assert "先独立阅读完整 Target，自由判断" in content
    assert "context_quote 必须是从 Target 逐字复制" in content
    assert "不提出修正答案" in content
    assert "不得因为已经找到一个高置信疑点而停止" in content
    assert "适合承担上下文赋予它的技术角色" in content
    assert "不得输出替换方向" in content
    assert "均不得出现具体替换词、缩写、名称、数值或示例候选" in content
    assert "只描述原表达的角色及可疑性" in content
    assert "不要输出 offset、出现次数或位置序号" in content
    assert "Reference 只能辅助判断疑点" in content
    assert "supporting_evidence_index 填最相关的 Reference index，无则为 null" in content
    assert len(content) < 900


def test_correction_risk_prompt_defines_boolean_gate_and_query_only_boundary() -> None:
    prompt, values = correction_risk_prompt("这次讨论的芯片型号和制程节点是什么？")
    system_content = prompt.invoke(values).to_messages()[0].content
    assert isinstance(system_content, str)

    assert "has_risk=true" in system_content
    assert "has_risk=false" in system_content
    assert "不再区分专名和数字的具体类型" in system_content
    assert "阿拉伯数字" in system_content
    assert "中文数字" in system_content
    assert "口语化数值表达" in system_content
    assert "不按字符形式判断" in system_content
    assert "不判断 ASR 是否真的出错" in system_content
    assert "只依据 query" in system_content


def test_adjudication_agent_prompt_explains_background_state_and_action_boundaries() -> None:
    recording_id = uuid4()
    evidence = Evidence(
        index=1,
        recording=EvidenceRecording(id=recording_id, title="测试", file_name="test.mp3"),
        chunk=EvidenceChunk(id=uuid4(), text="运行时证据", start_ms=0, end_ms=1_000),
        score=1,
        match_type="vector",
        url=f"/recordings/{recording_id}",
    )
    prompt, values = adjudication_agent_prompt(
        "问题",
        True,
        AnswerPlan(items=[AnswerPlanItem(statement="计划", evidence_indexes=[1])]),
        evidence,
        [],
        EvidenceAdjudicationCaseState(evidence_index=1, chunk_id=evidence.chunk.id),
        3,
        3,
    )
    system_content = prompt.invoke(values).to_messages()[0].content
    assert isinstance(system_content, str)

    assert "全部活跃 Candidate" in system_content
    assert "不遗漏任何 proposal_id" in system_content
    assert "accept" in system_content
    assert "web_search" in system_content
    assert "reconstruct" in system_content
    assert "reject" in system_content
    assert "一经采用不再搜索或重建" in system_content
    assert "后端使用 Candidate.search_query" in system_content
    assert "reconstruct_focus 必须给出具体反馈" in system_content
    assert "reject：当前候选不成立" in system_content
    assert "后端记录 reason" in system_content
    assert "可用 reconstruct_focus 补充重建方向" in system_content
    assert "仅按 schema 输出 decisions" in system_content
    case_payload = json.loads(values["case"])
    assert set(case_payload) == {
        "expression_audit",
        "proposals",
        "findings",
        "iteration",
        "search_count",
        "attempted_queries",
        "accepted_proposal_ids",
        "rejected_proposal_ids",
        "decision_history",
    }


def test_gemini_grounded_search_extracts_summary_and_deduplicated_sources() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-goog-api-key"] == "test-key"
        assert request.url.path.endswith("/models/gemini-test:generateContent")
        payload = json.loads(request.content)
        assert payload["tools"] == [{"google_search": {}}]
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "官方资料支持该候选。"}]},
                        "groundingMetadata": {
                            "groundingChunks": [
                                {"web": {"title": "Official", "uri": "https://example.com/spec"}},
                                {"web": {"title": "Duplicate", "uri": "https://example.com/spec"}},
                            ]
                        },
                    }
                ]
            },
        )

    client = GeminiGroundedSearchClient(
        api_key="test-key",
        model="gemini-test",
        transport=httpx.MockTransport(handler),
    )

    async def scenario() -> None:
        finding = await client.search("proposal-1", "  verify   candidate  ")
        assert finding.query == "verify candidate"
        assert finding.summary == "官方资料支持该候选。"
        assert [source.url for source in finding.sources] == ["https://example.com/spec"]
        await client.close()

    asyncio.run(scenario())


def test_gemini_grounded_search_retries_429_three_attempts_with_ten_second_intervals(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="rag")
    request_count = 0
    sleep_delays: list[float] = []

    async def retry_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count < 3:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "重试成功。"}]}}]},
        )

    client = GeminiGroundedSearchClient(
        api_key="test-key",
        model="gemini-test",
        transport=httpx.MockTransport(handler),
        retry_sleep=retry_sleep,
    )

    async def scenario() -> None:
        finding = await client.search("proposal-1", "verify candidate")
        assert finding.summary == "重试成功。"
        await client.close()

    asyncio.run(scenario())

    assert request_count == 3
    assert sleep_delays == [10.0, 10.0]
    assert "attempt=1/3 retry_in_seconds=10" in caplog.text
    assert "attempt=2/3 retry_in_seconds=10" in caplog.text


def test_chrome_ai_overview_reuses_background_tab_and_waits_for_stable_streamed_text() -> None:
    navigated_urls: list[str] = []
    sleep_delays: list[float] = []
    snapshots = [
        ChromeAiOverviewSnapshot(text="", sources=()),
        ChromeAiOverviewSnapshot(text="A 5-second latency", sources=()),
        ChromeAiOverviewSnapshot(text="A 5-second latency can happen due to buffering.", sources=()),
        ChromeAiOverviewSnapshot(text="A 5-second latency can happen due to buffering.", sources=()),
        ChromeAiOverviewSnapshot(
            text="A 5-second latency can happen due to buffering.",
            sources=(GroundedSource(title="Source", url="https://example.com/rf"),),
        ),
    ]

    class Automation:
        async def navigate(self, url: str) -> None:
            navigated_urls.append(url)

        async def snapshot(self) -> ChromeAiOverviewSnapshot:
            return snapshots.pop(0)

    async def poll_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    client = ChromeAiOverviewSearchClient(
        timeout_seconds=10,
        poll_interval_seconds=0.25,
        automation=Automation(),
        poll_sleep=poll_sleep,
    )

    finding = asyncio.run(client.search("proposal-1", "  RF maximum latency 5 seconds  "))

    assert len(navigated_urls) == 1
    assert "q=RF+maximum+latency+5+seconds" in navigated_urls[0]
    assert "hl=en" in navigated_urls[0]
    assert "gl=us" in navigated_urls[0]
    assert "ars_aio_bridge=1" in navigated_urls[0]
    assert finding.summary == "A 5-second latency can happen due to buffering."
    assert [source.url for source in finding.sources] == ["https://example.com/rf"]
    assert sleep_delays == [0.25, 0.25, 0.25, 0.25]
