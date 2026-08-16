from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _string_list() -> list[str]:
    return []


def _proposal_list() -> list[AdjudicationProposal]:
    return []


def _candidate_decision_batch_list() -> list[CandidateDecisionBatch]:
    return []


def _audit_item_list() -> list[ExpressionAuditItem]:
    return []


def _span_list() -> list[ExpressionTargetSpan]:
    return []


def _source_list() -> list[GroundedSource]:
    return []


def _finding_list() -> list[GroundedResearchFinding]:
    return []


def _overlay_list() -> list[EvidenceOverlay]:
    return []


def _confirmation_candidate_list() -> list[AdjudicationConfirmationCandidate]:
    return []


def _confirmation_item_list() -> list[AdjudicationConfirmationItem]:
    return []


def _item_decision_list() -> list[ClaimConfirmationItemDecision]:
    return []


def _case_list() -> list[EvidenceAdjudicationCaseState]:
    return []


class CorrectionRiskAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    has_risk: bool = Field(description="Query 是否涉及可能被 ASR 错转的数字、单位、专名、缩写或技术表达")


class ExpressionAuditItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="本次审计中唯一且稳定的条目标识，例如 audit-1", min_length=1, max_length=80)
    expression: str = Field(description="从 Target Evidence 原文逐字复制的待审计表达，不得改写", min_length=1, max_length=300)
    context_quote: str = Field(
        description="从 Target 原文逐字复制、用于定位并判断 expression 的上下文；该 quote 的全部匹配位置都属于该疑点",
        min_length=1,
        max_length=1_000,
    )
    semantic_role: str = Field(
        default="",
        description="该表达在当前技术语境中的作用",
        max_length=500,
    )
    supporting_evidence_index: int | None = Field(
        default=None,
        description="与疑点最相关的 Reference Evidence index；没有相关独立证据时为 null，不得填写 Target 自身的 index",
        ge=1,
    )
    reason: str = Field(
        default="",
        description=(
            "该状态的依据；明确说明数值/单位异常、术语角色不匹配、关系异常、因果异常或证据冲突及其具体关系和 Reference index，"
            "并区分 Target 原文、Reference 证据和待核验推断"
        ),
        max_length=500,
    )

    @model_validator(mode="before")
    @classmethod
    def discard_removed_legacy_fields(cls, value: object) -> object:
        """Allow persisted pre-change audit items to be restored from checkpoints."""

        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        normalized.pop("status", None)
        normalized.pop("occurrence_scope", None)
        return normalized


class ExpressionAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ExpressionAuditItem] = Field(
        default_factory=_audit_item_list,
        description="Target Evidence 中所有影响技术含义的疑点；没有疑点时为空，不得输出 supported 表达",
        max_length=20,
    )


class AdjudicationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="候选在当前 Case 中的唯一标识，例如 proposal-1", min_length=1, max_length=80)
    audit_item_id: str = Field(description="该候选所修正的 ExpressionAuditItem.id", min_length=1, max_length=80)
    evidence_index: int = Field(description="候选所属 Target Evidence 的 index", ge=1)
    chunk_id: str = Field(description="候选所属 Target Evidence 的 chunk_id", min_length=1)
    original_expression: str = Field(description="从 Target 原文逐字复制的待替换表达", min_length=1, max_length=300)
    proposed_expression: str = Field(description="建议替换为的技术表达，必须与 original_expression 不同", min_length=1, max_length=300)
    expression_type: Literal["number", "proper_noun", "compound"] = Field(
        description="修正类型：number 为数字或单位，proper_noun 为专名或术语，compound 为复合表达"
    )
    search_query: str = Field(description="用于公开资料核验该候选的精确搜索词", min_length=1, max_length=240)
    reason: str = Field(default="", description="候选代回全文后比原表达更自洽的理由及现有 Evidence 依据", max_length=500)
    supporting_evidence_index: int | None = Field(
        default=None,
        description="最能独立支持该候选的 Reference Evidence index；没有独立支持时为 null，不得填写 Target 自身的 index",
        ge=1,
    )

    @model_validator(mode="after")
    def validate_change(self) -> AdjudicationProposal:
        if self.original_expression.strip() == self.proposed_expression.strip():
            raise ValueError("adjudication proposal must change the expression")
        return self


class AdjudicationReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposals: list[AdjudicationProposal] = Field(default_factory=_proposal_list, description="本轮生成或保留的 ASR 修正候选", max_length=15)


class CandidateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(description="本轮待裁决的 AdjudicationProposal.id", min_length=1, max_length=80)
    action: Literal["accept", "web_search", "reconstruct", "reject"] = Field(
        description="对该候选的本轮动作：采用、联网核验、定向重建，或拒绝当前候选并重建"
    )
    confidence: float = Field(description="当前决策置信度，0 表示完全不可信，1 表示完全确定", ge=0, le=1)
    candidate_score: float = Field(
        description="候选本身作为该 Audit item 最佳修正的相对评分；用于同组候选排序，0 最差、1 最佳",
        ge=0,
        le=1,
    )
    reason: str = Field(
        description="选择该动作的上下文、证据和一致性依据；action=reject 时同时作为下轮重建反馈",
        min_length=1,
        max_length=500,
    )
    reconstruct_focus: str = Field(
        description="reconstruct 必填、reject 可选的候选重建反馈；accept 和 web_search 必须为空字符串",
        max_length=500,
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_candidate_score(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        if "candidate_score" in value:
            return cast(object, value)
        payload = dict(cast(Mapping[str, object], value))
        payload["candidate_score"] = payload.get("confidence", 0.0)
        return payload

    @model_validator(mode="after")
    def validate_reconstruct_focus(self) -> CandidateDecision:
        if self.action == "reconstruct" and not self.reconstruct_focus.strip():
            raise ValueError("reconstruct action requires reconstruct_focus")
        if self.action in {"accept", "web_search"} and self.reconstruct_focus:
            raise ValueError("reconstruct_focus is only allowed for reconstruct or reject action")
        return self


class CandidateDecisionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[CandidateDecision] = Field(
        description="当前全部活跃 Candidate 的逐项决策；每个 proposal_id 必须恰好出现一次",
        min_length=1,
        max_length=15,
    )


class GroundedSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="", description="公开资料来源的页面标题")
    url: str = Field(description="可访问且直接支持研究摘要的来源 URL", min_length=1)


class GroundedResearchFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(description="本次联网研究所核验的 proposal_id 或 audit item id")
    query: str = Field(description="实际执行的联网搜索词")
    summary: str = Field(description="基于来源提炼的核验结果，不得超出来源内容")
    sources: list[GroundedSource] = Field(default_factory=_source_list, description="支持 summary 的公开资料来源", max_length=8)


class ExpressionTargetSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start_char: int = Field(description="expression 在 Target chunk 文本中的起始字符下标，包含该字符", ge=0)
    end_char: int = Field(description="expression 在 Target chunk 文本中的结束字符下标，不包含该字符", gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> ExpressionTargetSpan:
        if self.end_char <= self.start_char:
            raise ValueError("expression target span must have positive length")
        return self


class EvidenceOverlay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(description="产生该覆盖结果的 AdjudicationProposal.id")
    evidence_index: int = Field(description="被覆盖的 Target Evidence index", ge=1)
    chunk_id: str = Field(description="被覆盖的 Target Evidence chunk_id")
    original_expression: str = Field(description="Target Evidence 中被替换的原文表达")
    resolved_expression: str = Field(description="自动裁决或用户确认后的最终表达")
    target_spans: list[ExpressionTargetSpan] = Field(
        default_factory=_span_list,
        description="后端从 Audit context_quote 解析出的精确替换位置，end_char 为开区间",
    )
    status: Literal["auto_resolved", "user_confirmed"] = Field(description="覆盖结果来自自动裁决还是用户确认")
    confidence: float = Field(description="最终修正的置信度，范围为 0 到 1", ge=0, le=1)
    source_urls: list[str] = Field(default_factory=_string_list, description="支持最终修正的公开资料 URL")


class AdjudicationConfirmationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="供用户选择的候选唯一标识，通常对应 proposal_id")
    expression: str = Field(description="供用户确认的候选修正表达")
    confidence: float = Field(description="候选正确的模型置信度，范围为 0 到 1", ge=0, le=1)
    source_urls: list[str] = Field(default_factory=_string_list, description="支持该候选的公开资料 URL")


class AdjudicationConfirmationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="确认项的唯一标识")
    evidence_index: int = Field(description="待确认表达所属 Evidence index", ge=1)
    recording_id: UUID = Field(description="待确认表达所属录音 ID")
    chunk_id: UUID = Field(description="待确认表达所属证据分块 ID")
    start_ms: int = Field(description="证据分块在录音中的开始时间，单位毫秒", ge=0)
    end_ms: int = Field(description="证据分块在录音中的结束时间，单位毫秒", ge=0)
    original_expression: str = Field(description="需要用户确认是否修正的 Target 原文表达")
    target_spans: list[ExpressionTargetSpan] = Field(
        default_factory=_span_list,
        description="后端从 Audit context_quote 解析出的精确目标位置，end_char 为开区间",
    )
    candidates: list[AdjudicationConfirmationCandidate] = Field(
        default_factory=_confirmation_candidate_list,
        description="可供用户选择的修正候选",
    )
    reason: str = Field(default="", description="请求用户确认的原因及当前证据情况")


class AdjudicationConfirmationBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["adjudication_confirmation"] = Field(default="adjudication_confirmation", description="前端确认块的固定类型标识")
    request_id: UUID = Field(description="本次用户确认请求的唯一 ID")
    source_generation_id: UUID = Field(description="产生该确认请求的 generation run ID")
    items: list[AdjudicationConfirmationItem] = Field(
        default_factory=_confirmation_item_list,
        description="本次需要用户处理的确认项",
        min_length=1,
    )


class ClaimConfirmationItemDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(description="对应的 AdjudicationConfirmationItem.id")
    action: Literal["accept_candidate", "keep_original", "unresolved"] = Field(
        description="用户决定：接受候选、保留原文或暂不解决"
    )
    candidate_id: str | None = Field(default=None, description="action=accept_candidate 时选中的候选 ID；其他操作必须为 null")

    @model_validator(mode="after")
    def validate_candidate(self) -> ClaimConfirmationItemDecision:
        if (self.action == "accept_candidate") != (self.candidate_id is not None):
            raise ValueError("candidate_id is required only for accept_candidate")
        return self


class ClaimConfirmationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID = Field(description="所响应的 AdjudicationConfirmationBlock.request_id")
    client_request_id: UUID = Field(description="客户端为本次提交生成的幂等请求 ID")
    decisions: list[ClaimConfirmationItemDecision] = Field(
        default_factory=_item_decision_list,
        description="用户对各确认项作出的决定",
        min_length=1,
    )


class EvidenceAdjudicationCaseState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_index: int = Field(description="当前 Case 所审查的 Target Evidence index", ge=1)
    chunk_id: UUID = Field(description="当前 Case 所审查的 Target Evidence chunk_id")
    expression_audit: ExpressionAudit | None = Field(default=None, description="第一步得到的 Target 原文表达审计结果")
    proposals: list[AdjudicationProposal] = Field(
        default_factory=_proposal_list,
        description="当前仍待核验或裁决的 ASR 修正候选",
        max_length=15,
    )
    initial_reconstruction_completed: bool = Field(default=False, description="当前 Case 是否已完成首次候选重建")
    pending_setup_phase: Literal["audit", "initial_reconstruct"] | None = Field(
        default=None,
        description="等待执行的固定前置阶段",
    )
    pending_decisions: CandidateDecisionBatch | None = Field(
        default=None,
        description="模型已输出、等待后端批量执行的当前 Candidate 决策",
    )
    decision_history: list[CandidateDecisionBatch] = Field(
        default_factory=_candidate_decision_batch_list,
        description="当前 Case 已执行的批量 Candidate 决策",
    )
    accepted_proposal_ids: list[str] = Field(
        default_factory=_string_list,
        description="当前 Case 已采用并转换为 EvidenceOverlay 的 proposal_id",
    )
    rejected_proposal_ids: list[str] = Field(
        default_factory=_string_list,
        description="当前 Case 已明确拒绝的 proposal_id",
    )
    findings: list[GroundedResearchFinding] = Field(default_factory=_finding_list, description="当前 Case 已获得的联网核验结果")
    attempted_queries: list[str] = Field(default_factory=_string_list, description="当前 Case 已执行过的标准化搜索词，用于阻止重复搜索")
    iteration: int = Field(default=0, description="当前 Case 已完成的 Candidate 决策轮数", ge=0)
    search_count: int = Field(default=0, description="当前 Case 已执行的联网搜索次数", ge=0)
    status: Literal["pending", "researching", "resolved", "needs_confirmation", "rejected"] = Field(
        default="pending",
        description="Case 状态：待处理、研究中、已解决、需用户确认或已拒绝修正",
    )
    error: str | None = Field(default=None, description="最近一次阶段或 Candidate 操作失败，以及 Case 终止的原因")


class AdjudicationAgentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk: bool = Field(description="Query 是否触发 ASR 表达裁决流程")
    cases: list[EvidenceAdjudicationCaseState] = Field(default_factory=_case_list, description="按 Evidence 分离的裁决 Case 列表")
    current_case: int = Field(default=0, description="当前正在处理的 cases 下标，从 0 开始", ge=0)
    overlays: list[EvidenceOverlay] = Field(default_factory=_overlay_list, description="已确定并可应用到最终答案证据上的表达修正")
    pending_confirmation: AdjudicationConfirmationBlock | None = Field(default=None, description="等待用户处理的确认请求；没有时为 null")
    applied_user_decision: ClaimConfirmationDecision | None = Field(default=None, description="已应用到当前状态的用户确认决定")
    web_search_enabled: bool = Field(default=False, description="当前裁决运行是否允许调用公开资料搜索")
    status: Literal["running", "completed"] = Field(default="running", description="整个裁决 Agent 仍在运行或已经结束")
