from __future__ import annotations

import json
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate

from l2_core.rag.adjudication.contracts import EvidenceAdjudicationCaseState, ExpressionAudit
from l2_core.rag.contracts import AnswerPlan, Evidence

AuditPromptVariant = Literal["relation_rules", "free_discovery"]

_RELATION_RULES_AUDIT_PROMPT = (
    "对半导体会议 ASR 的 Target 做高召回疑点审计：圈出所有值得重建或核验的原文，不提出候选，不输出 supported 表达。"
    "Target 是唯一审计对象；Reference 可能错误、换题或无关，只能辅助判断 Target，不审计其自身问题。\n"

    "先独立理解 Target 的整体技术叙述及每个关键表达的语义角色，包括对象或术语、属性或动作、关系、条件、参数和因果。"
    "这些类别具有相同的审计优先级，不得预设哪一类更可信或更可能出错。\n"

    "以完整语义关系为单位审计，并逐项检查：表达承担的角色；其通常技术身份是否适合该角色；"
    "它与关系中其他表达及前后文是否自然、一致；接受它是否需要原文未给出的关键假设、改变通常含义或忽略异常。"
    "表达真实存在、字面或语法成立、属于当前领域或重复出现，都不能单独证明它正确；"
    "ASR 可能稳定重复同一种合法但角色错误的表达。\n"

    "发现异常后，不得把同一关系中的其他表达视为已确认事实；应暂时把已发现的异常视为未知，重新审计其余关键表达。"
    "若异常可能来自多个表达，应标出所有具有独立嫌疑者，不得只选择最显眼或最容易解释的一项。"
    "可在内部比较解释以定位疑点，但不得输出替换方向。"
    "reason、semantic_role 等任何字段均不得出现具体替换词、缩写、名称、数值或示例候选；"
    "只描述原表达的角色及可疑性。\n"

    "item 的 expression 和 context_quote 必须逐字存在于 Target；quote 是包含 expression 的最短判断上下文，"
    "其全部匹配位置均属该疑点，不输出位置或序号。"
    "expression 按独立疑点拆分；同一可疑角色可单独成 item，无法归因到单个表达时才圈出最小关系片段。"
    "独立疑点分别输出，不可分割的才合并；不得遗漏或无故扩大范围。\n"

    "填写 semantic_role、supporting_evidence_index 和 reason。"
    "reason 先说明表达在 Target 中的语义角色及其与上下文的具体不一致；"
    "Reference 仅作辅助，reason 不得评价或修正 Reference。"
    "reason 不需要确定正确答案，也不得提出替换文本。"

    "supporting_evidence_index 填最相关的 Reference index，无则为 null。"
    "Query 和 Plan 不是证据。没有疑点时输出空 items。仅按 schema 输出。"
)

_FREE_DISCOVERY_AUDIT_PROMPT = (
    "你负责审计半导体会议录音的 ASR Target，找出所有可能被错误识别、且会影响技术含义的词、缩写或短语。"
    "只标记疑点，不输出 supported 表达，不提出修正答案。\n"

    "先独立阅读完整 Target，自由判断哪些原文放在当前上下文中不自然或不一致。"
    "不要只判断一个词是否真实存在或属于当前领域，还要判断它是否适合承担上下文赋予它的技术角色，"
    "以及它与前后文的动作、属性、约束和因果是否相容。"
    "一个表达即使单独合理、语法成立或多次出现，在当前上下文中仍可能是 ASR 错误。\n"

    "文本中可能同时存在多个独立错误。发现一个明显问题后，仍要继续检查其他关键表达，"
    "不得因为已经找到一个高置信疑点而停止。"
    "可在内部比较解释以判断疑点，但不得输出替换方向。"
    "reason、semantic_role 等任何字段均不得出现具体替换词、缩写、名称、数值或示例候选；"
    "只描述原表达的角色及可疑性。\n"

    "每个独立疑点输出一个 item。expression 必须是从 Target 逐字复制的最小可疑片段。"
    "context_quote 必须是从 Target 逐字复制、包含 expression 且足以体现疑点的最短上下文；"
    "它在 Target 中的全部匹配位置都属于该 item。"
    "同一 expression 在不同上下文中承担不同角色时，分别输出 item。"
    "不要输出 offset、出现次数或位置序号。\n"

    "reason 简要说明 expression 在上下文中承担什么角色，以及为什么值得核验。"
    "Reference 只能辅助判断疑点，不得限制 Target 中疑点的发现，也不得审计 Reference 自身。"
    "supporting_evidence_index 填最相关的 Reference index，无则为 null。"
    "Query 和 Plan 不是证据。没有疑点时输出空 items。仅按 schema 输出。"
)


def _expression_audit_system_prompt(variant: AuditPromptVariant) -> str:
    if variant == "free_discovery":
        return _FREE_DISCOVERY_AUDIT_PROMPT
    return _RELATION_RULES_AUDIT_PROMPT


def correction_risk_prompt(query: str) -> tuple[ChatPromptTemplate, dict[str, str]]:
    return (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "判断用户问题的答案是否依赖录音中的精确表达。"
                    "如果某个词、名称、标识、数值或其他细节一旦识别错误，就可能改变答案含义或结论，则 has_risk=true。"
                    "如果问题只需要概括性、原理性或定性回答，不依赖这些精确细节，则 has_risk=false。"
                    "仅按 schema 输出。"
                ),
                ("human", "{query}"),
            ]
        ),
        {"query": query},
    )


def evidence_review_prompt(
    query: str,
    plan: AnswerPlan,
    evidence: Evidence,
    *,
    reference_evidence: list[Evidence] | None = None,
    expression_audit: ExpressionAudit,
    findings: str = "[]",
    focus: str = "",
) -> tuple[ChatPromptTemplate, dict[str, str]]:
    evidence_payload: dict[str, str | int] = {
        "evidence_index": evidence.index,
        "chunk_id": str(evidence.chunk.id),
        "recording_id": str(evidence.recording.id),
        "start_ms": evidence.chunk.start_ms,
        "end_ms": evidence.chunk.end_ms,
        "text": evidence.chunk.text,
    }
    reference_payload: list[dict[str, str | int]] = [
        {
            "evidence_index": item.index,
            "chunk_id": str(item.chunk.id),
            "recording_id": str(item.recording.id),
            "start_ms": item.chunk.start_ms,
            "end_ms": item.chunk.end_ms,
            "text": item.chunk.text,
        }
        for item in reference_evidence or []
    ]
    expression_audit_payload = {
        "items": [item.model_dump(mode="json", exclude={"reason"}) for item in expression_audit.items]
    }
    return (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "任务：把 Expression Audit 中的表达重建为 ASR 修正候选。\n"
                    "一、证据边界\n"
                    "- Target Evidence 是唯一修改目标；Target 原文是待审查 ASR，不是事实，不能因为它能被讲成一个合理故事就保留原表达。\n"
                    "- Reference Evidence 只提供术语和主题背景，同一录音也可能换题，不得强行关联。模型知识只能提出候选和 search_query，不能作为最终证据。\n"
                    "- Audit 只定义疑点和范围，不提供候选。忽略其自由文本中意外出现的替换方向；"
                    "必须独立根据 Target、Reference 和 Findings 推导并代回验证。独立得到相同结果可用。\n"
                    "二、逐项重建\n"
                    "1. 每个 proposal 用 audit_item_id 绑定真实 Audit item；original_expression 必须等于该 item 的 expression，"
                    "并绑定 Target 的 evidence_index 和 chunk_id。\n"
                    "2. proposed_expression 去除首尾空白后必须与 original_expression 不同，禁止用原表达作为候选。候选与原表达不要求同音、近音或字形相似。\n"
                    "3. 把候选代入该 item 的 context_quote 及其所有匹配位置，检查技术对象、信号、动作、属性、数字、单位、条件和因果链是否更自洽、"
                    "是否减少额外假设、是否引入新矛盾；局部修正后，必须重新检查同一因果链。\n"
                    "4. 若候选仍依赖非字面解释、补充原文不存在的隐含术语或建立不自然关系，应检查相关 Audit item，并生成相互一致的关联候选。\n"
                    "三、生成职责\n"
                    "- 你没有否定或省略 Audit items 的权限。必须逐项覆盖 items 里的每一项，每一项输出1至2个 proposal；"
                    "依据不足时也要给出最可能且可检验的候选，由后续裁决 Agent 决定，禁止静默省略。\n"
                    "- 禁止用 no-op proposal 表示保留原文。\n"
                    "四、输出规则\n"
                    "- 不得为 Expression Audit 中不存在的表达编造 audit_item_id，不得修改 Reference 表达。\n"
                    "- Reference 明确支持候选时填写最强 supporting_evidence_index，否则为 null。后续结合 Findings 重建、收敛或删除候选。仅按 schema 输出。",
                ),
                (
                    "human",
                    "问题：{query}\n相关计划：{plan}\nExpression Audit：{expression_audit}\n"
                    "必须逐项覆盖的 audit_item_id：{required_audit_item_ids}\n"
                    "Target Evidence：{evidence}\nReference Evidence：{reference_evidence}\n"
                    "联网 Findings：{findings}\n本轮关注点：{focus}",
                ),
            ]
        ),
        {
            "query": query,
            "plan": plan.model_dump_json(),
            "evidence": json.dumps(evidence_payload, ensure_ascii=False, separators=(",", ":")),
            "reference_evidence": json.dumps(reference_payload, ensure_ascii=False, separators=(",", ":")),
            "expression_audit": json.dumps(expression_audit_payload, ensure_ascii=False, separators=(",", ":")),
            "required_audit_item_ids": json.dumps(
                [item.id for item in expression_audit.items],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "findings": findings,
            "focus": focus,
        },
    )


def expression_audit_prompt(
    query: str,
    plan: AnswerPlan,
    evidence: Evidence,
    reference_evidence: list[Evidence],
    *,
    focus: str = "",
    variant: AuditPromptVariant = "relation_rules",
) -> tuple[ChatPromptTemplate, dict[str, str]]:
    target: dict[str, str | int] = {
        "evidence_index": evidence.index,
        "chunk_id": str(evidence.chunk.id),
        "recording_id": str(evidence.recording.id),
        "start_ms": evidence.chunk.start_ms,
        "end_ms": evidence.chunk.end_ms,
        "text": evidence.chunk.text,
    }
    references: list[dict[str, str | int]] = [
        {
            "evidence_index": item.index,
            "chunk_id": str(item.chunk.id),
            "start_ms": item.chunk.start_ms,
            "end_ms": item.chunk.end_ms,
            "text": item.chunk.text,
        }
        for item in reference_evidence
    ]
    return (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    _expression_audit_system_prompt(variant),
                ),
                (
                    "human",
                    "问题：{query}\n相关计划：{plan}\nReference Evidence（仅辅助判断，不审计）：{references}\n"
                    "Target Evidence（唯一审计对象）：{target}\n关注点：{focus}",
                ),
            ]
        ),
        {
            "query": query,
            "plan": plan.model_dump_json(),
            "target": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
            "references": json.dumps(references, ensure_ascii=False, separators=(",", ":")),
            "focus": focus,
        },
    )


def adjudication_agent_prompt(
    query: str,
    risk: bool,
    plan: AnswerPlan,
    evidence: Evidence,
    reference_evidence: list[Evidence],
    case: EvidenceAdjudicationCaseState,
    max_iterations: int,
    max_searches: int,
    *,
    control: dict[str, object] | None = None,
) -> tuple[ChatPromptTemplate, dict[str, str]]:
    return (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是半导体会议录音 ASR 的 Candidate 裁决 Agent。每轮必须对当前 Case 中全部活跃 Candidate 分别选择且只选择一个动作。"
                    "Candidate 是待验证假设；你只做决策，不直接修改表达，也不遗漏任何 proposal_id。\n"
                    "对每个 Candidate，将 proposed_expression 代入其 context_quote 和 Target 全文，比较替换前后的技术角色、关系、条件、因果一致性及额外假设。"
                    "同时检查匹配的 Finding 与 Reference 是否真正支持完整含义，而非只匹配局部关键词。"
                    "为每项输出 candidate_score，表示它相对同一 audit_item 其他候选更可能是正确修正的程度；不同候选必须独立判断。\n"
                    "动作规则："
                    "accept：当前上下文或证据已足以采用，替换明显改善原异常且不引入关键冲突；一经采用不再搜索或重建。"
                    "web_search：候选合理但缺少决定性外部证据，且步骤控制显示仍有搜索预算；后端使用 Candidate.search_query。"
                    "reconstruct：当前候选不成立，但对应 Audit 疑点仍需要新的候选；reconstruct_focus 必须给出具体反馈。"
                    "reject：当前候选不成立；后端记录 reason，且仅当同组没有 accept/web_search 时参与重建；可用 reconstruct_focus 补充重建方向。"
                    "搜索失败或 Reference 缺失不等于原表达正确；Candidate 能单独讲通也不等于应当 accept。"
                    "同一 audit_item 可能有多个候选：后端按组执行 accept 优先、其次 web_search、最后 reconstruct/reject；"
                    "若多个候选 accept，只采用 candidate_score 最高者；没有 accept 时会执行组内全部 web_search。"
                    "有匹配 Finding 时必须据此更新决策。仅按 schema 输出 decisions。",
                ),
                (
                    "human",
                    "问题：{query}\n风险：{risk}\n相关计划：{plan}\nTarget Evidence：{evidence}\nReference Evidence：{reference_evidence}\n"
                    "当前 Case：{case}\n步骤控制：{control}\n最大轮数：{max_iterations}\n最大搜索次数：{max_searches}",
                ),
            ]
        ),
        {
            "query": query,
            "risk": str(risk).lower(),
            "plan": plan.model_dump_json(),
            "evidence": json.dumps(
                {
                    "evidence_index": evidence.index,
                    "chunk_id": str(evidence.chunk.id),
                    "text": evidence.chunk.text,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "reference_evidence": json.dumps(
                [
                    {
                        "evidence_index": item.index,
                        "chunk_id": str(item.chunk.id),
                        "text": item.chunk.text,
                    }
                    for item in reference_evidence
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "case": json.dumps(
                case.model_dump(
                    mode="json",
                    include={
                        "expression_audit",
                        "proposals",
                        "findings",
                        "iteration",
                        "search_count",
                        "attempted_queries",
                        "accepted_proposal_ids",
                        "rejected_proposal_ids",
                        "decision_history",
                    },
                ),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "control": json.dumps(control or {}, ensure_ascii=False, separators=(",", ":")),
            "max_iterations": str(max_iterations),
            "max_searches": str(max_searches),
        },
    )
