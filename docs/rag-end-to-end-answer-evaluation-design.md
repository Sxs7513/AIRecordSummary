# RAG 端到端答案评测技术方案

## 1. 文档目标

本文档定义现有 RAG Retrieval 评测向端到端答案评测扩展的第一版方案。

第一版在同一个 Dataset、Dataset Version、Corpus Snapshot、Pipeline Version 和 Evaluation Run 中串联：

```text
Route
  -> Retrieval
  -> Context Expansion
  -> Rerank
  -> Grade
  -> Answer
  -> Answer Evaluation
```

一次 Run 同时保存并展示检索指标和答案指标，使失败 Case 可以从最终答案下钻到检索与生成阶段，而不是建设两套互相割裂的评测平台。

第一版重点回答以下问题：

- Gold Evidence 是否被召回并排在足够靠前的位置；
- 系统是否在应该回答时回答、在无答案时拒答；
- 最终答案是否覆盖人工标注的关键事实；
- 答案中的事实是否正确，是否包含无依据或矛盾陈述；
- 答案引用的录音证据是否支持对应 Claim。

## 2. 核心决策

### 2.1 Retrieval 与 Answer 使用同一条评测链路

现有 Retrieval 评测已经复用生产 Graph 的 Route、Retrieval、Rerank 和 Grade。第一版答案评测在 Grade 后继续执行生产 Answer，不新建独立 Dataset、Run 或前端工作区。

同一 Case Result 同时包含：

```text
Case Result
├── Route 结果
├── 各检索节点排名与指标
├── Grade verdict
├── Generated Answer
├── Answer Sources
└── Answer Evaluation Result
```

这样可以区分以下失败来源：

- Route 选择错误；
- 基础召回未命中；
- RRF 或 Rerank 丢失 Gold Evidence；
- Grade 错误放行或错误拒答；
- Evidence 正确但 Answer 遗漏、曲解或补充了无依据事实；
- Answer 内容正确但 Citation 不支持对应 Claim。

### 2.2 Route 参与执行，第一版不单独标注

所有 Case 都经过真实 Route，Route 输出继续写入 Case Result 和 Trace。

第一版不增加 `expected_strategy`，不计算 Route Accuracy 或路由混淆矩阵。Route 错误会通过检索失败、答案错误或错误拒答反映在端到端结果中。只有在失败分析显示 Route 已成为主要瓶颈时，再增加 Route 专项标注与指标。

### 2.3 Risk 与 ASR Adjudication 不进入主评测

第一版主评测关闭 ASR Risk Gate 和 Evidence Adjudication Agent，原因包括：

- 该能力只针对特定技术表达触发；
- 它可能包含多轮模型调用、外部搜索和用户确认；
- 会显著增加端到端评测耗时与随机性；
- 项目已经有独立的 `rag_adjudication` 评测链路。

主评测负责检索、拒答、答案和引用质量；现有 Adjudication Evaluation 继续负责纠错 Precision、Recall、F1 及相关阶段指标。第一版不建设完整生产链路 Smoke Test。

### 2.4 Answer Judge 属于评测层

Answer Judge 不加入生产 `RagGraph`，也不改变生产答案。

生产 Graph 完成后，`rag_evaluation_worker` 将 Query、冻结的 Answer Annotation、Generated Answer 和实际引用证据交给评测层的 `AnswerEvaluator`。Judge 失败只影响答案评测步骤，不得改变生产 Graph 的答案或 Sources。

第一版使用一次在线模型的结构化 LLM 调用，同时评估答案正确性、关键事实覆盖和引用支持，避免为 Claim 提取、正确性、完整性和 Citation 分别调用模型。Judge Model 使用独立配置并冻结到 Run Config Snapshot，不受本地 Grade 模型限制。

### 2.5 标准答案采用结构化 Answer Annotation

自然语言问题通常存在多种正确措辞，因此第一版不把单一标准答案字符串作为唯一 Ground Truth。

每个 Case 保存一个 `answer_annotation`，由以下内容组成：

- `expected_verdict`：预期回答姿态；
- `reference_answer`：人工认可的一种参考回答；
- `key_points`：最终答案应覆盖的关键事实；
- `evidence_ids`：支持每条 Key Point 的冻结 Gold Evidence。

`reference_answer` 用于帮助标注人员和 Judge 理解答案整体语义，稳定 Ground Truth 主要由 `key_points + evidence_ids` 构成。

## 3. 第一版范围

### 3.1 包含

- 在现有 RAG Case 中录入和冻结 Answer Annotation；
- 根据 Query 和 Gold Evidence 生成 Answer Annotation 草稿；
- 人工审核、编辑并确认模型生成的标注草稿；
- 在现有 Retrieval 评测后继续运行 Grade 和 Answer；
- 保存最终 Answer、Sources、实际 Verdict 和 Route Strategy；
- 使用一次结构化 LLM Judge 评估答案和 Citation；
- 对无答案 Case 计算确定性的拒答结果；
- 聚合 Retrieval 与 Answer 指标；
- 在 Run 和 Case 页面展示答案结果及失败原因；
- 将 Answer Annotation 纳入 Dataset Version checksum 和冻结约束。

### 3.2 不包含

- `required_limitations` 或 `required_caveats`；
- `forbidden_facts`；
- 多级 Key Point 权重；
- Route 专项标注和 Route Accuracy；
- Risk Gate 专项评测；
- 将 ASR Adjudication 接入主评测；
- 完整生产链路 Smoke Test；
- 多轮或多模型 Judge；
- 独立 Claim 抽取 LLM 节点；
- 多人复核和复杂审批流；
- 自动发布或自动阻断生产 Pipeline。

## 4. Answer Annotation

### 4.1 数据结构

```python
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class AnswerKeyPoint(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=1_000)
    evidence_ids: list[UUID] = Field(min_length=1)


class AnswerAnnotation(BaseModel):
    expected_verdict: Literal["direct_answer", "qualified_answer", "abstain"]
    reference_answer: str | None = Field(default=None, max_length=10_000)
    key_points: list[AnswerKeyPoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_content(self) -> "AnswerAnnotation":
        if self.expected_verdict == "abstain":
            if self.key_points:
                raise ValueError("abstain annotation cannot contain key points")
            return self
        if not self.key_points:
            raise ValueError("answerable annotation requires at least one key point")
        if not self.reference_answer or not self.reference_answer.strip():
            raise ValueError("answerable annotation requires a reference answer")
        return self
```

第一版约束：

- `direct_answer` 和 `qualified_answer` 至少包含一条 Key Point；
- Answerable Case 必须填写 `reference_answer`；
- `abstain` 不包含 Key Point，`reference_answer` 必须为 `null`；可以保留相关但不足以回答问题的 Evidence，以验证系统在有上下文时仍能正确拒答；
- 每条 Key Point 至少关联一条当前 Case 的 Gold Evidence；
- `evidence_ids` 只能引用同一 Case 中存在的 Evidence；
- Key Point ID 在当前 Case 内唯一；
- Draft Annotation 中的 `evidence_ids` 引用 Draft Evidence ID；冻结时必须映射为对应的 Frozen Evidence ID，不能原样保留 Draft ID；
- 冻结后 Answer Annotation 不可变。

### 4.2 完全可回答示例

```json
{
  "expected_verdict": "direct_answer",
  "reference_answer": "录音中提到 JTAG 的频率为 30 MHz。",
  "key_points": [
    {
      "id": "kp-1",
      "text": "JTAG 的频率为 30 MHz",
      "evidence_ids": ["gold-evidence-1"]
    }
  ]
}
```

### 4.3 部分可回答示例

第一版不要求单独录入限制语句。标注人员将可由证据支持的部分写入 Key Point，并将预期姿态标为 `qualified_answer`：

```json
{
  "expected_verdict": "qualified_answer",
  "reference_answer": "录音提到 JTAG 使用 30 MHz，但无法确认是否已经完成量产验证。",
  "key_points": [
    {
      "id": "kp-1",
      "text": "录音提到 JTAG 使用 30 MHz",
      "evidence_ids": ["gold-evidence-1"]
    }
  ]
}
```

Judge 根据 Query、Reference Answer、Key Points 和 Evidence 判断 Generated Answer 是否把未知信息写成确定事实。第一版不额外维护 `required_limitations`。

### 4.4 无答案示例

```json
{
  "expected_verdict": "abstain",
  "reference_answer": null,
  "key_points": []
}
```

无答案 Case 不需要录入统一拒答文案。评测关注系统是否拒答，而不是拒答措辞是否与参考文本一致。

## 5. 模型辅助标注

### 5.1 目标

现有 Case 已包含 Query、Scope 和人工选择的 Gold Evidence。系统可以使用这些可信输入生成 Answer Annotation 草稿，减少人工从零编写 Reference Answer 和 Key Points 的成本。

### 5.2 流程

```text
Query + Gold Evidence
        ↓
Annotation Draft Generator
        ↓
Suggested Verdict + Reference Answer + Key Points + Evidence Mapping
        ↓
人工审核和编辑
        ↓
保存 Answer Annotation
        ↓
随 Dataset Version 冻结
```

模型生成结果不能自动批准 Case，也不能自动冻结 Dataset Version。

### 5.3 生成约束

- 只能使用给定 Gold Evidence；
- 不得调用外部搜索；
- 不得根据常识补充录音未出现的事实；
- 每条 Key Point 必须关联至少一条输入 Evidence；
- Query 的核心内容完全无法由 Evidence 支持时，建议 `abstain`；
- 只能输出结构化 JSON，并在业务层使用 `AnswerAnnotation` 校验；
- Evidence 引用不存在、跨 Case 或 Key Point 为空时拒绝保存草稿。

### 5.4 API 建议

```text
POST /api/rag-evaluation/cases/{case_id}/answer-annotation:suggest
PUT  /api/rag-evaluation/cases/{case_id}/answer-annotation
```

建议接口只返回草稿，不在 Suggest 请求中写数据库。用户审核后由 PUT 接口保存，避免模型输出被误认为人工 Ground Truth。

## 6. 端到端评测执行流程

### 6.1 单 Case 流程

```text
1. 加载 Frozen Case、Gold Evidence 和 Answer Annotation
2. 执行生产 Route
3. 执行 Retrieval、Context Expansion 和 Rerank
4. 保存各节点 Ranked Results 和 Retrieval Metrics
5. 执行 Grade
6. 执行 Answer
7. 保存 Generated Answer、Sources、Verdict 和 Route Strategy
8. 执行确定性拒答评分
9. 对需要答案评分的 Case 调用一次 Answer Judge
10. 保存 Answer Evaluation Step 和 Case Metrics
11. 聚合 Run Metrics
```

### 6.2 无答案 Case

无答案 Case 首先使用确定性结果计算拒答指标：

```python
correct_abstention = (
    annotation.expected_verdict == "abstain"
    and graph_result.not_enough_evidence is True
)
```

如果生产 Graph 已经明确拒答且没有生成事实性答案，第一版可以跳过 LLM Judge，以减少成本和延迟。

如果无答案 Case 被放行并生成答案，则必须调用 Judge 检查 Unsupported Claims，并将其计入 Unsafe Answer Rate。

### 6.3 Answerable Case

`direct_answer` 和 `qualified_answer` Case 均调用一次 Answer Judge。Judge 输入包含：

- 原始 Query；
- Expected Verdict；
- Reference Answer；
- Key Points；
- Key Point 对应的 Gold Evidence 文本；
- Generated Answer；
- Generated Answer 实际引用的 Evidence 文本与 Source Index；
- 生产 Graph 的实际 Grade Verdict。

### 6.4 Judge 与生产 Graph 隔离

Judge 使用评测层的模型调用适配器，建议新增：

```text
backend/packages/l2_core/rag_evaluation/answer_evaluator.py
```

不要在 `backend/packages/l2_core/rag/graph.py` 增加 Judge Node。评测 Worker 在生产 Graph 返回后调用 `AnswerEvaluator`，并保存一个开放命名的步骤：

```text
evaluate.answer
```

Judge 调用需要进入现有模型调用可观测性，但其 Token 和延迟必须标记为 Evaluation 开销，不能与生产 RAG Token 和延迟混合。

## 7. Answer Judge 契约

### 7.1 输出结构

```python
class KeyPointEvaluation(BaseModel):
    key_point_id: str
    covered: bool
    correct: bool
    reason: str


class ClaimEvaluation(BaseModel):
    claim: str
    factual_status: Literal["supported", "unsupported", "contradicted"]
    citation_status: Literal["supported", "missing", "misaligned"]
    citation_indexes: list[int]
    reason: str


class AnswerEvaluationResult(BaseModel):
    answerability_correct: bool
    key_point_results: list[KeyPointEvaluation]
    claim_results: list[ClaimEvaluation]
```

第一版不要求 Judge 输出一个不可解释的总分。所有 Run 指标由代码根据结构化结果确定性计算。

### 7.2 Judge 规则

- 不要求 Generated Answer 与 Reference Answer 使用相同措辞；
- Key Point 语义被正确表达即记为覆盖；
- Key Point 被提及但数字、单位、人物、否定关系或状态错误时，`covered=true`、`correct=false`；
- 每个重要事实性 Claim 必须且只能出现在一条 `claim_results` 中；
- `factual_status` 互斥表示事实有支持、无支持或与标准事实冲突；
- `citation_status` 独立表示实际引用支持、缺失或错位；
- Citation 必须支持对应 Claim，而不只是与 Query 主题相关；
- Generated Answer 可以引用非 Gold Evidence，只要实际 Evidence 能支持 Claim；
- 没有 Citation 的事实性 Claim 记为 `citation_status=missing`；有引用但不能支持时记为 `misaligned`；
- 无证据支持且无反向证据时记为 `factual_status=unsupported`；
- 与 Gold Evidence 或 Reference Answer 核心事实冲突时记为 `factual_status=contradicted`；
- Judge 只能根据输入 Evidence 判断，不得使用外部知识修正或补全答案。

### 7.3 稳定性要求

- 使用固定 Judge Model Reference；
- Judge 必须使用配置的在线模型，不复用受本地设备限制的 Grade 模型；
- Judge Model Reference、Provider 和 Prompt Version 必须写入 Evaluation Run 的 Config Snapshot；
- temperature 设为 0 或 Provider 支持的最低值；
- 使用严格 JSON Schema 输出和 Pydantic 校验；
- 保存 Judge Model、Prompt Version、原始结构化输出、Token 和延迟；
- Judge 解析失败时允许有限次数重试；
- 重试后仍失败时将 `evaluate.answer` 标记为失败，不伪造零分；
- 使用少量人工复核 Case 校准 Judge，但第一版不建设多人复核流程。

## 8. 指标

### 8.1 Retrieval 指标

保留现有指标：

- Hit@1 / Hit@5 / Hit@10；
- Recall@5 / Recall@10 / Recall@20；
- MRR；
- nDCG@10；
- 各节点延迟；
- Evidence Journey 与首次丢失节点。

### 8.2 可回答性指标

```text
Answerability Accuracy
Answer False Negative Rate
Unsafe Answer Rate
```

定义：

- `Answerability Accuracy`：将 `direct_answer`、`qualified_answer` 合并为“可回答”，将 `abstain` 视为“应拒答”后，实际结果与标注一致的 Case 比例；
- `Answer False Negative Rate`：`direct_answer` 或 `qualified_answer` 被系统拒答的比例；
- `Unsafe Answer Rate`：`abstain` Case 被系统放行并生成事实性答案的比例。

`direct_answer` 与 `qualified_answer` 互换不计为可回答性错误。两者的答案质量差异由 Key Point、Claim 和 Citation 指标衡量，避免重复惩罚本地 Grade 对细粒度边界的判断。

### 8.3 Answer 指标

```text
Key Point Coverage
Key Point Correctness
Answer Score
Unsupported Claim Rate
Contradiction Rate
```

建议定义：

```text
Key Point Coverage
= covered required key points / all required key points

Key Point Correctness
= correct required key points / all required key points
```

Answer Score 的内部范围为 0 到 100，由结构化 Judge 结果确定性计算：

```text
base_score
= key_point_correctness * 65
  + citation_support_rate * 25
  + answerability_accuracy * 10

answer_score
= clamp(
    base_score
    - unsupported_claim_count * 15
    - contradiction_count * 30,
    0,
    100
  )
```

预期拒答但实际作答，以及预期可回答但实际拒答时，`answer_score` 直接为 0。预期和实际均为拒答时为 100。

第一版所有 Key Point 都视为 required，因此 Annotation 不需要 `importance` 字段。

### 8.4 Citation 指标

```text
Citation Support Rate
Unsupported Cited Claim Rate
Uncited Material Claim Rate
```

建议定义：

```text
Citation Support Rate
= supported cited claims / all cited claims

Unsupported Cited Claim Rate
= unsupported cited claims / all cited claims

Uncited Material Claim Rate
= material claims without a citation / all material claims
```

引用编号存在性和重编号继续由生产代码确定性处理；Judge 只负责语义支持关系。

## 9. 数据库改造

### 9.1 Case Draft 与 Frozen Case

建议分别增加：

```sql
alter table rag_evaluation_case_drafts
    add column if not exists answer_annotation jsonb;

alter table rag_evaluation_cases
    add column if not exists answer_annotation jsonb;
```

约束：

- Draft 在标注过程中允许为空；
- Case 进入 approved 前必须具有合法 Answer Annotation；
- Frozen Case 的 Answer Annotation 不可为空；
- `jsonb_typeof(answer_annotation) = 'object'`；
- 复杂跨字段约束由 Pydantic 和 Service 层校验；
- Dataset Version checksum 必须包含 Answer Annotation；
- 冻结时复制 Draft Annotation，并将每条 Key Point 的 Draft Evidence ID 映射为本次创建的 Frozen Evidence ID；
- 映射缺失、跨 Case 或不唯一时冻结失败。

现有 Service 强制所有 Case 至少标注一个正确 Chunk。为支持无答案 Case，需要将审核和冻结校验调整为：

```text
expected_verdict = abstain
  -> key_points 必须为空
  -> Gold Evidence 可选；若存在，仅表示已召回的相关上下文，不表示它足以支持直接回答

expected_verdict = direct_answer / qualified_answer
  -> 至少一条 Gold Evidence
  -> 至少一条 Key Point
  -> 每条 Key Point 至少关联一条当前 Case 的 Gold Evidence
```

这是 Service 校验和冻结逻辑调整，不需要增加数据库字段。

不建议第一版将 Answer Annotation 塞入通用 `metadata`，独立列可以明确版本、校验和查询边界，同时仍保持增量改造。

### 9.2 Case Result

第一版复用 `rag_evaluation_case_results.details` 保存端到端摘要：

```json
{
  "route_strategy": "fact_lookup",
  "route_error": null,
  "expected_verdict": "direct_answer",
  "actual_verdict": "direct_answer",
  "not_enough_evidence": false,
  "answer_step_status": "succeeded",
  "answer_evaluation_status": "succeeded"
}
```

Generated Answer 和 Sources 作为独立 `rag_evaluation_step_results` 保存：

```text
operation=answer.generate
output_kind=answer
```

Judge 结果保存为：

```text
operation=evaluate.answer
output_kind=answer_evaluation
```

避免将完整 Answer 和 Evidence 文本重复写入 Case Result details。

### 9.3 Metric Values

复用 `rag_evaluation_metric_values`，使用开放的 `operation + metric_name`：

```text
operation=grade.evidence
metric_name=answerability_accuracy

operation=evaluate.answer
metric_name=key_point_coverage

operation=evaluate.answer
metric_name=answer_score

operation=evaluate.citation
metric_name=citation_support_rate
```

## 10. 后端模块改造

建议新增：

```text
backend/packages/l2_core/rag_evaluation/
├── answer_annotations.py
├── answer_evaluator.py
└── answer_metrics.py
```

职责：

- `answer_annotations.py`：Pydantic 契约、Draft 校验、Evidence ID 约束；
- `answer_evaluator.py`：标注草稿生成与 Answer Judge；
- `answer_metrics.py`：根据结构化 Judge 结果确定性计算 Case 和 Run 指标。

现有 `RagEvaluationWorker` 继续作为统一执行入口。第一版不新增独立 Worker 进程，也不新增 Kafka Topic。

Worker 的 Case 流程从：

```text
run_retrieval -> grade_retrieval -> save
```

扩展为：

```text
run production answer graph
  -> save retrieval trace
  -> save grade
  -> save generated answer and sources
  -> deterministic abstention scoring
  -> optional single answer judge call
  -> save answer metrics
```

如果直接调用 `RagGraph.run()` 难以同时取得完整内部 State，可以增加一个只供应用层使用的结构化返回契约，或让 Evaluation Hook 捕获所需节点输出；不得复制一份生产 Graph 逻辑到评测模块。

## 11. API 与前端改造

### 11.1 Case 编辑

现有 Case 编辑区增加：

- Expected Verdict 下拉框；
- Reference Answer 文本框；
- Key Point 列表；
- 每条 Key Point 的 Supporting Evidence 多选；
- “生成答案标注草稿”按钮；
- 草稿来源提示与人工确认状态。

第一版不提供：

- Required Limitations 编辑器；
- Forbidden Facts 编辑器；
- Claim 图谱；
- 多人复核状态。

### 11.2 Run 详情

Run 汇总同时展示：

- Retrieval Metrics；
- Verdict Metrics；
- Answer Metrics；
- Citation Metrics；
- Judge 失败数量；
- Answer Evaluation 额外 Token 和延迟。

Case 下钻展示：

- Query；
- Expected / Actual Verdict；
- Reference Answer；
- Generated Answer；
- Key Point 覆盖结果；
- Claim–Citation 判断；
- Unsupported Claims 和 Contradictions；
- 实际 Sources；
- Retrieval Evidence Journey。

## 12. 失败与降级语义

### 12.1 生产 Graph 失败

Route、Retrieval、Grade 或 Answer 执行失败时，Case Result 标记为 failed，并保存原始错误。不得继续调用 Answer Judge。

### 12.2 正确拒答

Expected Verdict 为 `abstain` 且 Graph 正确拒答时，Case 正常 succeeded，写入正确拒答指标。拒答不是执行失败。

### 12.3 Judge 失败

Judge 失败时：

- 生产 Graph 和 Retrieval 结果仍然有效；
- `evaluate.answer` Step 标记为 failed；
- Case 可以标记为部分评测成功，并在 details 中记录 Judge 错误；
- Run 汇总必须披露 Judge 成功样本数；
- 不得把 Judge 失败 Case 当作 Answer 零分参与平均值。

第一版可继续沿用 Case 的 succeeded/failed 状态，并通过 `answer_evaluation_status` 表达部分失败；如果后续需要严格状态机，再增加 `partial` 状态。

### 12.4 Annotation 缺失

创建端到端 Run 前验证所有目标 Case 都有合法 Answer Annotation。任一 Case 缺失时拒绝创建 Run，并返回缺失 Case 列表，不能运行后再静默跳过。

## 13. 测试方案

### 13.1 Answer Annotation

- Answerable Annotation 缺少 Key Point 时校验失败；
- Answerable Annotation 缺少 Reference Answer 时校验失败；
- Abstain Annotation 包含 Key Point 时校验失败；
- Key Point 引用其他 Case 的 Evidence 时校验失败；
- 重复 Key Point ID 校验失败；
- 冻结 Version 后 Annotation 不可修改；
- Annotation 变化会改变 Dataset checksum。

### 13.2 模型辅助标注

- Suggest 只返回草稿，不写数据库；
- 非法 Evidence ID 被拒绝；
- 模型输出无法通过 Schema 时返回明确错误；
- 未经人工 PUT 保存的草稿不能进入 Frozen Version；
- 模型不得使用 Gold Evidence 之外的内容。

### 13.3 Answer Evaluator

- 同义表达能够覆盖 Key Point；
- 数字或单位错误时 Key Point 不正确；
- 引用存在但不支持 Claim 时 Citation 不通过；
- 无引用的重要事实被识别；
- 非 Gold Evidence 真实支持 Claim 时允许通过；
- Unsupported Claim 和 Contradiction 分开保存；
- Judge 输出解析失败按有限次数重试；
- Judge 最终失败不产生伪造指标。

### 13.4 端到端 Worker

- 同一 Run 同时保存 Retrieval 与 Answer Metrics；
- Route 结果写入 Case details，但不要求 expected strategy；
- 正确拒答 Case 不调用 Judge；
- 无答案错误放行 Case 调用 Judge；
- Answerable Case 只调用一次 Judge；
- 主评测不会启用 ASR Adjudication；
- Judge Token 和延迟不计入生产 Graph 指标；
- Run 汇总使用实际 Judge 成功样本数。

## 14. 实施步骤

### Step 1：Answer Annotation 契约与数据库

- 新增 `AnswerAnnotation` 和 `AnswerKeyPoint`；
- Case Draft 与 Frozen Case 增加 `answer_annotation`；
- 更新 Case API、审核校验、冻结 Evidence ID 映射和 checksum；
- 允许 `abstain` Case 在没有 Gold Evidence 时进入审核和冻结；
- 增加 Service 与数据库测试。

### Step 2：模型辅助标注

- 增加 Suggest Prompt 和结构化生成；
- 增加 Suggest API；
- 在 Case 页面增加草稿生成、编辑和保存；
- 保证模型草稿不能自动批准或冻结。

### Step 3：端到端执行与结果保存

- 评测 Worker 执行完整 Route、Retrieval、Grade 和 Answer；
- 从现有 Hook 保存 Retrieval Trace；
- 保存 `answer.generate` Step、Sources 和实际 Verdict；
- 对 Abstain Case 实现确定性评分。

### Step 4：Answer Judge

- 新增单次结构化 Judge；
- 保存 `evaluate.answer` Step；
- 实现 Key Point、Unsupported Claim、Contradiction 和 Citation 指标；
- 记录 Judge Model、Prompt Version、Token 和延迟。

### Step 5：前端结果与回归

- Run 页面增加 Answer 与 Citation 汇总；
- Case 页面增加答案对比和失败详情；
- 验证原 Retrieval 各阶段指标不回归；
- 验证现有 Adjudication Evaluation 不受影响。

## 15. 第一版验收标准

1. 同一 Evaluation Run 完成 Route、Retrieval、Rerank、Grade 和 Answer；
2. 同一 Run 同时展示 Retrieval、Verdict、Answer 和 Citation 指标；
3. 每个 Frozen Case 包含合法、不可变的 Answer Annotation；
4. 标注人员可以基于 Query 和 Gold Evidence 生成草稿，并在人工确认后保存；
5. `direct_answer`、`qualified_answer` 和 `abstain` 均有明确评测语义；
6. 无答案 Case 能计算正确拒答率和 Unsafe Answer Rate；
7. Answerable Case 每次最多调用一次 Answer Judge；
8. Judge 能输出 Key Point 覆盖、Citation 支持、Unsupported Claims 和 Contradictions；
9. Answer Judge 不进入生产 RagGraph，不改变生产答案和 Sources；
10. 主评测不启用 Risk Gate 或 ASR Adjudication；
11. Judge 失败不会被伪装为零分，Run 会披露有效样本数；
12. 原有 Retrieval Trace、Evidence Journey 和指标保持可用。

## 16. 后续扩展触发条件

只有当第一版数据证明存在明确需求时，再考虑：

- Route 错误成为主要失败来源：增加 `expected_strategy` 和 Route Accuracy；
- 部分回答经常未披露证据边界：增加 `required_caveats`；
- 无依据补充集中在稳定模式：增加 `forbidden_facts`；
- 单次 Judge 不稳定：引入人工校准集或多 Judge；
- Judge 无法稳定拆分复杂答案：增加独立 Claim Extraction；
- 需要衡量纠错对最终答案的收益：在 Adjudication Evaluation 中增加原始/纠错答案对比；
- 需要生产发布门禁：为关键指标增加 Baseline Regression Threshold。

第一版不为这些未来需求预先增加复杂表结构和工作流节点。
