# RAG Adjudication 离线评测技术方案

> 文档状态：一期方案。
>
> 关联文档：[RAG 离线评测平台](./rag-offline-evaluation-platform-design.md)、[RAG ASR Evidence Adjudication Agent](./rag-asr-evidence-adjudication-agent-design.md)。

## 1. 背景与目标

`backend/packages/l2_core/rag/adjudication/agent.py` 中的 `EvidenceAdjudicationAgent` 会对最终 Evidence 中与用户 Query 相关的 ASR 关键表达进行审计、候选重建和裁决，必要时运行受限 Loop。

本评测回答的问题是：

> 给定人工固定的 Query、Target Evidence 和 Reference Evidence，Adjudication Agent 能否把人工标注的关键 ASR 错误修改成可接受的正确表达。

本评测不从 Query 重新执行召回。Target 和 Reference 由标注人员从 Workspace 的全部 SearchChunk 中搜索和选择，运行时直接从 Adjudication Agent 开始，从而隔离召回、排序和 Grade 变化。

## 2. 核心结论

### 2.1 平台整合，任务独立

复用现有评测平台控制面：

- `evaluation_datasets`；
- `evaluation_dataset_versions`；
- `evaluation_runs`；
- Dataset 审核、冻结、Run 状态和 Workspace 权限。

新增独立任务：

```text
task_type = rag_adjudication
evaluator_type = rag_adjudication
```

Adjudication 使用自己的 Case、Gold、Run Spec 和 Result 表，不复用 Retrieval Case 明细。

### 2.2 保持 Retrieval 和 Grade 独立

`run_retrieval` 和 `grade_retrieval` 继续作为两个独立入口。Grade 与召回流程可以独立评测和演进，本次不合并。

Adjudication 评测不调用二者：

```text
Frozen Adjudication Case
  -> EvidenceAdjudicationAgent
  -> AdjudicationAgentState
  -> Targeted Correction Metrics
```

### 2.3 一期只以最终修正结果为主

一期不把 `audit -> reconstruct -> decision` 做成三套独立评测。主指标只判断人工标注的关键错误最终是否被修改正确。

Run 仍保存完整 `AdjudicationAgentState`，失败 Case 可以展开查看 Audit、Proposal、Decision、Finding、Overlay 和终止原因。

## 3. 一期范围

### 3.1 包含

- 保存 Query，不评测 Risk Gate；
- 从全部 SearchChunk 中搜索和选择 Evidence；
- 每个 Case 选择一到两个 Target；
- 每个 Case 选择零到多个 Reference；
- 为 Target 标注一个或多个与 Query 相关的关键错误；
- 每个错误支持多个可接受的正确表达；
- 冻结 Evidence 文本快照和 Gold；
- 直接执行 Adjudication Agent；
- 计算 Gold Correction 级修正准确率；
- 保存完整 Agent Trace。

### 3.2 不包含

- 不从生产回答或 Generation Run 导入 Evidence；
- 不执行 Route、Retrieval 或 Grade；
- 不保存或评测 Risk；
- 音频不是 Frozen Case 的必需资产；
- 暂不要求 Web Search 跨运行可复现；
- 不要求枚举 Target 中的全部 ASR 错误；
- 未标注区域的额外修改不扣分；
- 不单独建设 Audit、Reconstruct 和 Decision 指标页。

## 4. 当前 Agent 行为约束

### 4.1 Target 数量

当前 `RagGraph` 配置：

```python
MAX_ADJUDICATION_CASES = 2
```

`EvidenceAdjudicationAgent.initialize()` 会将 `answer_evidence[:max_cases]` 中的每一条 Evidence 建立为 Target Case。因此：

- 一期 UI 最多允许两个 Target；
- 运行时 Target 必须排在 `answer_evidence` 前面；
- `max_cases` 设置为实际 Target 数量；
- Target/Reference 角色必须显式存储，不能只依赖数组位置。

### 4.2 Reference 规则

当前 `_context()` 对每个 Target 使用以下 Reference：

- 位于同一个 `answer_evidence`；
- 与当前 Target 属于同一录音；
- 不是当前 Target 自身；
- 按 `start_ms`、`end_ms` 和 `evidence_index` 排序。

由此得到以下交互约束：

- Reference 不需要人工排序；
- UI 应限制或提示只选择与至少一个 Target 同录音的 Reference；
- 与所有 Target 都不同录音的 Reference 会被忽略；
- 两个 Target 属于同一录音时，当前实现中它们也会互相作为 Reference。

### 4.3 Query、Risk 和 Plan

- Case 必须保存 Query；
- 评测不执行 Risk Gate，运行时直接使用 `query_correction_risk=true`；
- Dataset 不保存 Plan；
- 当前代码需要 `answer_plan` 字段时传入 `None`，由 Agent 使用 Query 构造兼容上下文。

## 5. 标注模型

### 5.1 Case

```json
{
  "query": "这里提到的接口协议是什么？",
  "targets": [],
  "references": [],
  "tags": []
}
```

Target/Reference 角色在数据库中显式保存。数组位置仅用于适配当前 Agent。

### 5.2 Evidence Snapshot

每条 Evidence 至少保存：

```json
{
  "role": "target",
  "position": 0,
  "source_recording_id": "uuid",
  "source_chunk_id": "uuid",
  "recording_title": "录音标题快照",
  "chunk_index": 12,
  "text": "完整 SearchChunk 文本快照",
  "start_ms": 120000,
  "end_ms": 128000,
  "content_checksum": "sha256",
  "metadata": {}
}
```

约束：

- Target 数量为 1 或 2；
- Reference 数量为 0 或多个；
- Target `position` 决定处理顺序；
- Reference 的人工顺序不进入 Agent；
- 来源录音和 Chunk ID 是弱引用，不建立阻止删除的外键；
- 原录音或 Chunk 删除后，文本快照仍可用于评测；
- 音频存在时可以辅助标注，不要求复制音频资产。

### 5.3 Gold Correction

标注人员只选择与 Query 相关且必须修正的错误表达：

```json
{
  "target_evidence_id": "uuid",
  "start_char": 12,
  "end_char": 14,
  "original_expression": "RF",
  "accepted_expressions": ["I²C", "I2C"]
}
```

`start_char` 包含起始字符，`end_char` 为开区间，并满足：

```python
target_text[start_char:end_char] == original_expression
```

约束：

- 每个 Target 一期至少有一条 Gold Correction；
- 同一个 Target 可以有多条 Correction；
- Gold Span 不能相互重叠；
- `accepted_expressions` 至少有一个值，去除首尾空白后去重；
- 未标注区域不被视为正确负例。

### 5.4 Span 计算时机

必须先完成并保存 Target 和 Reference 选择，固定 Evidence 文本及 Target 顺序，再进入 Gold 标注。Evidence 发生增删、角色变更或文本变化后，Case 回到 `draft`，后端重新验证所有 Span。

## 6. 前端标注流程

### 6.1 录入 Query

创建 Case 并录入 Query。

### 6.2 搜索和编排 Evidence

复用当前 RAG 评测的 SearchChunk 搜索能力，支持：

- 全文搜索；
- 按录音筛选；
- 查看录音标题、Chunk 文本和时间范围；
- 将结果加入 Target 或 Reference；
- 在两个角色间移动；
- 调整 Target 顺序；
- 删除已选 Evidence。

页面显式分组：

```text
Target Evidence（1-2 条）
├── Target 1
└── Target 2

Reference Evidence（0-N 条）
├── Reference
└── Reference
```

### 6.3 标注 Gold Correction

只有 Evidence 选择保存后才能进入该步骤：

1. 显示只读 Target 文本；
2. 用户通过浏览器文本选区选择错误表达；
3. 前端自动计算 `start_char/end_char`；
4. 用户录入一个或多个可接受表达；
5. 添加 Correction 并高亮区间；
6. 支持删除和重新标注。

用户不手工填写 Span。JavaScript Selection 通常使用 UTF-16 code unit，而 Python 使用 Unicode code point；前端提交前必须转换，后端必须使用快照文本再次校验。

### 6.4 音频降级

录音仍存在时，可以根据 `start_ms/end_ms` 提供播放入口。录音删除后：

- Case 仍可查看和运行；
- 音频入口禁用；
- Dataset Version、Run 和 Result 不删除。

## 7. 建议数据模型

### 7.1 通用枚举

```sql
evaluation_datasets.task_type += 'rag_adjudication'
evaluation_runs.evaluator_type += 'rag_adjudication'
```

继续复用 `evaluation_dataset_versions`，Frozen Version 不可修改。

### 7.2 Draft 表

```text
rag_adjudication_evaluation_case_drafts
  id
  dataset_id
  query
  tags
  status / revision
  reviewer / approver
  created_at / updated_at

rag_adjudication_evaluation_evidence_drafts
  id
  case_draft_id
  role                  target | reference
  position
  source_recording_id   weak provenance
  source_chunk_id       weak provenance
  recording_title
  chunk_index
  text
  start_ms / end_ms
  content_checksum
  metadata

rag_adjudication_evaluation_correction_drafts
  id
  target_evidence_draft_id
  start_char / end_char
  original_expression
  accepted_expressions[]
  created_at / updated_at
```

Service 负责校验 Target 数量、Evidence 角色、Span 文本一致性和区间不重叠。Evidence 或 Gold 变更后重置审核状态并增加 revision。

### 7.3 Frozen 表

```text
rag_adjudication_evaluation_cases
rag_adjudication_evaluation_evidence
rag_adjudication_evaluation_corrections
```

冻结时复制完整文本快照并计算 Case/Version Checksum。来源 ID 不设置强外键。

### 7.4 Run 与 Result

```text
rag_adjudication_evaluation_run_specs
  evaluation_run_id
  config_snapshot
  code_commit

rag_adjudication_evaluation_case_results
  id
  evaluation_run_id
  evaluation_case_id
  status
  latency_ms
  token_usage
  agent_state
  overlays
  pending_confirmation
  error_type / error_message
  started_at / finished_at

rag_adjudication_evaluation_correction_results
  case_result_id
  gold_correction_id
  matched_overlay_id
  passed
  actual_expression
  details
```

一期不额外建立 Audit、Reconstruct 和 Decision 分阶段结果表。

## 8. 评测执行

### 8.1 MVP：手动构建 Agent 与 State

手动构造 `EvidenceAdjudicationAgent` 入参可行，适合作为第一阶段验证方式。运行时：

```python
target_evidence = build_evidence(case.targets)
reference_evidence = build_evidence(case.references)
answer_evidence = [*target_evidence, *reference_evidence]

agent = EvidenceAdjudicationAgent(
    model_client=model_client,
    online_provider=online_provider,
    context_size=settings.rag_context_size,
    token_budget=token_budget,
    grounded_search_client=grounded_search_client,
    web_search_enabled=settings.asr_adjudication_web_search_enabled,
    auto_resolve_confidence=settings.asr_adjudication_auto_resolve_confidence,
    max_cases=len(target_evidence),
    max_iterations=MAX_ADJUDICATION_ITERATIONS,
    max_searches=MAX_ADJUDICATION_SEARCHES,
    audit_prompt_variant=settings.asr_adjudication_audit_prompt_variant,
    audit_model=settings.asr_adjudication_audit_model,
    audit_min_request_interval_seconds=settings.asr_adjudication_audit_min_request_interval_seconds,
    node_started=evaluation_node_started,
    node_completed=evaluation_node_completed,
    event_logger=evaluation_event_logger,
)

result = await agent.start(
    {
        "run_id": str(case_result_id),
        "query": case.query,
        "query_correction_risk": True,
        "answer_plan": None,
        "answer_evidence": answer_evidence,
        "evidence": answer_evidence,
        "adjudication_agent_state": None,
        "token_usage": 0,
    }
)
```

实际实现时按当前 `RagGraphState` 补齐必填默认值，并复用生产 Composition Root 中 Model Client、Token Budget 和观测回调的构造方式。

### 8.2 长期公共入口

MVP 稳定后增加薄适配层：

```python
async def run_adjudication_evaluation_case(
    *,
    query: str,
    targets: list[Evidence],
    references: list[Evidence],
    run_id: str,
) -> AdjudicationAgentState:
    ...
```

该入口只负责构造 State 和调用同一个 Agent，不复制 Audit、Reconstruct 或 Decision 逻辑。

### 8.3 Worker 分发

```python
if evaluator_type == "rag_retrieval":
    await run_rag_retrieval_evaluation(run)
elif evaluator_type == "rag_adjudication":
    await run_rag_adjudication_evaluation(run)
```

现有 Retrieval Worker 执行逻辑不改变。

### 8.4 Web Search

一期允许实时 Web Search，不要求冻结 Search Fixture。Run 仍保存实际 Finding、URL、查询和时间，以便解释当次结果。

## 9. 评分规则

### 9.1 Gold 匹配

对每条 Gold Correction，在相同 Target 的最终 `EvidenceOverlay` 中查找：

```text
overlay.evidence_index == target.evidence_index
overlay.chunk_id == target.chunk_id
overlay.target_spans 包含 (gold.start_char, gold.end_char)
normalize(overlay.resolved_expression) in normalize(gold.accepted_expressions)
```

`normalize()` 一期只执行 Unicode NFKC、首尾空白去除和连续空白合并，不做语义模糊匹配。

没有 Overlay、Span 不匹配、表达不在可接受列表、只产生 Proposal/Confirmation 但未形成 Overlay，均判为未通过。

### 9.2 未标注修改

Agent 对未标注区域的修改不影响一期评分，但仍保存在 Result 中供人工查看。因此指标只表示“Query 相关关键错误修正能力”，不表示完整文本纠错准确率或安全性。

### 9.3 聚合指标

一期只提供一个质量指标，并同时展示通过数和总数：

```text
gold_correction_accuracy
  = 通过的 Gold Correction 数 / Gold Correction 总数
```

某个 Case 执行异常时，该 Case 下的 Gold Correction 均按未通过计入分母。耗时、Token、循环次数和错误信息继续保存在 Run Result 中作为运行诊断信息，但不作为一期评测指标。

Audit Coverage、Proposal Coverage、未标注误改率、Confirmation 质量和最终 Answer 收益留待后续阶段。

## 10. API 草案

```text
GET    /api/evaluation/rag-adjudication/datasets
POST   /api/evaluation/rag-adjudication/datasets
GET    /api/evaluation/rag-adjudication/datasets/{dataset_id}

POST   /api/evaluation/rag-adjudication/datasets/{dataset_id}/cases
PATCH  /api/evaluation/rag-adjudication/cases/{case_id}
DELETE /api/evaluation/rag-adjudication/cases/{case_id}

GET    /api/evaluation/rag-adjudication/chunks?query=...&recording_id=...
POST   /api/evaluation/rag-adjudication/cases/{case_id}/evidence
PATCH  /api/evaluation/rag-adjudication/evidence/{evidence_id}
DELETE /api/evaluation/rag-adjudication/evidence/{evidence_id}

POST   /api/evaluation/rag-adjudication/evidence/{target_id}/corrections
PATCH  /api/evaluation/rag-adjudication/corrections/{correction_id}
DELETE /api/evaluation/rag-adjudication/corrections/{correction_id}

POST   /api/evaluation/rag-adjudication/datasets/{dataset_id}/versions:preview
POST   /api/evaluation/rag-adjudication/datasets/{dataset_id}/versions:freeze

POST   /api/evaluation/rag-adjudication/runs
GET    /api/evaluation/rag-adjudication/runs
GET    /api/evaluation/rag-adjudication/runs/{run_id}
DELETE /api/evaluation/rag-adjudication/runs/{run_id}
```

Chunk 搜索可复用当前 RAG 评测 Service，但 Evidence 加入 Case 时必须立即复制文本快照。

## 11. 前端页面结构

在现有 RAG 评测导航下增加任务切换：

```text
RAG 评测
├── 检索评测
└── ASR 文本裁决评测
```

一期页面：

```text
数据集
  -> Case
      -> Query
      -> SearchChunk 搜索
      -> Target Evidence（1-2）
      -> Reference Evidence（0-N）
      -> Gold Correction 标注
      -> Review / Approve
  -> Dataset Version
  -> Evaluation Run
      -> 总指标
      -> Case 通过/失败
      -> Gold 与 Overlay 对比
      -> Agent Trace 展开
```

一期不引入通用富文本编辑器，使用只读 Target 文本、浏览器 Selection API 和非重叠区间高亮。

## 12. 数据生命周期

Adjudication Dataset 是文本快照数据集。录音或原 SearchChunk 删除时：

- 不删除 Draft；
- 不删除 Frozen Version；
- 不删除历史 Run 和 Result；
- 不移除 Evidence Snapshot；
- 仅将来源状态显示为 unavailable，并禁用音频或原录音跳转。

当前 `RagEvaluationOrphanCleanup` 会删除引用已删除录音的 Retrieval Version、Corpus Snapshot 和 Run。实现 Adjudication 时不得将 `rag_adjudication` 纳入该破坏性清理策略。

任何影响 Agent 输入或 Gold 的 Draft 变更都必须重置审核状态、增加 revision 并重新计算 Checksum，不修改已冻结版本。

## 13. 实施顺序

### 阶段一：后端评测闭环

Runner、数据模型和 API 同期实现，避免维护只服务于临时离线文件的重复 Contract：

1. 定义 Case Snapshot、Evidence Snapshot 和 Gold Correction Contract；
2. 扩展 `task_type/evaluator_type`；
3. 增加 Draft、Frozen、Run Spec 和 Result 表；
4. 实现 SearchChunk 搜索、Evidence 选择和文本快照 API；
5. 实现 Review、Approve、Preview 和 Freeze；
6. 在 Worker 中手动构造 Agent 依赖和最小 State；
7. 实现 Worker 分发、Run 状态机及 Agent State 序列化；
8. 实现 Overlay 与 Gold 匹配和 `gold_correction_accuracy`；
9. 通过 API 和少量固定 Case 验证完整后端闭环。

### 阶段二：前端

1. 增加 RAG 评测任务切换；
2. 实现 Query 和 Evidence 搜索；
3. 实现 Target/Reference 分组及 Target 重排；
4. 实现文本选区、Unicode Span 转换和多区间高亮；
5. 实现 Dataset Version 和 Run 管理；
6. 实现 Gold/Overlay 对比与 Trace 展开。

## 14. 验收标准

- 可以从全部 SearchChunk 搜索并组建 Case；
- Case 支持 1-2 个 Target 和多个 Reference；
- Evidence 角色显式存储；
- 只在 Evidence 选择保存后标注 Gold Span；
- 后端拒绝越界、重叠或与原文不符的 Span；
- 每条 Gold 支持多个可接受表达；
- Frozen Version 不依赖原录音或 SearchChunk 继续存在；
- Worker 不执行 Retrieval 和 Grade，直接运行 Adjudication Agent；
- 评分只要求已标注关键错误被修改到任一可接受表达；
- 未标注区域的额外修改一期不扣分；
- Run 保存完整 Agent State、耗时、Token 和错误信息；
- `run_retrieval` 和 `grade_retrieval` 保持独立。
