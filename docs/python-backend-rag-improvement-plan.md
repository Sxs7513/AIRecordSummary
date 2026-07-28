# Python 后端 RAG 分步改造计划

## 1. 文档目标

本文档用于管理 `backend/packages/l2_core/rag` 的后续改造。改造采用小步提交：

- 每一步只解决一个明确问题；
- 每一步都可以独立测试、上线和回滚；
- 前一步稳定后再开始下一步；
- 不在同一个改动中同时重构检索、工作流和前端协议。

现阶段计划解决：

1. 将 `chunk_search` 升级为混合检索；
2. 只在复杂问题中执行 answer plan；
3. 让最终返回和持久化的 evidence 与回答实际使用的 evidence 一致；
4. 优化 chunk context 和 scope retrieval 的数据库查询；
5. 将录音日期边界改为 `Asia/Shanghai` 时区感知时间。

## 2. 本轮不做

以下能力暂不进入本轮改造：

- claim 级引用；
- 回答段落与录音时间段的逐条绑定；
- 可回放溯源；
- reranker 模型；
- GraphRAG；
- query decomposition；
- 外部向量数据库或搜索服务；
- 大规模 RAG 评测平台。

## 3. 总体原则

### 3.1 权限边界不变

所有向量、关键词和 scope 查询必须在 SQL 检索阶段应用当前用户的可访问录音范围。不得先检索全库再在 Python 中过滤。

### 3.2 原始召回与最终证据分离

工作流中区分：

```text
evidence
  原始召回并通过 grade 检查的证据

answer_evidence
  实际提供给最终回答模型，并最终返回、持久化的证据
```

最终必须满足：

```text
最终回答模型看到的 evidence
= API 返回的 sources
= generation result 中的 sources
= conversation message 持久化的 sources
```

### 3.3 新能力默认可回退

混合检索必须提供 feature flag。新检索分支出现问题时，可以退回现有纯向量检索，不影响 `scope_summary`。

### 3.4 每一步保持测试通过

每个步骤完成后至少执行：

```bash
backend/.venv/bin/pytest backend/tests/unit/test_rag_graph.py
backend/.venv/bin/pytest backend/tests/unit/test_rag_router.py
backend/.venv/bin/pytest backend/tests/unit/test_rag_model.py
backend/.venv/bin/pytest backend/tests/unit/test_rag_prompts.py
backend/.venv/bin/pytest backend/tests/unit/test_generation_streaming.py
```

涉及 SQL 时增加对应的 repository/retrieval 测试；涉及真实 PostgreSQL 查询计划时补充集成验证。

## 4. 目标工作流

```text
route
  -> retrieve
  -> grade
      -> insufficient -> rewrite/retrieve 或结束
      -> sufficient
          -> decide_plan
              -> direct -> select_direct_evidence
              -> planned -> plan -> validate_plan -> select_planned_evidence
  -> answer(answer_evidence)
  -> return and persist(answer_evidence)
```

`decide_plan` 只决定回答组织方式，不改变检索范围，也不重新判断证据是否充分。

## 5. 分步实施计划

### Step 1：混合检索基础设施

状态：`completed`

详细设计以 [Python 后端 RAG 混合搜索技术方案](python-backend-rag-hybrid-search.md) 为准。本步骤只建立结构，不立即切换线上行为。

#### 1.1 提取统一文本标准化

新增：

```text
backend/packages/l2_core/rag/retrieval/normalization.py
```

提供：

```python
def normalize_search_text(value: str) -> str:
    ...
```

索引写入和关键词查询必须复用同一实现。

验收：

- 中英文、全角字符、大小写和连续空白的标准化结果稳定；
- 原始 chunk `text` 不被修改；
- 现有向量检索行为不变。

#### 1.2 提取共享 SQL filters

将当前 `RagRetriever` 中的过滤条件提取为向量和关键词查询可复用的构建器。

验收：

- 两路查询使用相同的录音权限、时间、人物、地点和说话人过滤；
- `match_none` 不执行实际召回；
- 现有向量检索测试保持通过。

#### 1.3 增加 lexical retrieval

使用 `pg_trgm` 对 `normalized_text` 召回关键词候选。先只实现和测试，不接入最终检索结果。

数据库新增适合 Top N 距离排序的 GiST 索引：

```sql
create index if not exists recording_search_chunks_text_trgm_gist_idx
    on recording_search_chunks
    using gist (normalized_text gist_trgm_ops(siglen=64));
```

验收：

- 型号、英文缩写、人名、数字可以通过 lexical 分支命中；
- lexical 查询独立应用完整权限过滤；
- lexical 分支不占用 GPU queue。

#### 1.4 增加 RRF fusion

将 vector 和 lexical 候选按 `chunk_id` 去重并通过 RRF 融合。

初始配置沿用混合检索设计：

```text
vector candidate limit = 30
lexical candidate limit = 30
fused candidate limit = 20
rrf_k = 60
vector weight = 1.0
lexical weight = 1.0
```

验收：

- 相同 chunk 被两路命中时只保留一次；
- `vector`、`lexical`、`hybrid` match type 正确；
- 任意单路失败时可以使用另一分支；
- 两路都失败时 generation 明确失败；
- feature flag 关闭时保持纯向量检索。

实现说明：

- route 阶段先把录音级过滤统一解析成明确的 recording IDs；
- vector 和 lexical 只在同一 recording ID 集合内召回；
- 人物、speaker profile 和 target person 等 chunk 级约束仍在两路 SQL 中保留；
- query embedding 单独进入 GPU 队列，lexical SQL 可与 embedding 并行；
- 融合后只执行一次批量 context expansion，并合并同一录音内时间范围重叠的 evidence；
- 旧数据需要重新执行 embedding indexing 投影以按 NFKC 规则刷新 `normalized_text`，不需要重新训练或更换 embedding 模型。

### Step 2：批量扩展 chunk context

状态：`completed`

对最终候选执行一次批量上下文扩展。本步骤只优化查询次数；重叠 evidence 合并仍按混合检索详细方案随融合链路实施。

将逐候选的 `_expand_chunk_context()` 改为：

```python
def expand_candidates(
    connection: Connection,
    candidates: Sequence[RetrievalCandidate],
    window: int,
) -> list[ExpandedCandidate]:
    ...
```

实现要求：

- 一次查询得到全部候选的 source utterance 上下界和扩展 utterance；
- 按 `chunk_id` 在 Python 中分组；
- 每个 chunk 内的 utterance 按 `utterance_index` 排序；
- 不改变当前 evidence 数量、顺序和 URL 语义。

验收：

- context expansion SQL 次数不随候选数量增长；
- 扩窗内容与现有逻辑一致；
- 没有 source utterance 或无法解析边界的候选保留原始内容。

### Step 3：判断是否需要 answer plan

状态：`completed`

判断发生在 `grade` 成功之后，因为此时同时具备 query、route 和实际 evidence。

#### 3.1 扩展 EvidenceGrade

```python
class EvidenceGrade(BaseModel):
    sufficient: bool
    rewrite_query: str | None = None
    planning_required: bool = False
    planning_reason: str = ""
    reason: str = ""
```

规则：

- `sufficient=false` 时 `planning_required=false`；
- 简单事实、单一结论和简单局部总结不需要 plan；
- 多子问题、比较、时间线、分组、跨录音综合需要 plan；
- `scope_summary` 覆盖多条录音时由后端强制进入 plan。

#### 3.2 增加 decide_plan 分支

```text
grade
  -> decide_plan
      -> direct
      -> plan -> validate_plan
```

`decide_plan` 是纯状态节点，不增加新的 LLM 调用。

验收：

- 简单问题不调用 plan；
- 复杂问题只调用一次 plan；
- evidence 不足时不进入 plan；
- 多录音 scope summary 强制进入 plan；
- plan 决策原因写入日志但不返回前端。

### Step 4：统一最终 answer evidence

状态：`completed`

在 graph state 增加：

```python
answer_evidence: list[Evidence]
```

#### 4.1 未进入 plan

```python
answer_evidence = evidence
```

#### 4.2 进入 plan

从所有 `AnswerPlanItem.evidence_indexes` 计算 index 并集，再从原 evidence 中按原顺序选择：

```python
selected_indexes = {
    index
    for item in plan.items
    for index in item.evidence_indexes
}

answer_evidence = [
    item
    for item in evidence
    if item.index in selected_indexes
]
```

#### 4.3 Plan 校验

- 删除不存在的 evidence index；
- 对重复 index 去重；
- 删除清理后没有 evidence 的 plan item；
- 清理后 plan 为空时使用 fallback plan；
- 不允许 plan 引入原 evidence 之外的来源。

#### 4.4 输出规则

只有 `answer_evidence` 可以进入：

- `answer_prompt()`；
- `source_payload()`；
- generation result；
- conversation message sources。

证据不足且没有生成正常回答时，最终 sources 为空。原始召回结果只保留在内部日志或诊断信息中。

验收：

- plan 选择 `[2, 4]` 时，answer prompt、API sources 和数据库 sources 都只有 evidence 2、4；
- 未进入 plan 时三处都使用相同的全部 answer evidence；
- 非法 plan index 不会进入最终输出；
- source 中仍不持久化 chunk 全文。

### Step 5：优化 retrieve_scope 查询

状态：`completed`

将当前随录音数量增长的查询改为固定两次查询：

1. 查询选中的 recordings；
2. 一次查询全部选中录音的 bounded utterances、utterance count 和 speaker statistics。

实现可使用：

- `row_number() over (partition by recording_id order by utterance_index)`；
- `count(*) over (partition by recording_id)`；
- speaker statistics CTE；
- Python 按 recording ID 组装 Evidence。

验收：

- 查询次数不随录音数量增长；
- `MAX_SCOPE_UTTERANCES` 和 `MAX_SCOPE_CHARS` 语义不变；
- speaker count 基于完整录音，不受正文截断影响；
- 没有 utterance 的录音仍生成 scope evidence；
- 录音排序与当前实现一致。

### Step 6：时区感知的日期边界

状态：`completed`

将：

```python
created_from: str | None
created_to: str | None
```

改为：

```python
created_from: datetime | None
created_to: datetime | None
```

所有日期边界使用：

```python
ZoneInfo("Asia/Shanghai")
```

并采用半开区间：

```text
[created_from, created_to)
```

SQL 直接绑定 aware datetime：

```sql
recordings.created_at >= :created_from
recordings.created_at < :created_to
```

不再依赖：

```sql
cast(:created_from as timestamptz)
```

验收：

- 时间参数包含明确的 `+08:00` offset；
- PostgreSQL session timezone 为 UTC 或 Asia/Shanghai 时结果一致；
- 今天、昨天、本周、上周、本月、跨年和绝对日期区间测试通过。

### Step 7：RAG 全链路结构化日志

状态：`completed`

使用统一 JSON event 记录 RAG workflow 和 LangGraph 流转。每条事件至少包含：

```text
event
run_id
```

节点事件额外包含：

```text
node
attempt
elapsed_ms
```

已覆盖：

- workflow submitted、started、succeeded、failed、cancelled；
- graph started、succeeded、route error、evidence insufficient、failed；
- route、retrieve、grade、rewrite、decide_plan、plan、validate_plan、answer；
- conditional edge 的 source、target 和 reason；
- answer direct/planned 模式及首 token 延迟；
- grader、plan 和 plan validation fallback warning。

完整模型 raw output 只写入 DEBUG，不在默认 INFO 日志中输出。

## 6. 推荐执行顺序

严格按以下顺序逐步实施：

```text
Step 1.1 统一标准化
Step 1.2 共享 filters
Step 1.3 lexical retrieval
Step 1.4 RRF 与混合检索开关
Step 2   批量 chunk context
Step 3   自适应 plan
Step 4   最终 answer evidence
Step 5   批量 scope retrieval
Step 6   时区感知日期边界
```

其中 Step 3 和 Step 4 应连续完成，但仍拆成两个独立提交：

- Step 3 只改变是否调用 plan；
- Step 4 才改变最终 sources 的选择和持久化行为。

## 7. 进度记录

| Step | 内容 | 状态 | 备注 |
|---|---|---|---|
| 1.1 | 统一搜索文本标准化 | pending | |
| 1.2 | 提取共享 SQL filters | pending | |
| 1.3 | lexical retrieval | pending | |
| 1.4 | RRF 与混合检索开关 | pending | |
| 2 | 批量 chunk context | completed | 固定一次批量扩窗查询；重叠合并留待混合检索 |
| 3 | 自适应 answer plan | completed | 复用 grade 判断；多录音 scope summary 强制 plan |
| 4 | 统一 answer evidence | pending | |
| 5 | 批量 scope retrieval | completed | 从 `1 + 2N` 次查询降为固定两次 |
| 6 | 时区感知日期边界 | completed | 使用 Asia/Shanghai aware datetime |
| 7 | RAG 全链路结构化日志 | completed | 统一 JSON event，并使用 generation run ID 关联 |

每完成一步，应在本表更新状态，并记录对应提交或 PR。
