# RAG History Context 设计

## 1. 文档目标

本文定义多轮录音问答中历史对话的存储、选择、加载和使用边界，解决以下问题：

- 历史对话需要帮助 route 理解追问、指代和录音范围；
- 历史 assistant 回答可能来自旧版 ASR、Embedding、索引、检索参数或证据门禁，不能自动成为当前事实；
- answer 不应默认接收整段历史原文，但在继续、比较、编辑和总结历史内容时仍需要按需读取指定消息；
- 历史正文必须保留唯一事实来源，context 只保存引用和解析结果，不能复制一份容易过期的正文或 Evidence；
- route 只负责对话语义和录音范围，不提前承担 `expand_retrieval_terms` 的关键词提取职责。

本文是 [多轮录音问答与前端聊天方案](python-backend-multi-turn-chat-plan.md) 的细化设计，不改变 `conversation_messages` 作为聊天原文长期存储、`generation_runs` 作为单次生成执行记录的既有边界。

## 2. 当前问题

当前每轮会读取最近的 user / assistant 消息，形成 `RagHistoryMessage[]`。同一份 history 存在两条消费路径：

```text
history ──> route ──> content_query / filters ──> retrieval ──> evidence ──> answer
   └────────────────────────────────────────────────────────────────────> answer
```

第一条路径符合 RAG 的证据边界：route 使用历史理解当前问题与历史回答的关联，并可根据历史 assistant message 的 sources 收窄录音范围。

第二条路径把历史 user / assistant 原文直接拼入 answer prompt。历史 assistant 回答中可能包含“无法确认”“没有找到”等旧检索结论。Embedding、索引、chunk、ASR 纠偏或检索策略升级后，本轮 Evidence 可能已经能够回答，但旧措辞仍会锚定模型，导致同一答案同时出现确定结论和“无法确认”。

必须区分：

```text
旧运行没有检索到证据
≠ 录音中不存在相关内容
≠ 当前运行仍然无法回答
```

## 3. 设计原则

### 3.1 保存与使用分离

完整历史始终保存在 `conversation_messages`。某轮是否使用历史、使用哪些消息、以什么方式使用，由 route 为当前问题重新判断。

历史默认保留、按需参与；不能因为历史被持久化，就默认把全部原文注入 answer。

### 3.2 当前 Evidence 是录音事实依据

历史 assistant 回答是历史模型产物，不是录音事实证据。录音事实必须来自本轮重新获得并完成必要纠偏、评分的 Evidence 或可信结构化事实。

优先级为：

```text
本轮当前 Evidence
> 用户明确确认且仍有效的决定
> 本轮会话约束与任务状态
> 历史 assistant 回答
```

历史 assistant 原文只有在它本身是操作对象时才进入 answer，例如编辑上一版回答或总结指定历史问答。即使进入，也必须标注为“历史内容，不是录音事实证据”。

### 3.3 context 不复制历史正文和 Evidence

- 历史消息正文继续只存于 `conversation_messages.content_blocks`；
- 历史引用继续只存于 assistant message 的 `sources`；
- context 保存 message ID、录音 ID、使用方式和 route 解析结果；
- 需要历史原文时，根据 message ID 从 `conversation_messages` 读取；
- 需要验证历史事实时，根据当前授权和当前索引重新检索或重新加载来源，不能复用旧 answer 作为事实。

### 3.4 route 与关键词提取职责分离

route 输入不增加预提取的 `entities` 或关键词。route 继续读取当前问题、有限历史和可选持久会话状态，输出 `content_query`、strategy 和范围解析结果。

`expand_retrieval_terms` 继续在 route 之后从 `content_query` 提取 terms、phrases 和 lexical queries。History Context 设计不能在 route 前复制这项职责。

### 3.5 会话状态与单轮解析分离

一个会话维护一个可选的、持续更新的 `ConversationContext`；每次问答生成一个不可变的 `TurnContext` 快照。

- `ConversationContext` 保存当前仍有效的跨轮状态；
- `TurnContext` 保存本轮如何解释和使用历史；
- 历史各轮的使用方式由各自的 `TurnContext` 记录，不能压缩成会话级单值。

## 4. 数据模型

### 4.1 历史消息输入

当前 `RagHistoryMessage` 只有 role、content 和 sources，route 无法精确返回它引用了哪条历史消息。需要增加稳定标识：

```python
class RagHistoryMessage(BaseModel):
    message_id: UUID
    reply_to_message_id: UUID | None = None
    generation_run_id: UUID | None = None
    role: Literal["user", "assistant"]
    content: str
    sources: list[RagHistorySource] = Field(default_factory=list)
```

说明：

- `message_id` 用于 route 返回可验证的引用；
- `reply_to_message_id` 用于确定 assistant 对应的 user message；
- `generation_run_id` 用于审计历史回答的生成版本，不作为事实来源；
- `content` 仍受现有 turn 数量和字符预算限制；
- `sources` 在 route 中只用于解析录音范围。

### 4.2 ConversationContext

一期保持克制，只保存明确、稳定且跨轮仍有价值的状态：

```python
class ConversationContext(BaseModel):
    version: int
    active_recording_scope: ActiveRecordingScope | None = None
    response_constraints: ResponseConstraints = Field(default_factory=ResponseConstraints)
    confirmed_decision_refs: list[ConfirmedDecisionRef] = Field(default_factory=list)


class ActiveRecordingScope(BaseModel):
    recording_ids: list[UUID]
    source_message_ids: list[UUID]
    source: Literal["user_explicit", "history_sources", "mixed"]


class ResponseConstraints(BaseModel):
    answer_style: Literal["concise", "detailed"] | None = None
    requested_format: str | None = None
    recordings_only: bool = True


class ConfirmedDecisionRef(BaseModel):
    decision_type: Literal["transcription_correction"]
    recording_id: UUID
    chunk_id: UUID
    source_message_id: UUID
```

一期不在 `ConversationContext` 中保存：

- 从历史自动抽取的 entities；
- assistant 历史事实结论；
- Evidence 正文；
- 旧的 Evidence grade；
- 完整历史筛选项；
- 每轮的 `history_usage` 或 `resolved_query`。

确认过的纠偏内容仍由原有 adjudication / source 数据保存，`confirmed_decision_refs` 只保存引用，不复制纠偏实体。

### 4.3 TurnContext

`TurnContext` 由 route 为当前 generation 产生，并作为不可变快照绑定到 `generation_run_id`：

```python
HistoryUsage = Literal[
    "none",
    "reference_resolution",
    "continue",
    "compare",
    "edit",
    "summarize",
]


class ReferencedTurn(BaseModel):
    user_message_id: UUID
    assistant_message_id: UUID | None = None


class ResolvedReference(BaseModel):
    expression: str
    resolved_value: str
    source_message_id: UUID


class TurnContext(BaseModel):
    history_usage: HistoryUsage
    referenced_turns: list[ReferencedTurn] = Field(default_factory=list)
    history_recording_ids: list[UUID] = Field(default_factory=list)
    recording_scope_source: Literal["none", "user_explicit", "history_sources", "mixed"] = "none"
    resolved_query: str
    resolved_references: list[ResolvedReference] = Field(default_factory=list)
    response_constraints: ResponseConstraints = Field(default_factory=ResponseConstraints)
```

字段语义：

| 字段 | 含义 |
| --- | --- |
| `history_usage` | 本轮为何需要历史，决定后续是否加载原文以及加载什么内容。 |
| `referenced_turns` | route 认定与当前问题相关的历史问答。允许多轮，不能限制为最近一轮。 |
| `history_recording_ids` | 本轮从所引用历史 assistant sources 中继承的录音范围。 |
| `recording_scope_source` | 说明录音范围来自当前用户明确输入、历史 sources 或二者组合。 |
| `resolved_query` | 完成历史指代和省略解析、面向最终回答的问题。它与面向检索的 `content_query` 分开。 |
| `resolved_references` | 保存“它 → I²C”这类解析及其来源，供审计和评测。 |
| `response_constraints` | 本轮生效的回答风格、格式和证据边界。 |

`history_recording_ids` 是本轮解析后的物化范围，因此一期不保存完整历史筛选项。若未来需要支持“继承上轮动态条件并按当前录音库重新计算”，再引入 `inherited_scope_expression`，不能直接长期复用可能已经过期的 `ResolvedFilters`。

## 5. route 协议

### 5.1 输入

route 接收：

```text
current_query
+ ConversationContext（可为空）
+ 最近有限历史 RagHistoryMessage[]
+ 当前 API 显式 scope_recording_ids
```

route 不接收：

- 预提取 entities；
- 检索 terms / phrases；
- 历史 Evidence 正文；
- 历史 assistant answer 的可信事实摘要。

### 5.2 输出

现有 `RagRoute` 增加 `turn_context`：

```python
class RagRoute(BaseModel):
    status: RouteStatus
    strategy_id: StrategyId | None = None
    content_query: str | None = None
    recording_limit: int | None = None
    recording_rank: int | None = None
    time_range: TimeRange | None = None
    inferred_filters: InferredFilters = Field(default_factory=InferredFilters)
    turn_context: TurnContext | None = None
    error_code: RouteErrorCode | None = None
    reason: str = ""
```

`content_query` 与 `resolved_query` 的边界：

- `content_query` 服务正文检索和证据判断，可移除已经结构化表达的录音范围条件；
- `resolved_query` 服务最终回答，保留用户回答意图，并显式解析必要的历史指代；
- 两者都不能增加当前问题与历史上下文均不存在的事实。

### 5.3 route 判断规则

| 当前问题 | `history_usage` | route 行为 |
| --- | --- | --- |
| 独立问题，不依赖历史 | `none` | 不引用历史消息，不向 answer 加载历史原文。 |
| “它的时延呢” | `reference_resolution` | 输出 resolved query 和引用来源，不加载历史原文。 |
| “继续说刚才的风险” | `continue` | 引用相关轮次；后续可加载必要的历史内容。 |
| “比较你刚才说的两个方案” | `compare` | 引用一个或多个相关轮次。 |
| “把上一版改得简短一些” | `edit` | 指定 assistant message，原文作为待编辑对象。 |
| “总结我们刚才讨论的内容” | `summarize` | 指定需要总结的多轮消息范围。 |

route 负责“选哪些历史消息、为什么使用”；它不负责从数据库加载全文，也不直接把历史拼入 answer prompt。

## 6. route 输出校验

模型输出必须经过确定性校验：

1. `referenced_turns` 中的 message ID 必须存在于 route 输入历史；
2. assistant message 必须通过 `reply_to_message_id` 与对应 user message 配对；
3. `history_recording_ids` 只能来自被引用 assistant message 的 sources，或当前 API 明确 scope；
4. 所有录音 ID 必须重新经过当前 Workspace 权限校验；
5. `resolved_references.source_message_id` 必须属于 `referenced_turns`；
6. `history_usage = none` 时，`referenced_turns`、`history_recording_ids` 和 `resolved_references` 必须为空；
7. `reference_resolution` 不允许触发历史原文加载；
8. `edit` 必须引用至少一条 assistant message；
9. `summarize` 必须引用至少一轮历史消息；
10. 校验失败时不能退化为加载全部历史，应返回澄清或安全的 `none`。

## 7. 历史原文加载

### 7.1 唯一存储位置

历史原文继续保存在：

```text
conversation_messages.content_blocks
```

不新增 context 正文副本。assistant 的 `AGGRE_MSG` 已经包含 original / corrected 子版本；加载时必须根据用途选择版本，不能用字符串拼接猜测。

默认规则：

- 普通继续、比较和总结使用该消息当时的 primary sub-message；
- 编辑指定版本时由本轮请求或 route 明确版本；
- 引用历史 assistant 内容不自动加载其 Evidence；
- 历史 source 只用于范围解析或重新定位当前证据。

### 7.2 load_history_context 节点

在 route 校验后增加确定性节点：

```text
route
  ↓
validate_turn_context
  ↓
load_history_context
  ↓
strategy / retrieval
  ↓
answer
```

`load_history_context` 根据 `history_usage` 执行：

| `history_usage` | 是否加载原文 | 加载内容 |
| --- | --- | --- |
| `none` | 否 | 空。 |
| `reference_resolution` | 否 | answer 只接收 `resolved_query`。 |
| `continue` | 是，按需 | 被引用轮次中与承接有关的 user / primary assistant 内容。 |
| `compare` | 是 | 被比较的指定消息内容，保留 message ID。 |
| `edit` | 是 | 指定 assistant message 的目标版本原文。 |
| `summarize` | 是 | 被 route 选择的 user / assistant 消息集合。 |

节点必须：

- 只读取当前 conversation 内被 route 引用的消息；
- 保持原始时间顺序；
- 执行独立字符 / token 预算；
- 输出带 message ID、role 和用途标记的结构，不输出无来源的拼接文本；
- 超出预算时按完整 turn 裁剪或生成明确标注的历史内容摘要，不能截断后伪装成完整原文。

建议契约：

```python
class LoadedHistoryMessage(BaseModel):
    message_id: UUID
    role: Literal["user", "assistant"]
    content: str
    usage: Literal["conversation_context", "artifact"]


class LoadedHistoryContext(BaseModel):
    usage: HistoryUsage
    messages: list[LoadedHistoryMessage] = Field(default_factory=list)
    truncated: bool = False
```

## 8. answer 输入与证据隔离

answer 输入应分区：

```text
当前用户问题：
{current_query}

解析后的问题：
{turn_context.resolved_query}

本轮会话上下文：
{turn_context 中不含事实的字段}

历史内容（仅在 usage 要求时存在；是对话或操作对象，不是录音事实证据）：
{loaded_history_context}

当前录音证据（回答录音事实的唯一依据）：
{strategy_result.answer_context}
```

answer 规则必须明确：

- 历史 assistant 内容可能来自旧版 ASR、Embedding、索引或检索策略；
- 不得仅依据历史 assistant 内容断言录音事实；
- 历史中的“无法确认”“没有找到”只描述旧运行结果；
- 若历史内容与当前 Evidence 冲突，以当前 Evidence 为准；
- 只有当前 Evidence 不足时，才允许根据本轮 Evidence 状态输出“无法确认”；
- 当 `history_usage = edit` 时，历史原文是编辑对象，但其中事实仍不因此获得可信度；
- 当 `history_usage = summarize` 且用户要求总结“对话本身”时，可以总结历史说过什么，但应避免把历史 assistant 断言改写成已经验证的录音事实。

## 9. ConversationContext 更新

route 只生成本轮 `TurnContext`，不直接修改持久 `ConversationContext`。上下文更新由独立、可审计的 updater 在确定事件发生后执行。

### 9.1 用户消息到达

- 读取当前 `ConversationContext`；
- route 生成本轮 `TurnContext`；
- 校验后将 `TurnContext` 绑定当前 generation；
- 本轮 retrieval 和 answer 使用同一不可变快照。

### 9.2 用户明确确认或修改

以下内容可以立即更新 `ConversationContext`：

- 用户明确指定继续使用的录音范围；
- 用户确认的转写纠偏决定引用；
- 用户明确给出的长期回答风格或格式偏好。

### 9.3 assistant 回答完成

允许更新：

- 由本轮可靠解析得到、且后续可继续使用的 active recording scope；
- 相关 source message IDs；
- generation / message 的执行引用。

禁止更新：

- assistant 回答中的自由文本事实；
- “无法确认”“没有找到”等检索结论；
- Evidence grade；
- 本轮模型推断但用户未确认的领域事实。

### 9.4 失效与重算

- regenerate / resume 使用其对应的 `TurnContext` 快照或按产品语义重新 route；
- 用户编辑或删除被引用历史消息时，使依赖它的 context projection 失效；
- 录音权限变化后必须重新校验 `active_recording_scope`；
- Embedding、索引、chunk、ASR 或 adjudication 版本变化不会修改历史消息，但所有新事实回答仍重新走当前检索和 Evidence；
- 旧 assistant answer 永远不能因保存在 context 中而升级为事实。

## 10. 持久化建议

一期可采用两个 JSONB 快照：

```text
conversations
  ├── context_payload jsonb
  └── context_version integer

generation_runs 或 conversation_messages
  └── turn_context_payload jsonb
```

选择 `generation_runs` 时，每次 regenerate 都有独立的 route 快照，审计更准确；选择 assistant `conversation_messages` 时，读取一轮最终上下文更直接。建议以 `generation_runs` 为执行事实来源，并在 assistant message 需要时保存最终 run 引用。

所有更新使用 context version 做乐观并发控制。同一会话一期仍只允许一轮活跃 generation，但版本控制可以避免重试、恢复和未来并发扩展覆盖较新的状态。

## 11. 与 checkpoint 的关系

`TurnContext` 是本轮工作流输入的一部分，必须进入 RAG checkpoint 序列化。恢复时：

- 同一个 run 的普通断点恢复复用已校验的 `TurnContext`；
- 从旧 generation 重新生成时，根据产品选择决定复用还是重新 route；
- 如果被引用消息已编辑、删除、失去权限或 conversation context version 不兼容，必须重新 route；
- `LoadedHistoryContext` 可以按 message ID 重新加载，不需要把历史正文复制进 checkpoint。

## 12. 可观测性与评测

每轮至少记录以下非正文元数据：

```text
history_usage
referenced_user_message_count
referenced_assistant_message_count
history_recording_count
history_original_loaded
loaded_history_chars
resolved_reference_count
turn_context_validation_status
conversation_context_version
```

离线评测至少覆盖：

1. 独立事实问题：route 输出 `none`，answer 不含历史原文；
2. 单一指代追问：输出 `reference_resolution`，正确解析对象但不加载原文；
3. 多录音指代：可引用多个历史 sources，不默认只取最近录音；
4. 指代不清：返回澄清，不能加载所有历史；
5. 继续与比较：只加载 route 选中的消息；
6. 编辑上一版：历史 assistant 原文作为 artifact 进入 answer；
7. 对话总结：加载指定消息，但历史 assistant 结论不冒充当前 Evidence；
8. 旧 embedding 未召回、新 embedding 已召回：旧“无法确认”不得覆盖当前证据；
9. 历史 assistant 与当前 Evidence 冲突：最终答案服从当前 Evidence；
10. 权限变化：历史 recording ID 被当前授权过滤；
11. 历史消息删除或编辑：context 失效并重新 route；
12. 超预算：按完整 turn 裁剪，并正确记录 `truncated`。

## 13. 分阶段落地

### 阶段一：阻断默认 history 污染

1. 为 `RagHistoryMessage` 增加 message ID 和问答关联字段；
2. 扩展 `RagRoute`，输出 `TurnContext`；
3. answer 不再默认读取完整 `_history_text(state["history"])`；
4. `none` 和 `reference_resolution` 只向 answer 传 route 解析结果；
5. 增加旧“无法确认”与新 Evidence 冲突的回归评测。

### 阶段二：按需加载历史原文

1. 增加 `validate_turn_context`；
2. 增加 `load_history_context`；
3. 支持 continue、compare、edit 和 summarize；
4. 引入历史原文独立 token 预算和 answer prompt 分区。

### 阶段三：持久 ConversationContext

1. 增加会话级 context projection；
2. 保存明确录音范围、用户约束和确认决定引用；
3. 增加版本、失效和并发控制；
4. 根据评测结果决定是否引入更复杂的动态 scope expression，仍不保存 assistant 自由文本事实。

## 14. 最终边界

本设计最终形成以下职责：

```text
conversation_messages
  保存完整历史原文和历史 sources

ConversationContext
  保存当前仍有效的跨轮状态，不保存 assistant 事实

route
  判断哪些历史轮次相关、为何使用，并解析本轮问题与录音范围

TurnContext
  保存 route 对本轮 history 使用方式的不可变快照

load_history_context
  按 message ID 和 usage 读取必要原文

expand_retrieval_terms
  从 content_query 提取检索关键词，不由 history context 替代

retrieval / adjudication / grade
  生成本轮当前 Evidence

answer
  使用 resolved query、按需历史 artifact 和当前 Evidence；录音事实以当前 Evidence 为准
```

核心原则是：历史可以影响当前问题的理解、范围、任务连续性和操作对象，但历史 assistant 回答不能未经当前 Evidence 验证就成为当前录音事实。
