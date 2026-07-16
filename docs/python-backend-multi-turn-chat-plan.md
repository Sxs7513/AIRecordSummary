# Python 后端：多轮录音问答与前端聊天方案

## 1. 目标与边界

将当前只有一个 `askedQuery`、`runId` 和临时回答的 `RagChat`，升级为可持久化、可分页、可恢复的多轮聊天。会话内的每条回答仍使用项目既有的 Generation block / HTTP SSE 协议；**不新增 `ragClient`**，也不把路由、检索计划和模型推理过程暴露给前端。

核心边界：

- `conversation` 与 `conversation_message` 是长期业务数据，负责会话、排序、正文和引用的最终持久化。
- `generation_run` 是一条助手消息的一次可恢复流式执行，负责 `seq`、快照、事件、取消和连接状态。
- `GenerationStreamClient` 是前端唯一的流式 client；聊天只在它之上增加消息与会话状态。
- RAG 每轮都按当前用户的录音权限检索。历史消息帮助理解追问，但不构成事实证据。

本方案依赖 [通用生成任务与流式消息设计](generation-streaming-design.md)，不重复定义 block、SSE envelope 或断线续传协议。

## 2. 后端领域模型

```text
workspace
  └── conversation
        └── conversation_message (user / assistant)
              └── generation_run (仅 assistant，生成期间存在)
```

```text
conversations
  ├── id
  ├── workspace_id
  ├── owner_user_id
  ├── title
  ├── next_message_sequence  每次写入消息时原子递增的会话内计数器
  ├── archived_at
  ├── created_at
  └── updated_at

conversation_messages
  ├── id
  ├── conversation_id
  ├── role                  user / assistant
  ├── sequence              会话内严格递增的展示顺序
  ├── reply_to_message_id   assistant 指向其回复的 user message；user 为 null
  ├── content_blocks        标准 ContentBlock[] JSONB
  ├── sources               assistant 的最终引用 JSONB
  ├── generation_run_id     assistant 生成期间关联的 run，可为空
  ├── status                pending / streaming / completed / failed / cancelled
  ├── client_message_id     用户发送动作的幂等键
  ├── created_at
  └── updated_at
```

约束与索引：

- 一轮问答固定写入两条消息：一条 `role = user` 的问题，以及一条 `role = assistant` 的回答占位。`role` 就是上下行消息的标准区分字段，不再增加语义重复的 direction 字段。
- `unique(conversation_id, client_message_id)`：浏览器超时或重试不会创建重复的一问一答。
- `generation_run_id` 唯一：一个 Generation 只能驱动一条助手消息。
- `unique(conversation_id, sequence)`：消息只按 `sequence ASC` 展示和分页；`created_at` 仅用于显示时间，不能承担顺序语义。
- `reply_to_message_id` 仅用于 assistant，指向对应 user message，使一轮问答的配对关系可直接读取。
- 会话访问要求当前用户仍是所属 Workspace 成员。创建、读取、发送、取消和 SSE 均从消息回溯到该 Workspace 重新授权。

`generation_runs.parent_type = "conversation_message"`、`parent_id = assistant_message.id` 只用于运行关联；授权范围仍写入 generation 的通用 `access_scope`，不能从客户端传入。

## 3. 发送一轮消息

`POST /api/conversations/{conversation_id}/messages` 的请求体：

```json
{
  "content_blocks": [{ "type": "text", "value": "刚才提到的良率风险是什么？" }],
  "client_message_id": "浏览器生成的 UUID"
}
```

服务端在同一事务内：

1. 锁定 conversation，校验 Workspace 访问权，并检查是否已经存在未终态的 assistant message；首版存在时返回 409，保证同一会话只有一轮正在回答；
2. 先按 `client_message_id` 查询幂等结果；已存在则返回既有 user / assistant message 与 run；
3. 原子地将 `conversations.next_message_sequence` 增加 2，得到连续的 `N`、`N + 1`；不能使用 `max(sequence) + 1`；
4. 写入 `sequence = N` 的 user message；
5. 写入 `sequence = N + 1`、`reply_to_message_id = user_message.id` 的 `pending` assistant message；
6. 创建 `parent = conversation_message` 的 Generation，并将 `generation_run_id` 写回 assistant message；
7. 提交事务后启动 RAG 工作流；
8. 返回两条规范化消息和 `generation_run_id`。

RAG 工作流开始时将助手消息置为 `streaming`；Generation 成功、失败或取消时，在同一最终化事务中把 `content_blocks`、`sources` 和状态写入助手消息。Generation 表继续保存同一输出的快照与事件，用于流式恢复；聊天表是页面刷新后的长期读取来源。

请求重试命中同一 `client_message_id` 时，返回已创建的 user / assistant message 与既有 run，不重复提交模型任务。

## 4. 多轮 RAG 与授权

每轮输入由下列内容组成：

```text
当前用户问题
+ 最近 8～12 条可见聊天消息（受字符 / token 上限约束）
+ 当前用户可访问的录音范围
```

LangGraph：

- `route` 参考有限历史，处理“刚才那个”“它”“第二个方案”等追问。每条 assistant 历史消息按原问答顺序附带其 `sources` 中的录音 ID、标题和时间范围；route 只使用这些字段，不把旧 Evidence / chunk 原文送入模型；这些 source 只供 route 判断录音范围；
- `retrieve` 只在当下的数据库授权范围内检索 Evidence；
- `grade`、`plan` 与 `validate_plan` 校验证据；
- `stream_answer` 仅依据已验证的本轮 Evidence 和有限对话上下文，向该 assistant message 对应的 Generation 写入文本 delta。

历史消息不能替代事实证据。若用户后来失去某条录音权限，下一轮立即排除该录音；旧消息保留，但打开旧 source 或音频时重新校验权限，并显示“该引用当前不可访问”。会话很长后再引入可验证的会话摘要，不能无限拼接全文。

当范围既可能指对话中的历史 source，也可能指录音库按上传时间排序的录音，而 route 无法仅依靠语义与上下文确定时，它返回 `reason = "ambiguous_recording_scope"`。服务端立即生成澄清消息，不进行宽泛检索；这不是前端关键词匹配，也不新增 route schema 字段。

## 5. API

```text
GET  /api/conversations?cursor=...&limit=...
POST /api/conversations
GET  /api/conversations/{id}/messages?before=...&limit=50
POST /api/conversations/{id}/messages

GET    /api/generations/{run_id}
GET    /api/generations/{run_id}/events
DELETE /api/generations/{run_id}
```

`GET /messages` 返回：

```json
{
  "items": [
    {
      "id": "message-id",
      "role": "assistant",
      "content_blocks": [{ "type": "text", "value": "..." }],
      "sources": [],
      "generation_run_id": "run-id",
      "status": "streaming",
      "created_at": "..."
    }
  ],
  "next_before": "opaque-cursor",
  "has_more": true
}
```

首屏读取最近一页，前端按 `sequence ASC` 展示。向前分页的 cursor 以最早一条消息的 sequence 为边界，而不是时间戳。若页面发现 `pending` 或 `streaming` 的 assistant message，使用其 `generation_run_id` 连接统一 SSE。服务端首次发送 `snapshot`，因此刷新后先得到已生成全文，再接收新增 delta。

## 6. 前端目录与职责

```text
app/
  conversations/
    page.tsx                         # 会话列表 / 新建入口
    [conversationId]/page.tsx        # 服务端读取首屏会话元数据
  sdk/
    generation/                      # 现有通用 SSE client、transport、store、selectors
    conversations/
      types.ts                       # Conversation、Message、分页与 API DTO
      client.ts                      # 会话和消息 HTTP 请求；不是流式 client
      store.ts                       # Zustand：长期消息、分页、发送状态
      selectors.ts                   # 将 message + generation state 组合为展示模型
components/
  conversation-shell.tsx             # 页面装配、加载、恢复活跃 stream
  conversation-list.tsx              # 会话侧栏与分页
  message-list.tsx                   # 消息滚动、上拉加载、自动滚动策略
  message-item.tsx                   # user / assistant 气泡、状态和 sources
  chat-composer.tsx                  # 输入、提交、取消与错误提示
```

`sdk/conversations/client.ts` 只处理 REST：获取会话、分页获取消息、发送一轮消息。流式连接仍只由 `GenerationStreamClient` 执行，避免出现两个 SSE transport、两套重连语义或两个协议 reducer。

### 6.1 聊天路由与页面框架

聊天是独立工作区，不复用全局“页面切换”侧栏：

```text
/chat                    新建对话态
/chat/{conversationId}   已有会话态
```

- `/chat` 左侧只展示已有对话列表及“新建对话”入口；右侧不预先创建空会话或空消息，底部居中显示输入框。
- `/chat/{conversationId}` 保持相同的左侧对话列表；右侧展示该会话的消息与固定在底部的输入框。
- 聊天页面顶部右侧只提供“返回录音管理”按钮，不展示全局录音/账号等页面切换入口。
- “新建对话”只导航到 `/chat`，不立刻向数据库插入空 conversation；用户从未发送时不会遗留空会话。

## 7. 消息列表设计

### 7.1 展示模型

列表 key 必须是服务端 `message.id`，不能使用数组下标或 `generation_run_id`；列表数据按 `sequence ASC` 排列。每条消息展示：

- user：`content_blocks` 渲染出的正文、发送失败时的重试入口；
- assistant：正文、`pending / streaming / completed / failed / cancelled` 状态、阶段文案、取消或重新发送入口；
- assistant sources：仅在该条消息的 `sources` 非空时展示，绑定到该回答下方；
- 不向用户展示 route、检索 query、评分、向量分数或模型日志。

正文使用标准 `ContentBlock[]` 渲染。当前只存在 `text` block，selector 合并其 `value`；未来新增引用卡、图片或工具结果时，由 `MessageContent` 按 discriminated union 渲染，不需要改变消息列表或 SSE client。

### 7.2 标准聊天布局

消息使用标准聊天机器人双侧布局：

- user message 使用 `flex` 行的 `justify-end`，气泡位于右侧；使用主题色或浅色背景，右下圆角略小；正文之外不附带 sources 等复杂信息。
- assistant message 使用 `justify-start`，气泡位于左侧，可在左侧配 AI 头像；正文采用中性背景和 Markdown 渲染，sources、阶段文案、失败重试与停止生成操作均附在该消息下方。
- 同一 role 的连续消息缩小垂直间距；一轮 user → assistant 之间保留更明显的间距，强化问答关系。
- `pending` / `streaming` 的 assistant 先渲染左侧占位与 Generation `phase`，随后填充正文，避免空内容被误判为失败。
- 输入框固定在页面底部，消息列表独立滚动。仅当用户距离底部约 80px 以内时自动跟随新消息；否则显示“回到最新消息”按钮。
- 移动端保持 user 右、assistant 左的方向，气泡最大宽度提高到 85～90%。

### 7.3 分页与滚动

- 首次加载最近 50 条消息，按 `sequence ASC` 插入列表底部；
- 滚动到顶部时用 `next_before` 拉取更早一页，插入前记录 `scrollHeight`，插入后补偿滚动差，避免视图跳动；
- 新消息或 delta 到达时，只有用户原本距离底部不超过约 80px 才自动滚到底部；否则显示“有新消息”浮动按钮；
- 单条 markdown 高度会随 delta 改变，因此首版不急于虚拟列表。消息超过数百条后再采用支持动态高度和顶部锚定的虚拟化方案。

### 7.4 提交与乐观状态

用户点击发送后：

#### 新建对话：`/chat`

首条消息使用原子接口，例如 `POST /api/conversations/turn`。服务端在一笔事务中创建 conversation、user message、assistant placeholder 和 Generation，返回完整 turn 与 `conversation_id`。前端收到成功响应后：

1. 将会话和两条消息写入 Zustand；
2. 将左侧列表乐观加入新会话，并以首条问题更新标题；
3. 跳转 `router.replace('/chat/{conversationId}')`；
4. 使用返回的 `generation_run_id` 连接统一 Generation SSE。

不能先单独创建空 conversation 再发送，以免网络失败或用户离开时留下无内容会话。

#### 已有对话：`/chat/{conversationId}`

1. 生成 `client_message_id`；
2. 在 `conversationStore` 立即插入临时 user message 和 `pending` assistant placeholder，路由保持不变；
3. 调用 `POST /api/conversations/{id}/messages`；
4. 用响应中的服务端 message ID 替换临时项，并绑定 `generation_run_id`；
5. `GenerationStreamClient.connect(runId)`；
6. 请求失败时把临时 user message 标为 `failed`，保留原文和“重试”按钮；重试复用同一 `client_message_id`。

服务端即使已经返回 user message，也可能暂未返回首 token。因此 placeholder 必须独立显示“正在检索录音资料”或 Generation `phase`，不能把空正文误显示为失败。

## 8. Zustand Store 设计

### 8.1 两层状态，职责不重复

现有 `app/sdk/generation/store.ts` 继续是协议层唯一状态源：按 `runId` 保存 `snapshot`、blocks、sources、`lastSequence`、连接状态，并用纯 `reduceGenerationEvent` 做去重和顺序消费。

新增 `conversationStore` 管理长期会话与页面交互，而不是重新实现 SSE reducer：

```ts
type ConversationStore = {
  conversationsById: Record<string, Conversation>;
  messagesById: Record<string, ConversationMessage>;
  messageIdsByConversation: Record<string, string[]>;
  paginationByConversation: Record<string, MessagePagination>;
  sendingByClientMessageId: Record<string, SendState>;

  hydrateMessages: (conversationId: string, page: MessagePage) => void;
  prependOlderMessages: (conversationId: string, page: MessagePage) => void;
  insertOptimisticTurn: (input: OptimisticTurn) => void;
  reconcileTurn: (response: SendMessageResponse) => void;
  failTurn: (clientMessageId: string, error: string) => void;
  markMessageTerminal: (messageId: string, status: MessageStatus) => void;
};
```

`messagesById` 保存 API 返回的持久化 blocks、sources、状态、`sequence` 和 `generationRunId`；`messageIdsByConversation` 只保存按 `sequence ASC` 排列的 ID。这样更新某条流式消息不会重建整页数组，分页去重也只需按 message ID 完成。

流式正文的展示不在 `conversationStore` 复制一份。`selectMessageView(messageId)` 的规则是：

1. 若消息有 `generationRunId` 且 Generation store 存在该 run，优先读取该 run 的 `blocks`、`sources`、`status` 和 `phase`；
2. 否则读取 `conversation_messages` API 返回的持久化值；
3. Generation 到达终态后不立即清除 run 缓存；下一次历史刷新得到已固化消息后再安全回收。

这避免聊天页面与通用 SDK 各自累计 delta 而发生重复或序号错乱。

### 8.2 流式渲染：攒包与逐字显示

Generation SSE 的网络分片粒度不等于 React 的合适更新粒度。一个模型 token 或一次网络读取都调用 Zustand `set`，会在长回答时导致大量 React render 和 Markdown 重解析；但前端也不能因此合并或丢弃 SSE 的 `seq`。

#### 攒包：`GenerationStreamClient` 负责

- `content.delta` 到达后，不立即调用 store 的 `consume`；按 `runId` 追加到 client 私有的 pending event 队列，并请求一次 `requestAnimationFrame`。
- 同一动画帧内到达的多个 delta 由 `generationStore.consumeMany(events)` 在**一次** Zustand `set` 中按原 `seq` 顺序依次执行 `reduceGenerationEvent`。这保留去重、顺序校验和所有 ContentBlock，不把多个 SSE event 伪造成一条 event。
- `snapshot`、`phase`、`run.status`、`output.final`、`run.error`、`run.cancelled` 等非 delta 事件必须先同步 flush 该 run 的 pending delta，再立即消费自身，避免终态先于文本显示。
- 每个 `runId` 独立缓冲和调度；SSE 断线重连的 snapshot / 重放 event 仍走同一 reducer，不能依赖浏览器内存队列保持正确性。
- `requestAnimationFrame` 在后台标签页可能被显著限流，因此额外设置一个短的超时兜底（例如 100ms）强制 flush。它只保证状态及时推进，不改变按帧优先的渲染策略。
- client 被关闭、组件卸载或 run 终态时，取消该 run 的 frame / timer，并先 flush 已收到的数据；避免最后一小段回答丢失。

攒包后的 Generation store 仍保存“已接收的完整正文”，是消息恢复、重连和持久化对账的唯一前端事实来源。

#### 打字机效果：`MarkdownContent` 负责

`MarkdownContent` 接收完整的 canonical Markdown 文本，但只对处于 `streaming` 的 assistant 消息启用逐字显示：

1. 用 `Intl.Segmenter("zh-CN", { granularity: "grapheme" })` 将目标文本切为用户感知字符，避免 emoji、组合字符被切坏；没有该 API 时退化为 `Array.from(text)`。
2. 组件内部维护 `renderedLength` / `renderedText`，目标文本增加后启动自己的 `requestAnimationFrame`；每一帧最多前进一个 grapheme，因此视觉上是一个个字出现。
3. 每帧只更新 `MarkdownContent` 自己的本地 state；不更新 Zustand，也不修改 `content_blocks`。网络攒包与打字节奏完全解耦。
4. 历史消息、首次 snapshot、页面刷新恢复以及非 streaming 的终态消息直接显示完整文本，不从头重放打字动画。流式消息收到终态时继续消化已接收字符；若用户切换页面或消息不再可见，重新进入后直接显示完整文本。
5. 渲染器对当前的 `renderedText` 使用既有的安全 Markdown 白名单与 GFM 插件。Markdown 标记可能在逐字过程中暂时不完整，这是正常表现；不使用 `dangerouslySetInnerHTML`，也不把 think / 内部推理文本交给前端过滤。

因此数据流为：

```text
HTTP SSE event
  → GenerationStreamClient 按帧攒包
  → Generation store（完整、可恢复的 Markdown）
  → MarkdownContent（本地逐字可见 Markdown）
```

### 8.3 Stream 绑定与恢复

`ConversationStreamManager` 是 `conversation-shell` 使用的薄协调层，不是新的 client：

```ts
for (const message of activeAssistantMessages) {
  if (message.generationRunId) generationClient.connect(message.generationRunId);
}
```

它维护进程内 `Map<runId, { client, refCount }>`：同一 run 在 React Strict Mode、侧栏预览和详情页同时出现时只创建一个连接；最后一个订阅者离开后才关闭。连接回调仍只写 `generationStore`。

页面刷新恢复顺序：先拉取消息历史，再扫描 `pending/streaming` message 并连接 run。无 cursor 的 SSE 会先给 snapshot；reducer 替换 blocks 后再按 `seq` 接受新增 delta。网络断开则由现有 HTTP SSE transport 携带 `Last-Event-ID` 续传，聊天 store 无需理解网络序号。

### 8.4 派生 selector

```ts
type MessageView = {
  id: string;
  role: "user" | "assistant";
  blocks: ContentBlock[];
  sources: Source[];
  status: MessageStatus;
  phase: GenerationPhase | null;
  isStreaming: boolean;
  error: string | null;
};
```

`MessageList` 只订阅当前 conversation 的 ID 列表和每个 `MessageView`，而不订阅整个 `runs` 对象。这样任意其他总结或问答 run 的 delta 不会让聊天列表整体重新渲染。

## 9. 实施顺序与验证

1. 创建 `conversations`、`conversation_messages` 及访问服务，补充会话 / 消息 API 与游标分页。
2. 将 RAG 创建流程改为“写 user + assistant message + generation”，并在终态投影助手消息。
3. 让 LangGraph 读取受限的近期历史；所有检索 SQL 继续强制使用授权范围。
4. 新建 `sdk/conversations`、会话页与消息列表；保留并复用现有 Generation SDK。
5. 实现 optimistic turn、stream manager、刷新恢复、顶部翻页和自动滚动策略。
6. 覆盖以下测试：重复发送幂等、并发发送的 sequence 连续性、同会话进行中回答的拒绝策略、会话越权、SSE 越权、权限撤销后的再问、网络续传、刷新恢复、分页去重、流式 delta 不影响其他消息、同帧 delta 仅触发一次 store 更新、终态前会 flush 攒包、跨分片 Markdown 的逐字显示不切坏 emoji。
