# 通用生成任务与流式消息设计

## 1. 目标与范围

系统中的本地 LLM 生成不能分别为录音总结、录音问答各自实现一套流式逻辑。它们都应使用统一的 **Generation** 基础设施：

- 录音总结是后台、可恢复的生成任务；
- 录音问答是交互、低延迟的生成任务；
- 后续报告生成、搜索摘要和文本改写也可以复用相同的执行、流式传输和持久化能力。

本设计不实现鉴权和会话授权，但数据模型预留父对象关联，后续可关联用户、会话和消息。

旧 Node RAG 接口已经有 `evidence`、`answer_delta`、`answer_done` 和 `error` 的一次性 SSE 输出。本设计保留其用户可见能力，但将其迁入 Python，并补齐断线恢复、页面刷新恢复、取消、优先级和可观测性。

## 2. 总体架构

```text
录音总结 stage / 问答创建请求
             │
             ▼
      GenerationService
             │ 创建 generation_run
             ▼
      GenerationExecutor
             │ llama-cpp / 其他 provider 的 token stream
             ▼
        StreamEmitter
        ├── 进程内缓冲与实时发布
        └── 批量持久化快照与事件
             │
             ├── PostgreSQL generation_runs
             ├── PostgreSQL generation_events
             └── SSE / 前端 Generation SDK
```

职责划分：

- `GenerationService`：创建、查询、取消 Generation Run，负责幂等键和状态转换。
- `GenerationExecutor`：按任务类型调用模型；不直接操作 HTTP/SSE。
- `StreamEmitter`：将 provider 的 token、阶段变化和最终结果转换为统一事件，并批量持久化。
- `GenerationEventStore`：保存可恢复快照和顺序事件。
- `GenerationStreamHub`：同一进程内把新事件即时通知 SSE 订阅者。
- 前端 SDK：恢复快照、按序消费增量、处理连接状态；不包含具体问答或总结页面 UI。

当前 Python 代码按这个边界收敛到独立模块，而不是散落到 pipeline runtime 或 API route：

```text
backend/src/generation/
├── contracts.py  # run、事件 envelope、block 与状态的类型契约
├── store.py      # PostgreSQL 快照、事件与 sequence 的原子读写
├── emitter.py    # 模型回调 -> 批量 content.delta / phase / 终态事件
├── hub.py        # 单进程实时订阅者广播；不保存业务状态
├── service.py    # 调用方唯一入口：创建、查询、取消与 emitter 构造
└── __init__.py

backend/src/api/routes/generations.py  # GET run、HTTP SSE 续传和取消接口
```

`audio_processing/stages/generate_summary.py` 只作为 `recording_summary` 的适配器：创建关联 run、向 `StreamEmitter` 报告阶段和最终文本；它不直接处理 SSE、数据库 sequence 或前端连接。未来的 `rag_answer` 执行器也只需要调用同一 `GenerationService`。

## 3. Generation Run 数据模型

`generation_runs` 是一次生成任务的当前可信快照：

| 字段 | 说明 |
|---|---|
| `id` | Generation Run UUID。 |
| `kind` | `recording_summary`、`rag_answer` 等业务类型。 |
| `priority` | `interactive` 或 `background`。 |
| `parent_type` / `parent_id` | 所属对象；总结关联 `stage_run`，问答后续可关联 `conversation` 或 `message`。 |
| `status` | `queued`、`running`、`succeeded`、`failed`、`cancelled`。 |
| `input_payload` | 类型化输入的 JSON 快照，例如 query、过滤条件或 artifact 引用。 |
| `phase` | 用户可见阶段，例如 `retrieving`、`generating`、`finalizing`。 |
| `progress_percent` | 可选进度。 |
| `output_blocks` | 当前完整消息内容快照，类型为有序 `ContentBlock[]`。 |
| `output_payload` | 最终结构化结果，例如 citations、`not_enough_evidence`。 |
| `last_sequence` | 已成功持久化的最大事件序号。 |
| `first_token_at` | 首个用户可见文本到达时间，用于延迟观测。 |
| `cancel_requested` | 取消请求标志，由执行器在 token / chunk 边界协作检查。 |
| `error_code` / `error_message` | 可展示的失败信息和是否可重试判断所需的稳定错误码。 |
| 时间字段 | `created_at`、`started_at`、`finished_at`、`updated_at`。 |

`generation_events` 是 append-only 的恢复事件表：

| 字段 | 说明 |
|---|---|
| `id` | 数据库主键。 |
| `generation_run_id` | 所属 Generation Run。 |
| `sequence` | 同一 run 内严格递增的序号。 |
| `type` | 事件类型。 |
| `payload` | 事件数据 JSON。 |
| `created_at` | 事件生成时间。 |

必须有唯一约束：`unique (generation_run_id, sequence)`。

一次批量增量写入时，`output_blocks`、run 的 `last_sequence` 与对应批次 `generation_events` 必须在同一事务中更新。这样任意快照都代表“内容已经包含到 sequence N”，服务端可安全从 `N + 1` 续传。

## 4. 创建命令

所有调用方先提交统一的创建命令，实际 `input` 由 `kind` 判别校验。

```json
{
  "kind": "rag_answer",
  "priority": "interactive",
  "idempotency_key": "5eb8d3ca-3a1b-4c1f-98dd-9260fe5cf54c",
  "parent": { "type": "conversation", "id": "optional-conversation-id" },
  "input": {
    "query": "这次会议对良率的结论是什么？",
    "filters": { "recording_ids": ["recording-id"] },
    "output_language": "zh-CN"
  }
}
```

录音总结使用相同外壳：

```json
{
  "kind": "recording_summary",
  "priority": "background",
  "parent": { "type": "stage_run", "id": "summary-stage-run-id" },
  "input": {
    "recording_id": "recording-id",
    "utterances_artifact": { "type": "utterances.final", "uri": "artifacts/..." },
    "strategy": "large_context"
  }
}
```

`idempotency_key` 防止浏览器、代理或 SDK 重试时重复创建问答任务。总结由流水线创建时，以 `stage_run_id + attempt_count` 作为稳定幂等来源；同一次 stage 尝试重复进入时复用同一个 Generation Run，而 stage 重试会生成一条新的可追溯 run。

## 5. 流事件协议

每条 SSE 事件使用同一 envelope：

```json
{
  "v": 1,
  "run_id": "generation-run-id",
  "seq": 18,
  "type": "content.delta",
  "at": "2026-07-18T10:20:30.456Z",
  "data": {}
}
```

- `v` 是协议版本。
- `seq` 是恢复和去重的唯一顺序依据，不依赖时间戳。
- SSE 的 `id` 使用 `seq`。前端 HTTP SSE transport 会在重连请求中主动发送 `Last-Event-ID`。
- 不暴露模型的原始思维链。用户只接收安全的阶段标签，不使用 `thinking` 事件。

### 5.1 Block 协议

所有用户可见内容都以 block 表示；事件不会在顶层 `data` 中单独定义 `text` 字段。第一版只定义一种 block：`text`。它的 `value` 就是本次 delta 的文本内容：

```json
{
  "type": "text",
  "value": "会议认为良率问题主要来自……"
}
```

对应 TypeScript 契约：

```ts
type TextBlock = {
  type: "text";
  value: string;
};

type ContentBlock = TextBlock;
```

这里的 block 是**内容片段**，不是带独立持久化 ID 的 UI 卡片。问答正文和总结正文当前都只是一串按顺序追加的 `TextBlock`。检索阶段、排队状态、错误和最终引用仍由控制事件 / 最终结构化输出表达，不伪装成文本内容。

以后需要图片、引用卡片、工具调用结果或音频片段时，再给 `ContentBlock` 增加新的 discriminated union 成员，例如 `{ type: "citation", value: ... }`；不会改变事件 envelope、sequence 或 SDK 连接逻辑。

### 5.2 事件类型

| type | `data` | 说明 |
|---|---|---|
| `run.status` | `{ "status": "running" }` | 排队、运行和终态变化。 |
| `phase` | `{ "name": "retrieving", "label": "正在检索录音资料" }` | 用户可见阶段。 |
| `evidence` | `{ "items": [...] }` | 问答检索出的证据。 |
| `content.delta` | `{ "blocks": [{ "type": "text", "value": "..." }] }` | 有序追加内容片段。 |
| `output.final` | `{ "output": {...} }` | 最终结构化结果，例如已校验 citations。 |
| `run.error` | `{ "code": "...", "message": "...", "retryable": true }` | 失败。 |
| `run.cancelled` | `{ "reason": "user_requested" }` | 已取消。 |
| `heartbeat` | `{}` | SSE 保活；通常不落库。 |

Markdown 的流式增量使用内容 block，而非顶层 text：

```json
{
  "type": "content.delta",
  "data": {
    "blocks": [
      {
        "type": "text",
        "value": "会议认为良率问题主要来自……"
      }
    ]
  }
}
```

`StreamEmitter` 不按 token 逐条写数据库。它在内存中聚合 token，达到 **500ms** 或 **约 500 字符** 时，写一个 `content.delta`：将 blocks 追加到 `output_blocks`，递增 run 的 `last_sequence`，插入对应 event，并向实时订阅者发布。

### 5.3 问答与总结的内容组合

问答通常依次发出：`phase(retrieving)` → `evidence` → `phase(generating)` → 多个 `content.delta` → `output.final`。引用只在最终校验完成后进入 `output.final`，不在正文生成期间假定其正确。

总结通常发出：`phase` → 多个 `content.delta` → `output.final`。滚动总结内部的 JSON 片段总结和滚动记忆不产生内容 block，只更新 phase；最终综合总结才产生 `TextBlock`。

## 6. 两种断线恢复

### 6.1 同一页面的网络断开自动续传

同一个 HTTP SSE 连接因网络问题断开时，前端 SDK 会在重连请求中带上最近收到的 SSE `id`，即 `Last-Event-ID: N`。

服务端读取 `N` 后：

1. 查询 `generation_events` 中 `sequence > N` 的事件；
2. 按序回放；
3. 继续订阅新的实时事件。

客户端忽略 `seq <= last_sequence` 的重复事件。这是“至少一次投递 + 客户端按序去重”，而不是不必要的“恰好一次投递”。

### 6.2 页面刷新后的恢复

页面刷新会创建新的 HTTP SSE 请求，没有此前连接的 `Last-Event-ID`，前端内存文本也已丢失。服务端不能只发送后续 delta，必须先发送完整快照：

```json
{
  "v": 1,
  "run_id": "generation-run-id",
  "seq": 18,
  "type": "snapshot",
  "at": "2026-07-18T10:20:31.000Z",
  "data": {
    "status": "running",
    "phase": { "name": "generating", "label": "正在生成回答" },
    "blocks": [
      {
        "type": "text",
        "value": "截至 sequence 18 的完整已生成内容"
      }
    ]
  }
}
```

随后服务端从 `sequence > 18` 继续回放和订阅。前端收到 `snapshot` 时直接替换当前 blocks，随后再顺序追加新的 `content.delta` blocks。这样页面刷新后可以先“一次性展示已生成的全部内容”，再继续流式显示新增内容。

因此答案是：**数据库必须保存 sequence**，且必须同时保存 `output_blocks` 与 run 的 `last_sequence`。只保存临时内存或只保存内容而不保存其对应 seq，都无法无遗漏地处理刷新与并发写入之间的边界。

`GET /api/generations/{run_id}/events` 的约定：

- 请求含 `Last-Event-ID` 或 `after`：只做事件续传；
- 请求没有 cursor：先发 `snapshot`，再从 snapshot 的 `seq + 1` 续传。

## 7. 前端消息 SDK

SDK 放在 `app/sdk/generation/`，作为前端应用内部的稳定边界，而不是页面组件或 API route：

```text
app/sdk/generation/
├── types.ts       # 命令、快照、事件 discriminated union、视图状态
├── store.ts        # Zustand store：消费 snapshot/content delta/run status
├── transport.ts   # fetch HTTP SSE、帧解析、Last-Event-ID 和重连
├── client.ts      # GenerationStreamClient：连接、重连、订阅、取消
└── selectors.ts    # 从 blocks 派生纯文本、加载状态和最终结果
```

### 7.1 SDK 边界

- `types.ts` 与后端事件 envelope 一一对应，使用 discriminated union。
- `store.ts` 使用 Zustand 管理 `GenerationViewState`；事件归约逻辑仍实现为独立纯函数并由 store action 调用，便于单元测试与断线恢复测试。
- `HttpSseTransport` 使用 `fetch` 读取 GET SSE，负责帧解析、重连和 `Last-Event-ID`；其 `getHeaders` 可在账号系统接入后提供 Bearer Token。
- `GenerationStreamClient` 提供 `connect(runId)`、`close()`、`cancel()`；它不负责构造问答 query，也不负责渲染 Markdown 或引用卡片。
- 录音总结页与聊天页都订阅同一个 Zustand store，但各自负责展示 UI。

视图状态最小结构：

```ts
type GenerationViewState = {
  runId: string;
  kind: "recording_summary" | "rag_answer";
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  phase: { name: string; label: string } | null;
  blocks: ContentBlock[];
  error: GenerationError | null;
  lastSequence: number;
  connection: "connecting" | "connected" | "reconnecting" | "closed";
};
```

SDK 收到 `snapshot` 时替换 `blocks`、`status` 和 `lastSequence`；收到 `content.delta` 时仅接受 `seq > lastSequence` 的 blocks 并按顺序追加。`selectors.ts` 可将连续的 `TextBlock.value` 合并为页面所需的正文字符串。这样网络续传与刷新恢复对业务页面透明。

当前使用 `fetch` 实现 HTTP SSE。它自行解析 SSE 帧、维护 `Last-Event-ID` 并重连，因此未来增加 Bearer Token 时只需由 `getHeaders` 注入 Authorization header，不改变 reducer、client 或消息协议。

## 8. API 与执行接入

建议 API：

```text
POST   /api/generations/rag
GET    /api/generations/{run_id}
GET    /api/generations/{run_id}/events
DELETE /api/generations/{run_id}
```

- `POST` 创建后立即返回 `run_id`，不把长推理绑定在原始 HTTP 请求上。
- `GET /events` 由 SDK 订阅。
- `DELETE` 设置 `cancel_requested`；执行器在 token/chunk 边界结束生成并写入 `run.cancelled`。
- 总结不通过公开 `POST` 创建，而由 `generate_summary` pipeline stage 创建关联的 background Generation Run，并等待最终结果后投影到 `recording_summaries`。

## 9. 调度与模型运行时

Generation 资源优先级：

```text
gpu_interactive（问答）
    > gpu_high（ASR / diarization）
    > gpu_background（summary / embedding / correction）
```

优先级只能影响尚未开始的任务。单 GPU 上正在执行的 llama.cpp 推理不能安全抢占；一个已运行的 262K 上下文总结仍可能让问答等待。UI 应显示“等待 GPU”阶段，而不是伪造正在生成。

`LocalLlmRuntime` 应成为共享模型运行时边界，负责模型 profile、单模型并发锁、显存生命周期和首 token 指标。若产品要求问答在后台总结运行期间仍稳定秒级响应，需要独立 GPU 或独立模型进程；仅靠队列优先级无法做到真正抢占。

## 10. Redis 与保留策略

第一版不引入 Redis：

- PostgreSQL 是 `generation_runs` 和 `generation_events` 的可靠来源；
- 单进程 `GenerationStreamHub` 提供低延迟即时发布；
- SSE 断线时从 PostgreSQL 事件表恢复。

多 Web 实例或独立 worker 后，可在相同 `StreamHub` 协议下加入 Redis Pub/Sub。Redis 仅负责跨进程唤醒和实时广播，不能替代数据库快照。

完成后的最终业务结果保留在 `recording_summaries` 或未来的 `messages` 中；`generation_events` 的 delta 可按保留期清理，例如完成后 24 小时。即使历史 delta 被清理，`generation_runs.output_blocks` 与最终业务结果仍允许页面展示已完成内容。

## 11. 实施顺序

1. 新增 Generation 数据表、类型模型、事件存储和 `StreamEmitter`，并为 sequence/快照原子性编写测试。
2. 实现 Python SSE API 和 `app/sdk/generation/`；覆盖网络续传、页面刷新快照恢复、重复 event 去重。
3. 将 `generate_summary` 接入 GenerationService，先验证长任务、重试、取消和详情页展示。
4. 迁移旧 Node RAG stream 到 Python `rag_answer`，保留证据、文本增量、最终引用校验。
5. 增加 `gpu_interactive` 队列与 `LocalLlmRuntime`，最后按多实例需求决定是否引入 Redis。
