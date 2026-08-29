# Redis + Kafka 的 Kafka-first 架构改造设计

## 1. 文档状态

- 状态：主体改造完成（真实 Kafka/Redis 故障注入与音频端到端验证待运行）
- 日期：2026-08-06
- 适用范围：Production API、Processing、Compute Worker、Generation、RAG、Observability
- 核心决策：Kafka 是任务与过程事件的持久化事实源，Redis 是活跃状态与实时流投影；涉及 PostgreSQL 业务聚合写入的命令先与共享 Transactional Outbox 原子提交，再由 relay 投递 Kafka

本文档替代以下文档中与任务持久化、流式事件存储和多实例调度相关的旧结论：

- `generation-streaming-design.md` 中基于 PostgreSQL `generation_events` 和进程内 `GenerationStreamHub` 的方案；
- `http-compute-worker-architecture-upgrade.md` 中单实例、内存任务状态和内部 HTTP/SSE 作为主要任务通道的限制；
- `python-backend-rag-token-observability-design.md` 中通过进程内 bounded queue 和 HTTP 投递统计记录的方案。

本次改造不考虑历史数据兼容。数据库会从最终 Schema 重建，不实现迁移、双写或过渡兼容层。

## 2. 背景与目标

当前系统存在以下运行时限制：

- Compute Worker 的任务、状态和订阅者保存在单进程内存中，Worker 重启后任务丢失；
- RAG Workflow 通过 Production API 进程内的 `asyncio.create_task` 执行；
- Generation 实时广播依赖进程内 `GenerationStreamHub`，不能跨实例；
- Generation delta、phase 和 progress 高频写入 PostgreSQL；
- Pipeline 的运行时状态和历史事件由多张 PostgreSQL 表承担；
- Observability 使用进程内队列和 HTTP best-effort 投递，异常时可能丢事件。

目标如下：

1. Kafka 承载 Worker 任务、Processing 命令、Generation/RAG 命令、可靠取消、生命周期结果和 RAG 统计事件；
2. Redis Streams 承载 LLM/ASR delta、progress、phase 和面向 SSE 的短期续传；
3. Redis Hash/Set 承载活跃任务的当前状态、依赖完成集合、心跳、取消快速检查和短期幂等；
4. PostgreSQL 退出任务队列、高频进度和 Pipeline 运行时状态管理，只保存最终业务结果与统计查询投影；
5. 支持多 API、多 Consumer、多 CPU/GPU Worker、至少一次投递、幂等消费、重试和 DLQ；
6. 删除已经被 Kafka 或 Redis 完整接管的表、字段、Repository 和内存 Runner；
7. 保留现有类型化 Operation、Artifact 和面向前端的 SSE 协议中仍有价值的契约。

## 3. 存储职责边界

一句话原则：

```text
Kafka      保存“发生了什么、接下来做什么”
Redis      保存“现在进行到哪里、现在要展示什么”
PostgreSQL 保存“最后得到了什么、页面长期要查询什么”
```

| 组件 | 负责内容 | 不负责内容 |
| --- | --- | --- |
| Kafka | 命令、任务、可靠取消、生命周期事件、结果事件、统计原始事件、重试、DLQ | 逐 token SSE、任务状态快照、复杂业务查询、大型二进制 |
| Redis Streams | LLM/ASR delta、phase、progress、heartbeat、SSE 短期断线续传 | 唯一任务事实源、永久业务结果、无限期审计 |
| Redis Hash/Set/String | 活跃任务快照、Stage 完成集合、取消快速标记、Worker 心跳、锁、短期幂等和缓存 | 唯一一份终态、不可重建的业务数据 |
| PostgreSQL | 用户、录音、转写、说话人、Summary、Conversation、最终 Generation 结果、Observability 查询投影 | Worker/Processing 队列、逐 delta、progress、原始事件日志和中间 Artifact 元数据 |
| 文件/对象存储 | 音频、模型文件、大型 ASR/Embedding/中间 Artifact 和大型结果 | 消息调度和状态查询 |

### 3.1 判断数据应该放在哪里

1. 丢失后任务可能永远不执行：进入 Kafka；
2. 需要另一个服务收到后开始工作：进入 Kafka；
3. 需要重试、回放或审计：进入 Kafka；
4. 每秒高频更新、只用于当前进度或实时展示：进入 Redis；
5. 需要 TTL、锁、心跳或毫秒级读取：进入 Redis；
6. 是用户最终要长期查询的业务结果：进入 PostgreSQL；
7. 是大型不可内联数据：进入文件/对象存储，Kafka 只传 URI 和 checksum。

同一信息可以同时出现在 Kafka 和 Redis，但职责不同：Kafka 是可回放事实，Redis 是由事实构建的当前状态投影，不是两套互相竞争的事实源。

## 4. 总体架构

```text
Client
  │ HTTP command
  ▼
Production API
  ├── write file/object storage when needed
  ├── publish Kafka command and wait for broker ACK
  └── return 202 + generated run/task ID
                    │
                    ▼
                  Kafka
       ┌────────────┼──────────────────┐
       ▼            ▼                  ▼
Processing Worker  Compute Worker     Observability Worker
       │            │                  │
       ├── Kafka    ├── Kafka result   └── PostgreSQL projections
       ├── Redis    ├── Redis Streams
       └── final DB └── file/object storage

Client ◀── SSE / state query ◀── Production API ◀── Redis
Client ◀── final business query ◀── Production API ◀── PostgreSQL
```

### 4.1 业务事务使用 Outbox，内部工作流保持 Kafka-first

录音上传、重试、删除以及 Conversation RAG 创建同时修改 PostgreSQL 业务聚合并产生 Kafka 命令。它们使用共享 `integration_outbox`，由独立 `outbox-relay` 以 at-least-once 语义发送。业务行与消息意图在同一个 PostgreSQL 事务提交，API 成功表示命令已被 PostgreSQL 可靠接受，不再表示 Broker 已 ACK。

不伴随 PostgreSQL 业务事务的 standalone RAG command、Compute、Processing/RAG Worker 内部过程事件和 Observability 事件继续直接发布 Kafka，避免把数据库退化为所有运行时事件的任务队列。Generation terminal 是最终业务事实，因此 `generation_runs`、Conversation terminal projection 和 Redis/SSE terminal projection outbox 必须在同一个 PostgreSQL 事务提交。

命令入口必须遵循：

1. API 生成 `run_id`、`task_id` 和 `event_id`；
2. 如有文件，先原子写入文件或对象存储；
3. 在同一事务写业务聚合和完整 Kafka envelope 到 `integration_outbox`；
4. 提交 PostgreSQL 后返回 `202 Accepted`；
5. relay 使用租约、退避和耗尽告警投递 Kafka；
6. Kafka 不可用时命令留在 outbox 等待恢复，不回滚已提交业务事实。

不得在同一路径同时保留 outbox 与提交后的尽力直发；否则会制造无意义的重复投递和两套成功语义。

Relay 在 Kafka ACK 后、标记 `published_at` 前崩溃时会重复发送，因此消费者继续依赖稳定 `processing_id`/`generation_id`、终态检查、Stage 缓存和幂等投影。已发送行保留一段时间供审计后清理；当前不引入通用 inbox。

### 4.2 最终结果链路

- 小型结果，如 RAG 回答和 Summary，由 Generation Worker 在同一个事务内幂等 UPSERT `generation_runs`、更新 Conversation terminal projection 并写入 Redis/SSE terminal projection outbox；
- 大型结果先原子写对象存储，PostgreSQL 保存 URI、checksum 和 schema version；
- 数据库事务提交后即可提交原 Generation command Offset；即使 Worker 随后退出，outbox relay 仍会执行 Redis/SSE terminal projection；
- `outbox-relay` 使用 Redis 原子幂等投影写 terminal snapshot、`output.final`/`run.error`/`run.cancelled` 和 TTL；成功后才标记 Outbox 行已发布；
- 查询时 PostgreSQL terminal snapshot 优先于 Redis active snapshot，避免陈旧 `RUNNING` 遮住已提交终态。

不再通过 Kafka state topic 绕回同进程 Result Consumer。PostgreSQL 是 Generation 最终查询源；Redis terminal/SSE 只是由 Outbox Relay 构建的短期交付投影。

## 5. 基础设施由谁启动

Redis 和 Kafka 是独立基础设施进程，不由任何 L3 Python 包启动。

| 环境 | 启动方式 |
| --- | --- |
| 本地开发 | Docker Compose，Kafka 使用 KRaft 模式 |
| CI | Docker Compose 或 Testcontainers |
| 单机部署 | Docker Compose 或 systemd |
| Kubernetes | Helm Chart 或 Operator |
| 云环境 | 托管 Kafka 与托管 Redis |

L3 应用只初始化客户端连接。Kafka Client 的 `start()` 表示建立连接，不表示启动 Kafka Broker。

推荐根目录命令：

```text
npm run infra:up
npm run infra:down
npm run infra:status
npm run infra:logs
```

启动顺序：Docker Compose 启动 PostgreSQL、Redis、Kafka并等待 healthcheck，通过后再启动各 L3 应用。应用 readiness 在必要依赖不可用时返回失败，并使用有界退避重连。

## 6. 分层与 L3 应用边界

Kafka/Redis 通用适配器属于 L1，业务消息与状态转换属于 L2，可独立运行的进程装配属于 L3。

```text
backend/packages/
├── l1_foundation/
│   ├── messaging/
│   │   ├── contracts.py
│   │   └── kafka/
│   │       ├── producer.py
│   │       ├── consumer.py
│   │       ├── serialization.py
│   │       └── admin.py
│   └── streaming/
│       ├── contracts.py
│       └── redis/
│           ├── client.py
│           ├── stream_store.py
│           └── state_store.py
├── l2_core/
│   ├── processing/
│   │   ├── commands.py
│   │   ├── events.py
│   │   └── orchestrator.py
│   ├── generation/
│   │   ├── commands.py
│   │   ├── events.py
│   │   └── stream_service.py
│   └── observability/
│       ├── events.py
│       └── projector.py
└── l3_app/
    ├── production-api/
    ├── compute_worker/
    ├── processing-worker/
    ├── observability-api/
    └── observability-worker/
```

- `production-api`：发布 Kafka 命令、查询 Redis、输出 SSE、查询 PostgreSQL 最终结果；
- `processing-worker`：消费 Processing 命令和结果、推进版本化 DAG、发布 Compute Task并维护状态；
- `compute_worker`：按资源 lane 消费任务、执行 Operation、写 Redis delta/progress、发布 Kafka 终态；
- `observability-worker`：消费统计事件，批量 UPSERT PostgreSQL 查询投影；
- `observability-api`：只查询统计投影。

不创建通用 `redis-api` 或 `redis-worker`。`outbox-relay` 是独立进程，并为 `generation-state` channel 执行 durable terminal 的 Redis/SSE 投影；其他 channel 仍只负责 Kafka 投递。

## 7. Kafka 设计

### 7.1 Topic 规划

```text
processing.commands
processing.events
processing.cancel
processing.cancel.retry
processing.cancel.dlq
processing.retry
processing.dlq

compute.tasks.io
compute.tasks.cpu
compute.tasks.gpu-high
compute.tasks.gpu-normal
compute.results
compute.cancel
compute.retry
compute.dlq

generation.commands
generation.retry
generation.dlq

rag.execution-events
model.invocation-events
observability.dlq
```

所有 commands/events Topic 使用时间或容量 retention；Processing、Compute 和 Generation 状态只保存在 Redis，不再创建 compacted state topic。

Compute Topic 按资源 lane 拆分，避免耗时 GPU 任务阻塞短 I/O 任务，并复用现有 `ResourceQueue`。

### 7.2 Message Key 与信封

- Compute Task：`task_id`；
- Processing：`processing_id`；
- Stage：`processing_id:stage_name`；
- Generation/RAG：`generation_id`；
- Observability：`generation_id` 或 `trace_id`。

Kafka 只保证同一 Topic、同一 Partition 内顺序，不依赖跨 Topic 全局顺序。

```json
{
  "event_id": "019...",
  "event_type": "compute.task.requested",
  "schema_version": 1,
  "occurred_at": "2026-08-05T10:30:00Z",
  "producer": "processing-worker",
  "correlation_id": "...",
  "causation_id": "...",
  "workspace_id": "...",
  "processing_id": "...",
  "generation_id": "...",
  "task_id": "...",
  "trace_id": "...",
  "payload": {}
}
```

`event_id` 全局唯一；`schema_version` 支持契约演进；`correlation_id` 串联请求；`causation_id` 指向上游消息；Payload 必须通过 Pydantic Contract 校验；大型数据只传 URI 和 checksum。

### 7.3 Producer 与 Consumer 可靠性

Producer 至少要求：

```text
acks=all
enable.idempotence=true
retries>0
```

直接 Kafka 路径等待 Broker ACK 后才返回成功；outbox-backed API 在 PostgreSQL 提交后返回，由 relay 等待 Broker ACK。生产环境结合副本数配置 `min.insync.replicas`。

Consumer 采用“至少一次投递 + 幂等处理”：

1. 拉取并校验消息；
2. 执行业务或更新投影；
3. 副作用成功后提交 Offset；
4. 可重试错误进入 retry；
5. 超过次数进入 DLQ；
6. 重复消息通过 `event_id`、`task_id` 或业务唯一键消除副作用。

数据库事务完成前不得提交 Offset。数据库成功但 Offset 未提交时会重复消费，所以最终写入使用 UPSERT 或唯一约束。

`processing-worker` 的一次消息处理覆盖完整录音 DAG，默认使用独立的 `PROCESSING_CONSUMER_MAX_POLL_INTERVAL_MS=7200000`（2 小时），不与短任务 Consumer 共用 15 分钟配置。即使发生超时、重平衡或提交 Offset 前崩溃，Handler 也会先读取 Redis；相同 `processing_id` 已处于 `succeeded`、`partial_failed`、`failed` 或 `cancelled` 时直接跳过业务执行，让 Consumer 提交该重复消息。

## 8. Redis 设计

### 8.1 当前状态 Key

```text
processing:{processing_id}                 Hash
processing:{processing_id}:stages          Hash
processing:{processing_id}:completed       Set
compute:{task_id}                           Hash
generation:{generation_id}                 Hash
task:{task_id}:cancel                       String
worker:{worker_id}:heartbeat                String
idempotency:{scope}:{key}                   String
```

```text
HSET processing:{processing_id}
  status running
  pipeline recording_processing
  version 2
  current_stage transcribe
  progress 63
  updated_at 2026-08-05T10:30:00Z

HSET processing:{processing_id}:stages transcribe
  {"status":"running","attempt":2,"progress":63}

SADD processing:{processing_id}:completed normalize
```

活跃状态由 Consumer 根据 Kafka 事件写入。状态值携带事件版本或 `event_id`，避免重复或乱序消息覆盖更新状态。

### 8.2 Redis Streams

```text
generation:{generation_id}:events
compute:{task_id}:events
processing:{processing_id}:events
```

适合写入：`phase`、`progress`、`content.delta`、`asr.delta`、`snapshot`、`heartbeat` 和实时终态通知。

不能只写 Redis 的内容包括：任务命令、可靠取消、唯一终态、RAG 统计原始事件和永久业务结果。

### 8.3 SSE、TTL 与持久化

- SSE 用 `XREAD BLOCK` 读取，Stream ID直接作为 SSE `id`；
- 浏览器使用 `Last-Event-ID` 续传；
- 无 Cursor 时先读取累计 snapshot，再消费后续 Stream；
- Stream 过期且任务终结时读取 PostgreSQL 最终结果；
- 活跃 Key 不设 TTL，终态 Hash 保留约 24 小时；
- 完成后的 Stream 保留 1 至 24 小时并使用近似 `MAXLEN`；
- 建议 `appendonly yes`、`appendfsync everysec`、`maxmemory-policy noeviction`；
- 生产环境使用副本与故障转移，但 Redis 仍不是唯一事实源。

### 8.4 取消

可靠取消进入 Kafka：`task.cancel.requested`、`generation.cancel.requested`、`processing.cancel.requested`。Consumer 收到后设置 Redis Cancel Key，Worker 在 token/chunk/operation 边界快速检查。Kafka 保存不可丢的取消事实，Redis 负责低延迟检查。

Generation、RAG Checkpoint 与通用 Compute execution scope 的详细目标设计参见 [`generation-cancellation-and-rag-checkpoint-design.md`](./generation-cancellation-and-rag-checkpoint-design.md)。

录音删除路径会在删除数据库记录和文件前发布 `processing.cancel.requested`，等待 Kafka ACK 后才继续。独立的 Processing Cancel Consumer 将该可靠事实投影为按 `recording_id` 保存的 Redis Cancel Key，写入成功后提交 Offset；这使取消不依赖 API 先从 Redis 找到活跃 `processing_id`，也能覆盖“取消先到、任务后到”的竞态。Processing Handler 在启动前、每个 Stage 边界以及异步 Stage 执行期间同时检查 `processing_id` 和 `recording_id` 取消标记；观察到取消后停止剩余 DAG、将活跃 Stage 和 Processing 写为 `cancelled`，发布终态并结束本次 Kafka 消费。异步 Stage 正在等待 Compute task 时，Compute Worker 的取消暂时仍使用现有 Redis 快速标记。

## 9. Processing 状态与 Pipeline DAG

以下 PostgreSQL 运行时表删除：

```text
pipeline_runs
stage_runs
stage_run_dependencies
pipeline_events
```

| 旧职责 | 新位置 |
| --- | --- |
| Pipeline 当前状态 | Redis `processing:{id}` |
| Stage 当前状态 | Redis `processing:{id}:stages` |
| 已完成 Stage 集合 | Redis `processing:{id}:completed` |
| Redis 丢失后的粗状态兜底 | PostgreSQL `recordings` 业务投影 |
| 状态变化历史 | Kafka `processing.events` |
| Stage 依赖 | 代码中的版本化 Pipeline Definition |
| progress | Redis Stream/Hash |
| retry、DLQ | Kafka retry/DLQ Topic |

Processing Worker 收到 `stage.succeeded` 或 `compute.completed` 后：

1. 幂等更新 Redis 运行态；
2. 将 Stage 加入 Redis completed Set；
3. 从版本化 Pipeline Definition 获取下游依赖；
4. 对 fan-in 节点检查全部依赖；
5. 为 ready 节点生成确定性 `task_id` 或幂等键；
6. 发布到对应 `compute.tasks.*`；
7. 重复结果不会产生重复调度副作用。

Redis 丢失时不重建逐 Stage运行态；录音详情从 PostgreSQL `recordings` 合成粗粒度 Pipeline状态，正在执行的任务按失败或重新提交策略收敛。

## 10. Compute Worker

- Consumer Group 按 `io`、`cpu`、`gpu-high`、`gpu-normal` lane 消费；
- Handler 继续使用类型化 Operation Contract；
- 流式任务的 progress、delta 和终态进入 Redis Streams；
- 每个请求携带 `reply_to` 和 `requester_id`，发起方实例使用独立 Consumer Group 接收 `compute.results` reply；
- Kafka reply 只保存 locator 或终态错误，不承载 progress 和业务结果；序列化结果不超过 256 KiB 时写 Redis，超过阈值时写 FileStore；
- 非流式 client 从 Redis task state 读取 progress，完成后按唯一一次 terminal reply 的 locator 读取并清理结果；流式 client 从唯一一次初始 reply 指定的 Redis Stream 消费事件；
- `task_id` 是幂等边界，成功处理后提交 Offset；
- HTTP 只保留 health、readiness、metrics、Operation 列表和管理接口。

## 11. Generation 与模型流式输出

### 11.1 delta 与最终结果

模型输出重要，但要区分：

- `delta` 是 Provider/SDK/网络决定的传输切片，没有独立业务语义；
- 拼接后的完整回答、Summary 或结构化输出才是最终业务结果。

```text
逐 delta / progress       → Redis Streams
累计 snapshot             → Redis Hash/String
Generation terminal       → PostgreSQL generation_runs
终态 Redis/SSE 短期投影     → Transactional Outbox Relay
最终大型结果               → Object Storage URI → PostgreSQL
```

不得把 Redis 作为完整模型结果的唯一存储，也不把每个 token 发入 Kafka，以避免无业务价值的消息爆炸和磁盘写放大。

### 11.2 流式执行

1. Worker 将每个 delta 写入 Redis Stream；
2. Worker 内存同时累积完整输出；
3. 每 1 秒或约 1,000 字符更新 Redis 累计 snapshot；
4. 完成后幂等写入 `generation_runs`，Conversation/Recording 业务投影按各自边界更新；
5. 在同一事务写入 Redis/SSE terminal projection outbox，由 relay 可靠投影；
6. 设置 Redis Stream 和状态 TTL，随后提交原 command Offset。

### 11.3 故障语义

- 浏览器断开：用 Stream ID 和 `Last-Event-ID` 续传；
- Redis 重启：可能损失一小段观看过程，但 Kafka 任务和最终结果不丢；
- Redis 丢失且 Worker 仍运行：Worker 重写累计 snapshot；
- Worker 中途崩溃：不提交任务 Offset，重新消费并完整重做，不把半截文本作为最终结果；
- command 重投：Generation Worker 按终态检查跳过计算，并按 `generation_id` 幂等 UPSERT；
- Stream 过期：API 返回 PostgreSQL 最终结果，而不是历史 token 切片。

未来只有出现“必须审计每个原始 token 切片”的合规需求时，才单独设计批量 chunk 或对象存储日志。

## 12. RAG 与 Observability

```text
Production API
  → Kafka rag.answer.requested
  → RAG Worker
  → Redis answer delta + phase/progress
  → PostgreSQL generation_runs / conversation_messages
  → Outbox Relay → Redis terminal/SSE projection
```

RAG 原始统计事件进入 `rag.execution-events` 和 `model.invocation-events`。

Observability Worker 持续消费并幂等写入查询投影；数据库写入成功后提交 Offset，失败事件转入 retry topic，耗尽后进入 `observability.dlq`。当前实现逐事件投影，后续吞吐量需要时可在不改变消息契约的前提下增加批量写入。

Kafka 保存原始统计事件；PostgreSQL `rag_execution_spans` 和 `model_invocations` 保存 Dashboard 查询投影。投影使用事件 ID 或调用 ID 幂等 UPSERT。Observability 的 `generation_id` 是关联 ID，不依赖活跃任务数据库外键存在。

## 13. PostgreSQL Schema 清理

### 13.1 删除的表

```text
generation_events
pipeline_runs
stage_runs
stage_run_dependencies
pipeline_events
outbox_events
artifacts
```

同时删除相关索引、外键、注释、Repository、Cleanup 和测试代码。

### 13.2 Generation 最终结果投影

独立 RAG Query、Summary regeneration 和通用 Generation 查询仍需要在 Redis 过期后读取最终结果。因此保留精简后的 `generation_runs`，但它不再承担 queued/running 调度、delta、progress、sequence 或取消状态，也不是任务事实源。

它只作为终态查询投影，由 Generation Worker 在终态事务中写入：

```text
id
kind
idempotency_key
parent_type
parent_id
owner_user_id
subject_type
subject_id
terminal_status
input_metadata
output_payload
error_code
error_message
created_at
finished_at
```

删除旧字段：

```text
priority
phase
progress_percent
output_blocks
last_sequence
first_token_at
cancel_requested
started_at
updated_at
```

如果后续移除独立 Generation 查询，并保证所有最终结果都有明确业务归属，可进一步删除 `generation_runs`，由 `conversation_messages`、`recording_summaries` 等业务表承担终态结果。

Kafka-first 下，Generation 关联 ID 可能先于最终投影行出现，因此同步调整外键：

- `conversation_messages.generation_run_id` 改为无外键的 `generation_id` 关联字段；
- `rag_execution_spans.generation_run_id` 改为无外键的 `generation_id`；
- `model_invocations.generation_run_id` 改为无外键的 `generation_id`；
- 这些字段保留普通索引，用于跨表查询和 Trace 关联，不再要求活跃任务先写 PostgreSQL。

### 13.3 Recording 终态与 Artifact

Processing 活跃状态只在 Redis，但录音长期页面仍需要知道最终处理结果。`recordings` 只保存终态投影，不保存 Stage 过程：

```text
processing_status    succeeded / failed / cancelled，活跃时以 Redis 为准
processing_error     可选的最终错误摘要
processed_at         最终结束时间
```

数据库 `artifacts` 表一并删除。中间产物由文件/对象存储保存，URI、checksum 与类型化引用随 Processing/Compute 事件和 Redis 活跃状态传递；最终需要长期查询的内容由对应业务投影表保存。Artifact key 由 `processing_id`、确定性 `stage_run_id`、Stage 名称/版本及 Artifact 类型/版本生成 SHA-256；当前本地适配器将它保存为 `uploads/artifacts/{hash}.json`，生产环境可直接使用相同 hash 作为对象存储 key。每个 Audio Processing Stage 通过 `try_restore()` 自行校验并恢复完整 `StageResult`；命中后跳过计算、恢复下游 ArtifactRef，并重新执行幂等数据库投影。

### 13.4 保留的表类别

| 类别 | 表 |
| --- | --- |
| 身份与权限 | `users`、`workspaces`、`workspace_memberships`、`user_sessions` |
| 录音业务 | `recordings`、转写、说话人、Utterance、Summary、Search Chunk 等表 |
| 对话 | `conversations`、`conversation_messages` |
| 最终 Generation 查询投影 | 精简后的 `generation_runs` |
| Observability | `rag_execution_spans`、`model_invocations` |
| Evaluation/Training | `sql/evaluation.sql` 中的离线评测和训练表，不受本次改造影响 |

不新增：

```text
compute_tasks
worker_tasks
processing_tasks
rag_events
stream_events
inbox_events
```

### 13.5 Schema 原则

- 直接修改 `sql/base.sql` 为最终结构；
- 不加入历史迁移用 `ALTER TABLE ... DROP COLUMN` 或 `DROP TABLE`；
- 不实现新旧双写；
- 不保留旧 Repository、内存 Hub/Runner 或 HTTP ingestion 兼容代码；
- 测试数据库全部从新 Schema 初始化。

## 14. 故障处理

| 故障 | 处理 |
| --- | --- |
| Kafka 在 outbox-backed 命令创建时不可用 | API 正常接受，relay backlog 增长并告警；恢复后自动投递 |
| Producer 超时但结果未知 | 使用相同 `event_id/task_id` 重试 |
| Consumer 处理前崩溃 | Offset 未提交，消息重新消费 |
| 数据库写成功、Offset 未提交 | 重复消费，UPSERT/唯一约束消除副作用 |
| Redis 不可用 | 实时体验和运行态不可用；命令与最终业务结果不丢，录音详情回退 PostgreSQL粗状态 |
| Worker 生成中崩溃 | 不提交 Offset，完整重做 |
| Artifact 写完但完成事件未发 | 重试发布；孤立 Artifact 由 Cleanup 清理 |
| retry 耗尽 | 原始 envelope、错误和尝试信息进入 DLQ |

## 15. 可观测性

至少监控 Kafka Producer 成功率/超时/重试、Consumer lag/rebalance/耗时、Topic 速率、retry/DLQ、各资源 lane 排队与执行时间、Redis 内存/Stream 长度/eviction、SSE 连接与重连、Observability 批量落库。

日志统一携带：

```text
event_id
task_id
processing_id
generation_id
correlation_id
causation_id
trace_id
topic
partition
offset
attempt
```

## 16. 实施顺序

1. 增加 Docker Compose 中的 Kafka KRaft、Redis 和 healthcheck；
2. 在 L1 实现 Kafka/Redis Client、envelope、序列化和生命周期；
3. 定义 Processing、Compute、Generation、Observability Topic Contract；
4. 实现 Consumer 幂等、retry 和 DLQ；
5. 实现 Redis State/Stream Store 和基于 Stream ID 的 SSE；
6. 将 Generation/RAG 提交改成 Kafka command，移除进程内 Runner；
7. 将模型 delta/progress 改写 Redis，完整终态通过业务事务写入数据库；
8. 将 Compute Worker 主入口改成 Kafka lane Consumer；
9. 将 Pipeline Coordinator 改成 Kafka 驱动的 Processing Worker；
10. 将 Observability HTTP ingestion 改成 Kafka Consumer；
11. 清理 `sql/base.sql` 中被替代的表、字段、索引和外键；
12. 删除 Outbox、内存 Hub/Runner、旧 HTTP ingestion 和相关测试；
13. 重建数据库并完成单元、集成、故障注入和端到端验证。

### 16.1 当前实施进度

- 已完成：Docker Compose、Kafka/Redis L1 适配器、统一 envelope/topic、Kafka topic 初始化；
- 已完成：Generation delta/progress/SSE 迁入 Redis，删除 `GenerationStreamHub` 和 `generation_events`；
- 已完成：RAG API 改发 `generation.commands`，新增独立 `generation_worker`，删除进程内 `RagWorkflowRunner`；
- 已完成：Compute Worker 消费资源 lane topic；Redis 保存 live state/stream 和不超过 256 KiB 的同步结果，大结果写 FileStore，Kafka Request/Reply 只传结果 locator；Production API 与各 Worker 使用独立 reply Consumer Group；
- 已完成：RAG/模型统计改发 Kafka，新增 `observability_worker` 投影 PostgreSQL；
- 已完成：新增共享 `integration_outbox`、独立 relay、失败退避/耗尽和保留清理；Worker 不接入通用 inbox，继续依赖稳定任务 ID、终态检查和业务幂等；
- 已完成：Generation 提交以 Redis 活跃快照 + Kafka command 为起点，终态由 Generation Worker 幂等写入精简后的 `generation_runs`，不再创建数据库 queued 行；
- 已完成：新增独立 `processing-worker` 推进版本化 DAG，删除数据库 Pipeline Coordinator、Repository、运行时表及旧 E2E；
- 已完成：Processing 使用独立 2 小时 `max.poll.interval`，终态 `processing_id` 重投直接跳过；删除录音会先发布 Kafka 可靠取消事件，由独立 Cancel Consumer 设置 Redis 快速标记并协作式停止 Processing；
- 已完成：Compute Worker 删除 task/status/cancel/SSE HTTP API，只保留 health/readiness/metrics，任务改由 Kafka lane 消费；
- 已完成：Generation 与 Compute 接入独立 Kafka Cancel Consumer；Compute task 使用通用 execution scope，Worker 同时覆盖取消先到与任务先到，并在 Handler 安全退出后释放模型；
- 已完成：命令 Consumer 与 Generation/Observability 投影具备 retry/DLQ；Generation终态通过 Outbox Relay直接投影 Redis/SSE；
- 已完成：`db:init` 改为重建 `public` schema，旧表与历史数据不会残留；
- 待验证：在可用 Docker 环境运行真实 Kafka/Redis/PostgreSQL 的重启、重复投递、DLQ、Redis 清空和音频端到端故障测试。

## 17. 验收标准

- Redis/Kafka 由基础设施启动，L3 只连接；
- Outbox-backed Production API 在 PostgreSQL 业务行与消息意图原子提交后返回成功；
- Kafka 不可用时不产生数据库 queued 孤儿；
- Worker 重启后未确认任务可重新消费；
- 重复任务和完成事件不产生重复副作用；
- `generation_events`、Pipeline 运行时表、`artifacts` 和 `outbox_events` 不再存在；
- delta/progress 不再写 PostgreSQL；
- SSE 支持 Redis Stream ID 续传与累计 snapshot；
- Redis 清空后不丢最终业务数据，录音详情可回退 PostgreSQL粗状态；
- Redis 丢失不会丢任务或最终模型输出；
- 完整模型输出通过 Kafka结果或 Artifact URI 进入最终存储；
- RAG 统计通过 Kafka 持续消费并幂等落库；
- Processing DAG 不依赖数据库依赖表；
- retry、DLQ、Consumer lag 和 Redis Stream 均有监控；
- `sql/base.sql` 只包含最终 Schema，不包含历史兼容迁移。
