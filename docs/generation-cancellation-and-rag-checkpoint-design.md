# Generation 取消传播与 RAG Checkpoint 设计

> 文档状态：Phase 1、Phase 2 与 Phase 3 已实现。
>
> 依赖架构：Generation、Compute 和 Processing 使用 Kafka 承载可靠命令与终态，Redis 承载活跃状态、取消快速标记和流式事件，PostgreSQL 保存最终业务投影。整体架构参见 [`redis-kafka-architecture-refactor.md`](./redis-kafka-architecture-refactor.md)。

## 1. 背景

对话当前支持流式生成，但停止能力只在 Generation 层写入 Redis 取消标记。RAG 只在 LangGraph 执行前后检查取消，因此用户在 route、retrieval、rerank 或 answer 中途停止时，当前节点可能继续运行，后续节点也可能继续调度。

本设计解决以下问题：

- 用户停止后，取消命令可靠送达，不因 API、Worker 或 Redis 短暂故障丢失；
- Generation Worker 停止 RAG 后续调度；
- Compute Worker 停止属于同一次上层执行的排队或运行任务；
- 保留已完成的 RAG 节点输出，后续可以从最近 checkpoint 继续；
- 覆盖“取消先于任务消费”和“任务先于取消消费”两种竞态；
- 保持当前线程执行模型，不引入每任务子进程；
- Compute 的取消归属不绑定 Generation，可复用于 Audio Processing 和 Evaluation。

## 2. 设计结论

1. 取消以 Kafka 事件作为可靠事实，Redis Cancel Key 作为低延迟投影。
2. Kafka 中已发布的任务不删除；Consumer 消费后根据取消状态跳过，并发布明确的 cancelled 终态。
3. Compute task 携带通用 `execution_scope`，而不是 Generation 专用 owner。
4. Compute Worker 保持线程执行，通过 token、item 或 micro-batch 边界协作取消，不强杀线程。
5. `model.release()` 只能在推理安全退出后执行，不能由取消线程并发调用。
6. RAG checkpoint 记录已完成节点的输出；运行中的节点取消后在恢复时重跑。
7. “继续生成”创建新的 Generation run，复用旧 run 的 checkpoint，不恢复模型内部推理状态。

## 3. 当前实现与缺口

当前已有能力：

- Generation API 通过 Kafka 发布可靠取消命令；
- Generation Worker 使用独立 Cancel Consumer 将取消投影到 Redis，并监控活跃 RAG/Summary task；
- Compute Client 通过 Kafka 发布 task 级可靠取消命令，并在 ACK 后写 Redis 快速标记；
- Compute task 携带通用 `execution_scope`，Generation、Processing 和 Evaluation 均会传播对应 scope；
- Compute Worker 使用独立 Cancel Consumer、Redis scope marker 和 Runtime scope 索引取消任务；
- Compute Worker 每 200ms 检查 task Cancel Key，并调用 `runtime.cancel(task_id)`；
- Compute Runtime 已有 `cancel_event` 和 `WorkerExecutionContext.raise_if_cancelled()`；
- 流式 LLM 在 token 事件边界检查取消；
- Compute Runtime 在 Handler 退出后的 `finally` 中调用 `operation.release()`；
- Processing 已实现 Kafka 可靠取消事件、独立 Cancel Consumer 和 Redis 投影，可作为实现范例。

当前缺口：

- 非流式 LLM 进入 `model.complete()` 后没有取消安全点；
- Rerank 进入整批 `model.predict()` 后只能等待该调用返回；
- Embedding/Rerank 尚未拆分为可快速取消的 micro-batch；
- SQL 召回尚未注册可取消 connection，第一版只能丢弃迟到结果；
- RAG 尚无生产 checkpoint 持久化与恢复入口。

## 4. 总体架构

```text
Browser
  │ POST /generations/{id}:cancel
  ▼
Production API
  │ generation.cancel.requested
  ▼
Kafka generation.cancel
  ▼
Generation Cancel Consumer
  ├─ 写 Redis generation cancel marker
  ├─ 停止本进程 active RAG execution
  └─ 发布 execution/compute cancel command
                  │
                  ▼
           Kafka compute.cancel
                  ▼
          Compute Cancel Consumer
           ├─ 写 execution scope cancel marker
           ├─ 取消 queued task
           └─ 对 running task 设置 cancel_event
                         │
                         ▼
                Handler 协作退出
                         │
                         ▼
              finally -> model.release()
                         │
                         ▼
              compute.task.cancelled
                         │
                         ▼
               generation.cancelled
```

Kafka 保证取消事实可重放；Redis 负责快速判断；Worker 的进程内索引负责立即唤醒活跃任务。

## 5. 通用执行作用域

### 5.1 数据契约

Compute 不应理解 conversation、recording 等具体业务对象，只需要知道任务属于哪一次上层执行。

```python
class ExecutionScope(BaseModel):
    kind: Literal["generation", "processing", "evaluation", "standalone"]
    id: UUID


class ComputeTaskRequest(BaseModel):
    task_id: UUID
    operation: str
    operation_version: str
    resource_queue: ResourceQueue
    execution_scope: ExecutionScope | None = None
    input: JsonObject
```

示例：

```json
{"kind":"generation","id":"generation-run-id"}
{"kind":"processing","id":"processing-id"}
{"kind":"evaluation","id":"evaluation-run-id"}
```

`standalone` 或空值只用于没有上层工作流的独立 Compute 调用，这类任务仍可按 `task_id` 取消。

### 5.2 Scope 传播

RAG、Processing 和 Evaluation 在执行入口创建 run-scoped `ContextVar`。构造 `ComputeCommand` 时读取该上下文，并将 scope 显式写入 Kafka payload。ContextVar 只用于减少应用层参数传递，不能替代消息协议中的显式字段。

Compute Runtime 维护：

```text
task_id -> TaskState
ExecutionScope -> set[task_id]
```

同一 scope 取消时，Worker 可以找到其所有排队和运行中的 Compute task。

第一版每个 task 只携带一个主执行作用域。Recording 删除等业务级取消先由 Processing 层映射到对应 `processing_id`，再向 Compute 发布 scope 级取消，避免 Compute 层理解业务对象层级。

## 6. Kafka Topic 与事件

新增或补全：

```text
generation.cancel
generation.cancel.retry
generation.cancel.dlq

compute.cancel
compute.cancel.retry
compute.cancel.dlq
```

Generation 取消命令：

```json
{
  "event_type": "generation.cancel.requested",
  "generation_id": "...",
  "reason": "user_requested",
  "requested_by": "...",
  "requested_at": "..."
}
```

Compute 取消命令同时支持 task 和 scope：

```json
{
  "event_type": "compute.cancel.requested",
  "target": {
    "type": "execution_scope",
    "scope": {"kind":"generation","id":"..."}
  },
  "reason": "upstream_cancelled"
}
```

```json
{
  "event_type": "compute.cancel.requested",
  "target": {"type":"task","task_id":"..."},
  "reason": "user_requested"
}
```

Kafka 事件不可修改。取消后应追加事件并更新 compacted state：

```text
compute.task.started
compute.task.cancel_requested   # 可选生命周期事件
compute.task.cancelled          # 唯一终态
```

## 7. 为什么不能从 Kafka 删除待执行任务

Kafka 是追加日志，已确认写入的单条 command 不能像传统内存队列一样按 task ID 删除。Log Compaction 也不是实时撤回机制：它按 key 在后台保留最新值，旧消息在清理前仍可能被 Consumer 读取。

因此取消使用追加事实，而不是删除 command：

```text
task.requested
task.cancel.requested
```

如果取消先被处理，之后才消费到 command：

```text
读取 command
→ 检查 task/scope cancel state
→ 不加载模型
→ 发布 cancelled_before_start
→ 提交 command offset
```

如果 command 先开始执行：

```text
注册 active task
→ 收到 cancel
→ 设置 cancel_event
→ Handler 协作退出
→ 发布 cancelled
```

不同 Topic 之间没有全局顺序保证，因此两条路径都必须正确。

## 8. Compacted State 与 Redis Cancel Key

项目已有 compacted state topic：

```text
processing.state
compute.state
generation.state
```

Compacted topic 是按消息 key 保存最新状态的 Kafka 日志。例如同一个 Generation 依次写入 `running`、`cancel_requested`、`cancelled`，后台压缩后长期保留的主要是最后状态。它不是即时删除，也不替代实时事件流。

Redis Cancel Key 提供低延迟检查：

```text
execution:generation:{id}:cancel
execution:processing:{id}:cancel
task:{task_id}:cancel
```

不能只依赖短 TTL Redis key，否则 Kafka command 长时间积压、Redis key 先过期后，旧任务可能被错误执行。取消状态至少应保留到：

- 对应上层执行进入终态；
- 相关 Kafka command 不再可能重投；
- Worker 重建状态时仍能识别该执行已经取消。

第一版优先将取消状态写入现有 `generation.state`、`processing.state` 和 `compute.state`。Compute Worker 若需要独立重建所有上游 scope 的取消屏障，可后续引入统一的 `execution.state` compacted topic；第一版不必立即增加该抽象。

## 9. Consumer 竞态处理

### 9.1 取消先到、任务后到

Cancel Consumer 先写可靠 state 和 Redis marker。Command Handler 在注册任务前检查：

```python
if cancellation_registry.is_task_cancelled(request.task_id):
    return cancelled_before_start(request)

if request.execution_scope is not None and cancellation_registry.is_scope_cancelled(request.execution_scope):
    return cancelled_before_start(request)
```

### 9.2 任务先到、取消后到

Command Handler 先注册 active task。Cancel Handler 根据 scope 索引找到任务并调用 `runtime.cancel(task_id)`。

### 9.3 检查与注册之间的竞态

以下操作必须位于 Runtime 的同一临界区：

```text
检查 scope 是否已取消
注册 active task
```

Cancel Handler 在同一临界区内：

```text
登记 scope 已取消
读取该 scope 的 active task 集合
```

随后在锁外逐个设置 `cancel_event`，避免清理或事件发布长时间占锁。

## 10. Generation Worker 取消

Generation Command Consumer 可能正在等待整个 RAG 执行，因此必须有独立 Generation Cancel Consumer。

每个活跃 RAG 执行安装一个基于 Generation cancel marker 的协作式取消上下文。取消检查从 Redis 读取共享标记，并在单次执行内做短周期节流和 sticky 缓存，避免每个 token 都访问 Redis：

```python
with rag_cancellation_scope(cancel_signal):
    await graph.run(...)
```

收到取消后：

1. 写 Generation cancel marker；
2. 发布对应 execution scope 的 Compute 取消命令；
3. `RagExecutionMiddleware` 在所有 LangGraph 节点进入/返回处理 checkpoint，并在 answer delta 前统一检查取消信号；节点实现不直接感知取消；
4. 检查命中后抛出业务级 `RagExecutionCancelled`，由 RAG Service 收口为 `generation.cancelled`；
5. 等待当前 Compute task 退出并完成清理；
6. 保存 checkpoint 和部分回答。

Generation Worker 不对正在执行的 `asyncio.Task` 调用 `cancel()`。`Task.cancel()` 会在任意 `await` 注入 `CancelledError`，不适合作为业务停止协议，也不能终止已经运行的线程或远程 Compute task。因此 RAG 调度停止和 Compute scope 取消必须分别协作传播。

## 11. Compute Worker 取消

### 11.1 状态机

```text
queued -> cancelled
running -> cancel_requested -> cancelled
running -> succeeded
running -> failed
```

`cancel_requested` 表示已收到命令但 Handler 尚未退出；`cancelled` 表示 Handler 已退出并已执行任务清理流程。

### 11.2 线程内协作取消

继续使用现有接口：

```python
context.raise_if_cancelled()
context.emit_delta(text)
context.report_progress(progress, message)
```

取消粒度：

| Operation | 取消安全点 |
| --- | --- |
| 流式 LLM | 每个 token / stream event |
| 非流式 LLM | 内部改为流式迭代，对外只聚合最终文本 |
| Embedding | 每个 micro-batch |
| Rerank | 每个 micro-batch |
| Audio batch | 每个 item 或现有 chunk 边界 |
| 排队任务 | 启动前直接取消 |

Python 线程不能安全强杀。已经提交给 GPU 的单个 kernel 或不可中断 native 调用允许执行到当前最小单元结束，结果随后丢弃，不再执行剩余批次。

### 11.3 `model.release()` 顺序

不能在 Cancel Consumer 线程中直接调用 `model.release()`，因为模型可能仍在另一个线程的 `complete()`、`stream()` 或 `predict()` 中使用。正确顺序是：

```text
收到 cancel
→ 设置 cancel_event
→ Handler 在安全点退出
→ 执行线程 finally 调用 operation.release()
→ Runtime 标记 cancelled
→ 发布 compute.task.cancelled
```

当前 Runtime 已经在执行函数的 `finally` 中调用 `operation.release()`，应保留此边界。若 release 失败，任务仍可进入 cancelled，但应附带 cleanup failure 元数据并将 Worker 标记为需要关注。

普通取消不强杀 Worker。若超过 cancel grace period 仍未退出，记录 `cancel_timeout` 并由 Worker 健康策略决定是否停止接单或重启实例；重启属于故障恢复，不是停止按钮的默认路径。

## 12. 数据库召回取消

PostgreSQL 和 psycopg 支持取消当前查询，但当前 RAG Retriever 在 `asyncio.to_thread()` 中临时获取 SQLAlchemy connection，外层没有正在执行连接的句柄，因此现状不能从 Generation Cancel Handler 精确取消 SQL。

第一版：

- 当前 SQL 返回后，RAG 在结果边界检查取消信号并丢弃结果，不再调度后续节点；
- 允许当前 SQL 返回，结果丢弃；
- 为召回查询设置合理的 `statement_timeout`；
- 不使用 `pg_terminate_backend`。

若实测长查询造成明显资源占用，再增加可取消 connection registry，并通过 psycopg `cancel_safe()` 取消当前 query。该能力与 Compute Worker 取消分开设计。

## 13. RAG 节点 Checkpoint

### 13.1 保存粒度

每个可恢复节点成功完成后，将该节点完成后的完整 `RagGraphState` 快照写入 Redis：

```text
generation:{generation_id}:rag-checkpoint:{node_hash}
  -> workflow_version
  -> node / completed
  -> input_hash
  -> state
```

State 中 Evidence 仅保存录音、chunk、时间范围和评分等引用信息，不保存 `chunk.text`；`retrieval_candidates[].text` 和派生的 `strategy_result.answer_context` 也不保存。恢复时根据引用从 `recording_search_chunks` 与 `utterance_segments` 重新加载正文并重建完整 State。

State 和 `completed` 状态通过单条 Redis `SET` 原子提交并设置 TTL。运行中被取消的节点不提交 completed checkpoint，恢复时从该节点重跑。

### 13.2 恢复规则

```text
route ✓ -> expand ✓ -> retrieval cancelled -> rerank -> answer
                       ↑ 从 retrieval 重跑
```

第一版将 retrieval 视为原子节点，不恢复 vector、lexical 等内部支路的部分结果。

Checkpoint 至少携带：

```text
generation_id
workflow_version
node
status
state（Evidence 正文除外）
input_hash
Redis TTL
```

知识库版本、Prompt/Workflow 版本、用户问题、过滤条件或附件变化时，旧 checkpoint 必须失效。

### 13.3 继续生成

“继续生成”创建新的 Generation run，并关联来源 run：

```text
new_generation.parent_generation_id = cancelled_generation_id
```

新 run 加载有效 checkpoint，从第一个未完成节点继续。Answer 阶段恢复的是 RAG 上下文和已有可见文本，不是模型内部 token、KV Cache 或推理状态。

## 14. 前端状态与用户体验

建议状态：

```text
pending -> streaming -> cancel_requested -> cancelled
pending/streaming -> completed
pending/streaming -> failed
```

用户点击停止后，Kafka ACK 成功即可：

- 按钮立即进入“正在停止”；
- 前端停止展示生成动画，但保留已经收到的内容；
- 后台继续等待 Compute 协作退出和 `model.release()`；
- 收到 `generation.cancelled` 后展示“继续生成”和“重新生成”。

“继续生成”复用 checkpoint；“重新生成”默认创建新 run 并重新执行完整 RAG。

## 15. 幂等与终态

- Cancel command 必须可以重复消费；重复取消已经终止的任务直接返回当前终态。
- Compute command 重投时，若 task 已终止则不重复执行。
- `cancelled_before_start` 仍需发布终态，不能静默丢弃。
- Generation 终态只在 RAG 停止调度、活动 Compute task 终止、checkpoint/部分内容落下后发布。
- Kafka state projector 和 PostgreSQL terminal projector 继续使用现有事件 ID/业务 ID 幂等机制。
- 第一版不引入节点 `attempt_id`：每次继续生成使用新的 Generation ID，每次 Compute 调用已有独立 task ID。只有未来允许同一 Generation 内原地并发重试同一节点时，才需要 execution revision/fencing token。

## 16. 分阶段实施

### Phase 1：可靠停止

1. [x] 新增 Generation Cancel Kafka Publisher、Topic、Retry/DLQ 和独立 Consumer；
2. [x] 将 Generation API 的 cancel 改为异步发布 Kafka 命令；
3. [x] RAG 安装共享取消信号，在节点、操作返回和流式 delta 边界协作停止 LangGraph；
4. [x] `ComputeCommand/ComputeTaskRequest` 增加通用 `execution_scope`；
5. [x] 补全 Compute Cancel Publisher、Consumer、Retry/DLQ；
6. [x] Compute Runtime 增加 scope 索引和取消前置检查；
7. [x] 提前取消与运行中取消均发布明确终态。

### Phase 2：缩短取消延迟

1. [x] 非流式 LLM 内部改为可中断流式执行；
2. [x] Embedding 和 Rerank 拆分 micro-batch；
3. [x] Online LLM 在退出迭代时关闭当前 HTTP response；
4. [x] 定义 cancel grace period、`cancel_timeout` 指标和 Worker 健康策略；
5. [x] 为 RAG SQL 设置 statement timeout。

### Phase 3：Checkpoint 与继续生成

1. [x] 增加 RAG 节点 checkpoint store；
2. [x] 在节点成功边界提交增量输出；
3. [x] 停止时保存部分 answer 和当前节点；
4. [x] 新 Generation run 加载有效 checkpoint；
5. [x] 前端增加“继续生成”和“重新生成”。

## 17. 验收标准

- Kafka ACK 前，API 不返回取消请求已接受；
- Redis 被清空后可以从 Kafka state 恢复取消屏障；
- 取消先于 command 被消费时，任务不会加载模型且产生 cancelled 终态；
- command 先执行时，取消能停止后续 RAG 节点并传播到活动 Compute task；
- 流式 LLM 在一个 token/stream event 边界内观察到取消；
- Embedding/Rerank 在一个 micro-batch 边界内观察到取消；
- `compute.task.cancelled` 只在 Handler 退出和 `operation.release()` 尝试完成后发布；
- 已完成 RAG 节点可从 checkpoint 复用，运行中取消节点恢复时重跑；
- Kafka 重投取消或任务命令不会产生重复执行或重复业务终态；
- 停止后已生成文本保留，继续生成使用新 Generation run。
