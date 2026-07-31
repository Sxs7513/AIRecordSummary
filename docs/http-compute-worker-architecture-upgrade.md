# HTTP Compute Worker 架构升级方案

> 文档状态：现有 HTTP/内存 Worker 的历史实现设计。Redis + Kafka 目标架构将 Compute 主任务入口改为按资源 lane 的 Kafka Consumer，将实时 delta/progress 改为 Redis Streams，并保留 HTTP 仅用于 health、readiness、metrics 和管理接口。参见 [`redis-kafka-architecture-refactor.md`](./redis-kafka-architecture-refactor.md)。本文中的单实例、内存任务状态和内部 HTTP/SSE 主链路不再是目标终态。

## 1. 文档状态

- 状态：Phase 1～6 已实现（单实例、内存任务状态）
- 适用范围：Production API、录音处理 Pipeline、RAG、本地 LLM、本地 ASR、Embedding 等本地 CPU/GPU 计算
- 核心决策：使用内部 HTTP 服务承载原子计算任务；Worker 任务状态只保存在内存中，不新增 `compute_tasks` 数据表

本文记录 Compute Worker 的目标架构、接口协议、流式方案、失败语义、删除与取消规则，以及从当前实现迁移到目标架构的步骤。

## 2. 背景与现状问题

当前系统已经有 `PipelineCoordinator`、`PipelineExecutor` 和 `ResourceScheduler`：

```text
PipelineCoordinator
        │
        │ 将整个 Stage 作为闭包提交
        ▼
ResourceScheduler
        │
        ├── CPU thread
        └── GPU thread
```

当前做法的主要问题不是有没有 Worker，而是 Worker 的执行颗粒度过大：

- `audio_processing` 会把完整 Stage 交给 `ResourceScheduler`。
- Stage 同时包含业务编排、数据库访问、产物管理、进度更新和模型计算。
- RAG 中的检索、Embedding、LLM 生成也以闭包形式直接提交给进程内 scheduler。
- Worker 无法形成稳定的操作协议，后续很难单独部署。
- 本地模型的加载周期和 API 业务进程绑定，职责边界不清晰。
- 流式结果依赖进程内回调，不适合作为跨进程调用协议。

目标是保留 Pipeline 作为业务编排层，只把纯 CPU/GPU 计算抽成可注册、可调用的原子操作。

## 3. 已确认的架构决策

### 3.1 Worker 定位

Worker 是内部计算服务，不是持久化任务队列。

它负责：

- 注册原子 CPU/GPU 计算操作。
- 接收内部 HTTP 请求。
- 在内存中排队、执行和保存短期状态。
- 通过 SSE 输出 progress 和 delta。
- 管理本地模型的加载、复用与释放。
- 将最终大产物写入 `uploads`。

它不负责：

- 编排完整 AudioProcessing Stage。
- 决定 Pipeline 的下一节点。
- 更新录音、对话等业务聚合。
- 持久化 Compute Task 状态。
- 保证 Worker 重启后恢复未完成计算。
- 代替现有 `pipeline_runs`、`stage_runs`、`generation_runs` 等业务状态。

### 3.2 接受的失败语义

本方案明确接受：

- Worker 重启后，内存中的任务状态全部丢失。
- Worker 重启后，调用方重新提交尚未确认完成的计算。
- Production API 重启后，不恢复原来正在等待的 Worker HTTP 请求。
- Worker 任务重试记录不持久化。
- 同一个原子计算在异常情况下可能重复执行。

因此本方案提供的是 **best-effort、可重试且允许重复执行的计算调用**。在调用方仍存活时会主动重试，但不保证跨进程重启的 at-least-once，更不保证 exactly-once。

Pipeline 和 Generation 的业务状态仍可保留现有数据库持久化。API 或 Pipeline Stage 重新执行时，会重新发起原子计算。

### 3.3 不新增 Compute Task 数据表

本方案不创建：

```text
compute_tasks
compute_task_events
compute_task_dependencies
```

Worker 任务的 status、progress、实时订阅者和取消信号都保存在 Worker 进程内存中。

### 3.4 HTTP 而不是数据库队列

内部调用统一使用 HTTP：

- 普通结果使用 JSON。
- 流式结果使用 SSE。
- 状态恢复使用轮询。
- 取消使用 HTTP `DELETE`。
- 健康检查使用 `/healthz` 和 `/readyz`。

第一版不引入 gRPC。当前 Python、FastAPI、httpx 技术栈使用 HTTP + SSE 更简单，并且能与现有在线 LLM streaming 形式保持一致。

## 4. 总体架构

```text
┌───────────────────────────────────────────────────────────────┐
│ Production API                                                │
│                                                               │
│  Pipeline Stage / RAG / Application Service                   │
│                 │                                             │
│                 ▼                                             │
│  L1 WorkerClient（所有计算任务只通过该 API 发起）              │
└─────────────────┬─────────────────────────────────────────────┘
                  │ HTTP JSON / SSE
                  ▼
┌───────────────────────────────────────────────────────────────┐
│ L3 Compute Worker                                             │
│                                                               │
│  HTTP Router                                                  │
│       │                                                       │
│       ▼                                                       │
│  InMemoryTaskManager ─── ComputeOperationRegistry             │
│       │                          │                             │
│       ▼                          ▼                             │
│  ComputeExecutionPool    ComputeHandler                       │
│   ├── I/O queue           ├── ASR                              │
│   ├── CPU queue           ├── Online LLM                      │
│   ├── GPU high queue      ├── Diarization                     │
│   └── GPU normal queue    ├── Embedding                       │
│                          └── LLM（Local/Gemini/Zhipu）         │
└─────────────────┬─────────────────────────────────────────────┘
                  │
                  ▼
            uploads/compute-tasks/
```

业务编排与计算的边界：

```text
Pipeline Stage
    ├── 解析业务输入
    ├── 调用一个或多个原子 Compute Operation
    ├── 消费 progress / delta
    ├── 解释计算结果
    ├── 写业务数据库
    └── 生成 Pipeline Artifact

Compute Handler
    ├── 读取已经声明的输入文件
    ├── 执行单一 CPU/GPU 算法或模型推理
    ├── 报告 progress / delta
    └── 返回结构化结果或文件引用
```

## 5. 分层与代码组织

建议目录：

```text
backend/packages/
├── l1_foundation/
│   ├── worker/
│   │   ├── __init__.py
│   │   ├── contracts.py       # 完整类型化的请求、状态、事件、结果
│   │   ├── client.py          # HTTP 提交、轮询、SSE、重试、取消
│   │   ├── errors.py          # 稳定错误类型
│   │   └── sse.py             # SSE envelope 解析
│   └── llm/
│       ├── local.py           # 本地模型实现，只由 Worker Handler 调用
│       ├── gemini.py          # Gemini provider，只由 Worker Handler 调用
│       ├── zhipu.py           # 智谱 provider，只由 Worker Handler 调用
│       ├── worker_handler.py  # LLM 任务协议、序列化和 Handler
│       └── contracts.py       # provider 内部统一流式与非流式接口
│
├── l2_core/
│   ├── audio_processing/
│   │   ├── worker_tasks.py    # 音频与 Embedding 的类型化任务构造
│   │   └── stages/            # 业务编排；计算入口由 L3 handler 组装
│   └── rag/
│       └── ...                # 检索编排与业务转换
│
└── l3_app/
    ├── compute_worker/
    │   ├── app_factory.py
    │   ├── routes.py
    │   ├── runtime.py
    │   ├── registry.py
    │   └── registry_factory.py # 注册 L1 提供的具体 Handler
    └── production-api/
        └── app_factory.py
```

依赖方向：

- L1 Worker 定义通用任务、Handler、HTTP 和 SSE 契约，不依赖具体业务。
- L1 LLM 只依赖 L1 Worker 的 Handler 侧类型，并实现 LLM 任务协议与 Handler；不依赖 Worker Client、URL 或 L3。
- L2 提供音频、RAG 等领域内可复用的纯计算实现和业务编排。
- L3 组装 Worker HTTP 服务，并把 operation name 注册到具体 handler。
- Production API 只依赖 L1 client，不直接访问 Worker 的内存对象。

## 6. 核心命名

不再使用含义模糊的 `WorkerItem`，统一使用：

| 名称 | 职责 |
|---|---|
| `ComputeWorker` | HTTP 计算服务及其运行时。 |
| `ComputeOperation` | 稳定、带版本号的原子操作标识。 |
| `ComputeHandler` | 某个原子操作的具体执行器。 |
| `ComputeOperationRegistry` | 将 `operation + version` 映射到 handler。 |
| `ComputeTask` | Worker 内存中的一次调用实例。 |
| `WorkerClient` / `SyncWorkerClient` | L1 中供 Pipeline、RAG、Application Service 直接使用的类型化 HTTP 客户端。 |
| `ComputeEvent` | SSE 输出的 progress、delta 和终态事件。 |

## 7. 原子操作边界

一个 `ComputeHandler` 必须满足：

- 只做一种明确的 CPU/GPU 计算。
- 输入和输出可以完整序列化。
- 不决定 Pipeline 下一步。
- 不直接修改录音、对话、Pipeline 等业务表。
- 不持有业务 Service。
- 不在内部创建另一个业务 Stage。
- 可以重复执行；重复执行不应破坏业务数据。
- 大输入通过 `uploads` 中的路径引用传递，不通过 JSON 上传整段音频。
- 大输出写入 `uploads/compute-tasks/{task_id}/`。

### 7.1 候选操作

第一阶段可抽取：

| Operation | 资源 | 说明 |
|---|---|---|
| `diarization.pyannote.infer` | GPU | 只执行 Pyannote 推理；片段平滑和合并留在 Stage。 |
| `asr.qwen_asr.infer_batch` | GPU | 接收准备好的窗口列表，逐条执行 Qwen ASR 推理。 |
| `asr.funasr_nano.infer_batch` | GPU | 接收准备好的窗口列表，逐条执行 FunASR 推理。 |
| `alignment.qwen.infer_batch` | GPU | 只执行 ForcedAligner；说话人归属和结果整理留在 Stage。 |
| `embedding.encode` | GPU | 输入文本并返回向量；数据库写入仍由业务层完成。 |
| `llm.generate.local` | GPU | 本地 Qwen 的流式或非流式生成。 |
| `llm.generate.gemini` | I/O | Gemini 在线生成。 |
| `llm.generate.zhipu` | I/O | 智谱在线生成。 |
| `llm.generate_batch.*` | GPU/I/O | 一个任务携带多个 prompt，Worker 内逐条推理。 |

以下内容不应作为一个 Compute Operation：

- 完整的 `recording_processing` Pipeline。
- 完整的 `generate_summary` Stage。
- 完整的 RAG Graph。
- “生成 embedding 并写数据库索引”。
- “生成总结并更新 `recording_summaries`”。
- “检索证据、调用模型并保存对话消息”。

这些流程应保留在 L2/Application/Pipeline 中，只把内部模型调用或纯计算步骤交给 Worker。

轻量 CPU 转换不要求全部远程化。是否调用 Worker 应由隔离、耗时、并发控制和部署需求决定，而不是仅因为代码使用了 CPU。

模型 operation 的“原子”边界是一次纯模型推理，可以接收多个 item。当前约定是 HTTP task batch size 可以大于 1，但实际模型 inference batch size 固定为 1：Worker 在一个任务内复用同一模型并按输入顺序逐条推理。音频裁剪、FFmpeg 增强、窗口规划、结果平滑、说话人归属等非模型逻辑留在 Stage 或 engine。

## 8. HTTP API

### 8.1 提交任务

```http
POST /internal/v1/compute/tasks
Content-Type: application/json
X-Internal-Token: <service-token>
```

请求：

```json
{
  "task_id": "a client-generated UUID",
  "operation": "asr.qwen_asr.infer_batch",
  "operation_version": "1",
  "resource_queue": "gpu_high",
  "wait_for_subscriber": true,
  "input": {
    "audio_uri": "recordings/{recording_id}/audio.wav",
    "provider": "qwen_asr",
    "language": "zh"
  }
}
```

流式调用设置 `wait_for_subscriber=true`。Worker 接受任务后先保持 `queued`，等 `/events` SSE 订阅建立成功再开始执行，避免 `POST` 返回到 SSE 建连之间丢失首批 delta。它只解决首次订阅竞态，不提供断线后的重连、补发或续传。

返回：

```http
HTTP/1.1 202 Accepted
```

```json
{
  "task_id": "UUID",
  "status": "queued",
  "status_url": "/internal/v1/compute/tasks/{task_id}",
  "events_url": "/internal/v1/compute/tasks/{task_id}/events"
}
```

`task_id` 必须由调用方在第一次请求前生成，不能由 Worker 收到请求后才生成。这样即使 POST 响应丢失，调用方仍可查询或使用同一个 ID 重试。

### 8.2 查询状态

```http
GET /internal/v1/compute/tasks/{task_id}
```

```json
{
  "task_id": "UUID",
  "operation": "asr.qwen_asr.infer_batch",
  "status": "running",
  "progress": 0.42,
  "message": "正在识别第 8 个音频窗口",
  "created_at": "2026-07-31T08:00:00Z",
  "started_at": "2026-07-31T08:00:01Z",
  "finished_at": null,
  "result": null,
  "error": null
}
```

### 8.3 订阅事件

```http
GET /internal/v1/compute/tasks/{task_id}/events
Accept: text/event-stream
```

SSE 示例：

```text
event: progress
data: {"progress":0.42,"message":"正在识别第 8 个音频窗口"}

event: delta
data: {"text":"本次会议主要讨论了"}

event: completed
data: {"artifact_uri":"compute-tasks/{task_id}/result.json"}
```

### 8.4 取消任务

```http
DELETE /internal/v1/compute/tasks/{task_id}
```

返回当前状态。取消是协作式的：

- `queued` 任务可立即转为 `cancelled`。
- `running` 任务先转为 `cancel_requested`。
- Handler 在 token、音频窗口、segment 或其他安全边界检查取消信号。
- 无法中断的底层调用完成后必须丢弃结果，不再发布 `completed`。

### 8.5 健康接口

```text
GET /healthz
GET /readyz
GET /metrics
```

- `healthz`：进程存活。
- `readyz`：scheduler 已启动、uploads 可访问，必要的模型运行环境可用。
- `metrics`：任务数、排队时间、执行时间、失败数、取消数、GPU/CPU 队列深度。

## 9. 类型契约

L1 中的 Python 类型必须完整，禁止将通用协议退化为无约束的 `dict[str, Any]`。

示意：

```python
from collections.abc import AsyncIterator
from typing import Generic, Literal, Protocol, TypeVar
from uuid import UUID

InputT = TypeVar("InputT")
ResultT = TypeVar("ResultT")

type ComputeTaskStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancel_requested",
    "cancelled",
]


class ComputeCommand(Generic[InputT]):
    task_id: UUID
    operation: str
    operation_version: str
    resource_queue: Literal["io", "cpu", "gpu_high", "gpu_normal"]
    input: InputT


class WorkerClient(Protocol):
    async def execute(
        self,
        command: ComputeCommand[InputT],
        *,
        result_type: type[ResultT],
    ) -> ResultT: ...

    def stream(
        self,
        command: ComputeCommand[InputT],
        *,
        result_type: type[ResultT],
    ) -> AsyncIterator[ComputeEvent[ResultT]]: ...
```

每个 operation 使用自己的输入和输出模型，例如：

```text
TranscribeInput     -> TranscribeResult
DiarizeInput        -> DiarizeResult
EmbeddingInput      -> EmbeddingResult
LlmGenerateInput    -> LlmGenerateResult
```

HTTP envelope 通用，operation 的 `input` 和 `result` 必须在两端按注册类型校验。

## 10. Worker 内存状态

Worker 为每个任务保存：

```text
task_id
operation / operation_version
request_hash
resource_queue
status
progress / message
result
error
cancel_event
subscriber queues
created_at / started_at / finished_at
```

约束：

- 同一个 `task_id` 和相同 `request_hash` 重复提交时返回已有任务。
- 同一个 `task_id` 但请求内容不同，返回 `409 Conflict`。
- 已完成任务保留固定 TTL，建议默认 30 分钟。
- Worker 设置全局最大任务数和最大并发数，超过时返回 `429` 或 `503`。
- 清理任务状态时同步清理订阅者和临时文件。

第一版只允许一个 Worker 实例。内存任务状态无法在多个实例之间共享；如果提交和查询落到不同实例，会得到错误的 `404`。

## 11. SSE 与流式处理

### 11.1 统一事件

建议事件类型：

| Event | 说明 |
|---|---|
| `queued` | 已进入 Worker 内存队列。 |
| `started` | Handler 开始执行。 |
| `progress` | 百分比和可展示阶段信息。 |
| `delta` | LLM token batch 或 ASR 增量文本。 |
| `retrying` | Worker 进程内重试。 |
| `completed` | 最终结果已经可读。 |
| `failed` | 最终失败。 |
| `cancelled` | 已停止或结果已丢弃。 |
| `heartbeat` | 长时间没有业务事件时维持连接。 |

### 11.2 内部 SSE 不做断线续传

Worker 到 Production API 的 SSE 只负责当前连接上的实时传输：

- 不定义内部 event ID。
- 不接受 `Last-Event-ID`。
- 不保存事件历史或 ring buffer。
- 不补发已经错过的 progress 和 delta。
- 不实现内部 SSE 自动重连。

内部 SSE 一旦断开，`WorkerClient` 抛出类型化的 stream disconnected 异常，并尽力取消对应任务。本次原子调用由上层按失败处理；Pipeline、Generation 或用户操作可以决定是否重新发起一次完整调用。

断线时 Worker 可能仍在执行底层不可中断操作。即使它最终生成了结果，原调用方也不再消费该结果，之后由 TTL 和 Storage Cleanup 清理任务状态及孤立产物。

真正面向用户的断线续传只保留在 Production API 到前端这一层，由现有 Generation snapshot、`generation_events` 和前端 `Last-Event-ID` 机制负责。

### 11.3 流式不是最终产物

`delta` 用于实时体验，不作为最终可信结果。

Worker 完成任务时必须：

1. 将完整结果写入临时文件。
2. flush 并关闭文件。
3. 原子 rename 到最终路径。
4. 更新内存中的 `result` 和 `status=succeeded`。
5. 发布 `completed` 事件。

只有发布 `completed` 后，调用方才能认为 ASR 或本地 LLM 操作完成。

## 12. uploads 产物协议

建议目录：

```text
uploads/
└── compute-tasks/
    └── {task_id}/
        ├── input-manifest.json
        ├── result.json
        └── temporary files
```

规则：

- HTTP 只传相对 storage URI，不传绝对路径。
- Worker 必须验证解析后的路径仍位于配置的 storage root 下，防止路径穿越。
- 输入文件由业务层或上一 Pipeline Stage 准备。
- 大结果返回 `artifact_uri`；小结果可以直接放在状态响应中。
- 最终文件使用临时文件加原子 rename。
- Worker 启动时清理超过 TTL 的 `.tmp` 文件。
- 录音删除或任务取消产生的孤立目录由 Storage Cleanup 定期清理。

## 13. 重试、超时和幂等

### 13.1 调用方重试

`WorkerClient` 负责以下重试：

- Worker 尚未启动、连接失败：退避后重试。
- `429`、`502`、`503`、`504`：退避后重试。
- POST 响应超时：使用原 `task_id` 查询状态；不存在时使用相同请求重新提交。
- SSE 建立前失败：按普通 HTTP 临时错误重试。
- SSE 已经开始传输后断开：不续传、不自动重连，抛出类型化异常并交给上层处理。
- 状态查询返回 `404`：认为 Worker 状态已丢失，重新提交计算。

业务错误和确定性输入错误不应自动重试：

```text
400 invalid_input
404 input_artifact_not_found
409 task_id_conflict
422 unsupported_operation_version
```

### 13.2 Worker 内部重试

Worker 可以对少量明确的临时错误做进程内重试，例如瞬时显存不足或模型服务暂时不可用，但：

- 重试次数不持久化。
- 每次重试必须记录 INFO 日志并发出 `retrying` 事件。
- Worker 重启后由调用方重新提交，不恢复重试次数。
- 重试必须有上限，避免一个任务永久占用资源。

### 13.3 重复执行

任务可能因为网络超时、Worker 重启或 API 重启而重复执行。Handler 必须将输出限制在 task 专属目录，业务数据库更新仍由 Stage 在获得最终结果后幂等执行。

## 14. ASR 完成感知

ASR Stage 的目标流程：

```text
1. 生成 task_id
2. 在 Stage/engine 中准备音频窗口并 POST `asr.*.infer_batch`
3. 订阅 SSE
4. 将 progress 映射到 StageContext
5. 可选地转发 ASR delta
6. 收到 completed
7. 读取 result/artifact_uri
8. 构造 StageResult
9. 由 PipelineRepository 标记 Stage succeeded
```

异常流程：

- SSE 已经开始传输后断开：本次 ASR 原子调用失败，尽力取消 Worker task，再由 Pipeline 重试策略决定是否重新执行。
- Worker 返回 `failed`：抛出类型化异常，由 Pipeline 现有重试策略决定 Stage 是否重试。
- Worker 返回 `404`：重新提交原子任务。
- API 重启：原等待关系丢失；本方案不保证自动恢复。后续如果 Pipeline Stage 被人工或其他恢复机制重新执行，则重新创建原子 ASR 任务。

Worker 不直接更新 `stage_runs`。它只报告原子计算结果，Stage 负责将结果投影回 Pipeline。

## 15. 本地与在线 LLM

L2 不再持有 `LanguageModel`，也不调用 `model.complete()` 或 `model.stream()`。所有 LLM 发起方直接把类型化任务交给 L1 Worker API：

```text
L2 RAG / Summary / Correction / Topic Detection
  └── WorkerClient.execute/stream
        └── HTTP llm.generate.{provider}
              └── L3 registry
                    └── L1 LlmWorkerHandler
                          ├── Local Qwen（GPU queue）
                          ├── Gemini（I/O queue，不占 CPU/GPU lane）
                          └── Zhipu（I/O queue，不占 CPU/GPU lane）
```

约束：

- 本地 Qwen 必须通过 Worker，不能在 Production API 中直接加载。
- Gemini、智谱与本地模型使用相同 Worker 任务协议；在线 provider 进入独立 I/O lane，不占 CPU/GPU lane。
- provider 是序列化输入的一部分，operation 为 `llm.generate.local|gemini|zhipu`。
- 非流式调用使用 `WorkerClient.execute`，流式调用使用 `WorkerClient.stream` 消费统一 `ComputeEvent`。
- 模型加载、释放日志由 L1 的本地模型实现记录。
- Worker HTTP 请求、任务开始、结束、重试和取消日志使用 `worker` logger。
- LLM 请求和 provider 生命周期日志继续使用 `llm` logger，级别为 INFO。

## 16. RAG 迁移

RAG Graph 保留在 L2，不整体移入 Worker。

应迁移的调用：

- 本地 query embedding -> `embedding.encode`。
- 回答生成（本地或在线）-> L1 `WorkerClient` 的 `llm.generate.{provider}` 任务。
- 其他明确的本地模型推理 -> 对应原子 operation。

不迁移：

- 数据库 lexical/vector 查询。
- evidence fusion。
- route 和 filter 业务逻辑。
- citation 校验。
- 对话消息和 Generation 状态更新。

RAG Graph 不再注入模型对象，只注入 L1 `WorkerClient` 和 LLM 任务配置。

## 17. AudioProcessing 迁移

Pipeline 图和 Stage 依赖关系继续由现有 Pipeline 基础设施管理。

迁移前：

```text
PipelineCoordinator
  -> ResourceScheduler.submit(queue, execute_whole_stage)
```

迁移后：

```text
PipelineCoordinator
  -> PipelineExecutor.execute(stage)
       -> Stage orchestration
            -> WorkerClient.execute/stream(atomic operation)
```

因此：

- `PipelineCoordinator` 不再根据 Stage 的 `resource_queue` 把整个 Stage 放入 `ResourceScheduler`。
- Stage 的 `resource_queue` 字段已删除。
- 真正的 resource queue 由每个 Compute Operation 声明。
- 不需要 GPU 的业务编排运行在 API/Coordinator 的普通异步上下文。
- 阻塞但不值得远程化的轻量 CPU 操作可以使用 `asyncio.to_thread`。

## 18. Generation 持久化的边界

现有 `generation_runs` 和 `generation_events` 属于面向业务和用户的生成状态，与 Worker 内存任务不是同一个概念。

可以保留：

- 对话消息的持续生成状态。
- 用户刷新页面后的回答恢复。
- 已生成内容和引用的持久化。
- 面向用户的 SSE sequence。

调用链为：

```text
Worker SSE delta
      │
      ▼
L1 LLM stream
      │
      ▼
GenerationEmitter
      ├── 面向当前连接实时发布
      └── 按现有策略持久化业务消息
```

因此“Worker 不持久化任务”不等于“用户可见回答不持久化”。两者应保持解耦。

## 19. 删除录音、对话与任务取消

Worker 没有 Compute Task 数据表，因此不存在数据库级联删除。

### 19.1 已知 task_id

删除业务对象前：

1. 请求取消相关 Worker task。
2. `queued` 任务立即取消。
3. `running` 任务协作式取消。
4. 可以短暂轮询终态。
5. 删除业务数据和业务存储目录。

### 19.2 task_id 已丢失

如果 API 已重启或没有保存 task ID：

- 允许直接删除录音或对话。
- Worker 后续完成时，其结果不会再被业务层采纳。
- Worker 输出成为孤立文件，由 Storage Cleanup 定期清理。

Handler 必须只写 task 专属目录，不直接写录音业务表，从而保证孤立结果不会污染业务状态。

## 20. 运行与部署

### 20.1 当前独立部署

Worker 已经是独立 Python 进程，不再挂载到 Production API：

```text
Production API
      │
      │ private HTTP
      ▼
Compute Worker
      ├── CPU queue
      └── GPU queue
```

部署约束：

- 两个进程共享 `uploads`，或切换为双方可访问的对象存储。
- `COMPUTE_WORKER_BASE_URL` 指向内部服务地址。
- 使用私网、service token 或 mTLS 保护内部接口。
- Worker 可以继续保持单实例。

本地开发分别启动：

```bash
npm run dev:compute-worker
npm run dev:production-api
```

Worker 默认监听 `127.0.0.1:8010`。Production API 启动时强制检查 `/readyz`；Worker 不可用时直接启动失败，不提供绕过 Worker 的降级路径。

### 20.3 多实例之前必须重新评估

内存状态模式不支持普通的无状态水平扩容。需要多个 Worker 时，至少选择一种方案：

- 粘性路由。
- Worker Gateway 持有 task -> worker 路由。
- 按 task ID 一致性哈希。
- 引入 Redis Streams、消息队列或数据库共享状态。

在此之前不应直接把单实例 Worker 扩成多个副本。

## 21. 配置建议

新增配置建议：

```text
COMPUTE_WORKER_BASE_URL=http://127.0.0.1:8010/internal/v1/compute
COMPUTE_WORKER_HOST=127.0.0.1
COMPUTE_WORKER_PORT=8010
COMPUTE_WORKER_INTERNAL_TOKEN=...
COMPUTE_WORKER_COMPLETED_TTL_SECONDS=1800
COMPUTE_WORKER_MAX_TASKS=100
COMPUTE_WORKER_CPU_CONCURRENCY=1
COMPUTE_WORKER_GPU_CONCURRENCY=1
COMPUTE_WORKER_CONNECT_TIMEOUT_SECONDS=5
COMPUTE_WORKER_STATUS_TIMEOUT_SECONDS=10
COMPUTE_WORKER_STREAM_READ_TIMEOUT_SECONDS=0
```

SSE 是长连接，不能沿用普通请求的短 read timeout。应使用：

- 有限 connect timeout。
- SSE read timeout 关闭或设置为明显大于 heartbeat 间隔的值。
- Worker 定期发送 heartbeat。

## 22. 日志与可观测性

Worker logger 统一使用 `worker`，INFO 日志至少包括：

```text
worker started/stopped
task accepted
task queued
task started
task retrying
task succeeded
task failed
task cancel requested
task cancelled
task evicted
```

公共字段：

```text
task_id
operation
operation_version
resource_queue
attempt
duration_ms
queue_wait_ms
error_type
```

禁止在日志中输出：

- API key。
- 完整 prompt。
- 完整转录文本。
- 用户音频内容。

指标：

- 当前各状态任务数。
- CPU/GPU queue depth。
- 各 operation 的执行次数和耗时。
- 连接失败、SSE 中断、任务重试次数。
- 首个 delta 延迟。
- 取消延迟。

## 23. 错误协议

统一错误结构：

```json
{
  "error": {
    "code": "model_overloaded",
    "message": "GPU worker is overloaded",
    "retryable": true,
    "details": {}
  }
}
```

稳定错误码示例：

```text
invalid_input
unsupported_operation
unsupported_operation_version
input_artifact_not_found
task_id_conflict
queue_full
model_load_failed
model_overloaded
compute_failed
cancelled
worker_restarted
```

L1 `WorkerClient` 将 HTTP 和 SSE 错误映射为类型化异常，上层不直接判断原始状态码和字符串。

## 24. 安全边界

- Worker API 只接受内部调用。
- 本地使用 loopback 地址并校验 service token；部署环境使用私网服务地址。
- 独立部署后只监听私网。
- 所有 storage URI 做 root containment 校验。
- operation 必须来自静态 registry，不能接收任意 Python callable 或模块路径。
- 输入大小、字符串长度、批量条数和输出大小必须有限制。
- 错误响应不能包含本地绝对路径和敏感环境变量。

## 25. 迁移步骤

### Phase 1：建立协议和 Worker 骨架（已完成）

1. 创建 L1 Compute contracts、client、errors 和 SSE parser。
2. 创建 L3 Compute Worker app、内存 TaskManager、registry 和 HTTP routes。
3. 在 Worker 内建立专用 `ComputeExecutionPool`，不复用或兼容旧 `ResourceScheduler`。
4. 接入 health、ready、日志和基础指标。
5. 在 Production API 中以单进程模式启动 Worker Runtime。

Worker 是 Production API 的必选基础设施，不提供 `COMPUTE_WORKER_ENABLED` 开关。Worker Runtime 启动失败时，Production API 必须启动失败，不能降级为绕过 Worker 直接执行本地模型。

### Phase 2：迁移 LLM（已完成）

1. 注册 `llm.generate`。
2. L1 增加 `WorkerLLMProvider`。
3. 本地 Qwen 的流式和非流式调用改为 Worker HTTP。
4. Gemini、智谱也通过相同 Worker HTTP 契约调用，并使用 I/O lane。
5. 验证 RAG、文本润色和 Summary 的统一 stream/complete 接口。
6. 直接移除 Production API、Pipeline、RAG 和 Summary regeneration 对 `ResourceScheduler` 的依赖，不保留过渡兼容装配。

### Phase 3：迁移 ASR（已完成）

1. 注册 `asr.qwen_asr.infer_batch` 和 `asr.funasr_nano.infer_batch`。
2. ASR Stage 只保留输入准备、progress 映射和 StageResult 构造。
3. ASR progress 和增量文本使用 SSE。
4. Worker 先原子提交最终产物，再发出 `completed`。
5. 验证 SSE 断线失败、状态轮询、取消和 Worker 重启后的重做。

### Phase 4：迁移其他重计算（已完成）

依次迁移：

1. Diarization。
2. Embedding encode。
3. 音频预处理和对齐中值得隔离的重计算。
4. RAG 中仍直接运行在 API 进程里的本地 Embedding 等重计算。

已迁移音频标准化、ASR 前置处理、Diarization、Qwen/FunASR、对齐、录音索引 Embedding 与 RAG query Embedding。

### Phase 5：移除旧调度方式（已完成）

1. 清理已经失去执行作用的 Pipeline Stage `resource_queue` 元数据。
2. 删除数据库和 Definition 中只服务于旧 Stage 调度方式的字段。
3. 确认所有需要资源隔离的重计算都已迁移为 Compute Operation。

### Phase 6：独立部署（已完成）

1. 将 Worker Router 和 Runtime 移到独立进程。
2. 配置共享 storage。
3. 切换 `COMPUTE_WORKER_BASE_URL`。
4. 加入内部鉴权和部署健康检查。
5. 保持 HTTP 契约不变。

## 26. 测试策略

### 26.1 单元测试

- operation registry 注册、重复和版本校验。
- task 状态转换。
- 相同 task ID 幂等提交。
- task ID 冲突。
- SSE 实时广播和订阅者清理。
- 协作式取消。
- TTL 清理。
- 类型化输入输出和错误映射。
- storage URI 路径穿越防护。

### 26.2 HTTP 集成测试

- 提交、查询、完成。
- SSE progress、delta、completed。
- POST 响应超时后的状态查询。
- SSE 传输中断后抛出稳定错误且不重连。
- queue full。
- Worker 重启后返回 `404` 并重新提交。
- 取消 queued/running 任务。

### 26.3 业务回归

- AudioProcessing Pipeline 全链路。
- ASR progress 和最终转录。
- 文本校正。
- Summary 流式输出。
- RAG 本地 LLM 流式回答。
- Gemini、智谱在线调用进入 Worker 的 I/O lane，且不占 CPU/GPU lane。
- 删除录音/对话时的取消和孤立文件清理。

## 27. 验收标准

- Pipeline Stage 不再整体运行在 CPU/GPU scheduler 中。
- Worker registry 只注册原子 CPU/GPU operation。
- 本地 Qwen LLM 只由 Worker Runtime 加载，Production API 业务代码不直接加载。
- 本地 LLM 和 ASR 同时支持流式、非流式调用。
- Worker 内部 SSE 不提供断线续传；传输中断后返回稳定错误。
- 前端断线续传继续由现有 Generation 事件机制负责。
- HTTP POST 超时后可通过调用方生成的 task ID 查询或重试。
- Worker 重启后调用方能识别任务丢失并重新执行。
- 最终产物先原子落盘，再发布 `completed`。
- Worker 不直接修改录音、对话、Pipeline 和 Generation 业务状态。
- 不新增 Compute Task 数据表。
- 当前保持单 Worker/单 ASGI 进程限制。

## 28. 后续演进触发条件

出现以下任一情况时，应重新评估纯内存 Worker：

- 需要多 Worker 实例或水平扩容。
- 任务不能因 Worker 重启而丢失。
- ASR 单任务执行时间长到重新执行成本不可接受。
- 需要跨 API 重启恢复等待。
- 需要严格审计每次计算任务。
- 服务端内部也需要跨连接恢复流式内容。
- 需要跨机器动态调度 GPU。

届时优先考虑引入 Redis Streams、消息队列或持久化任务表，而不是在当前 HTTP 协议中逐步加入不完整的持久化逻辑。HTTP operation、SSE event 和 artifact 契约仍可以继续复用。

## 29. 参考

- [NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html)：HTTP/REST、gRPC、健康检查和推理服务模式。
- [Hugging Face Text Generation Inference](https://huggingface.co/docs/text-generation-inference/en/conceptual/streaming)：基于 SSE 的 token streaming。
