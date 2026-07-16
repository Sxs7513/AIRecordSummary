# Python 服务端与录音处理流水线改造方案

## 1. 目标

将当前由 Next.js API、Node.js scheduler、`worker_threads` 和 Python 脚本共同承担的服务端能力迁移为 Python 服务端。Next.js 保留为前端页面与交互层；Python 直接提供 HTTP API、管理数据库并执行异步录音处理。

本次不实现身份认证、工作空间、文档授权或资源分享。API 默认服务于单一可信使用范围，后续如需多用户能力，再在 Python API 层增加认证与授权边界。

核心目标是将录音处理由当前“一个任务完成后创建下一个任务”的串行实现，升级为自研、可观测、可重试、支持并行分支的流水线：

```text
Browser
  │
  ▼
Next.js（页面、组件、播放器）
  │ HTTP
  ▼
Python API（录音、任务、搜索、RAG）
  ├── PostgreSQL + pgvector
  ├── 对象存储 / 本地存储适配层
  └── Python Pipeline Workers
          ├── 音频标准化
          ├── 转写 ────────┐
          ├── 说话人分离 ──┼──► 对齐与话语切分 ──► 后续处理
          └───────────────┘
```

设计原则：

- Python 是唯一的业务后端；不再保留 Node.js 数据库访问、任务调度和 Python 子进程桥接代码。
- 流水线内核自行设计，不依赖 Celery、Temporal、Airflow 等第三方工作流框架。
- 每个阶段有明确输入、输出、状态、重试和进度事件，模型实现与编排逻辑解耦。
- 在 Python 中全面使用类型系统，保证 API、数据库、任务载荷和模型产物都有清晰契约。
- 第一期保留 PostgreSQL 和本地文件存储的兼容路径，后续可平滑替换为 S3/MinIO。

## 2. Python 框架与基础设施选型

### 2.1 Web API：FastAPI

采用 FastAPI 提供 `/api/v1` HTTP 接口。它与 Python 类型标注和 Pydantic 深度集成，适合生成稳定的请求/响应模型，并能自动产出 OpenAPI 文档。

建议接口范围：

- `POST /api/v1/recordings/uploads`：上传音频或创建上传会话
- `POST /api/v1/recordings`：创建录音并触发 pipeline run
- `GET /api/v1/recordings`、`GET /api/v1/recordings/{id}`：录音列表和详情
- `GET /api/v1/recordings/{id}/progress`：读取流水线阶段状态与进度
- `POST /api/v1/stage-runs/{id}/retry`：重试失败阶段
- `POST /api/v1/rag/query`、`POST /api/v1/rag/query/stream`：搜索与问答

### 2.2 类型系统：Python 3.14.4、Pydantic v2、Pyright

项目固定使用 Python 3.14.4，并开启严格静态检查。类型体系分三层：

- 原生类型与 `typing`：领域接口、泛型、`Protocol`、`TypedDict`、`NewType`。
- Pydantic v2：HTTP 请求/响应、配置、外部模型与 JSON 载荷的运行时校验。
- Pyright strict：CI 中执行静态检查，禁止未标注的公共函数、隐式 `Any` 和不安全的可空值访问。

建议约束：

- 所有公共函数、类字段、异步任务输入输出必须标注类型。
- `UUID`、`RecordingId`、`PipelineRunId`、`StageRunId` 在领域层使用 `NewType` 区分，避免将不同 ID 混用。
- pipeline stage 以泛型定义输入和输出，例如 `Stage[TranscriptionInput, TranscriptionArtifact]`。
- 数据库行、领域实体、API schema 不共用同一个模型，显式转换，防止存储细节泄漏到 API。

### 2.3 数据库：PostgreSQL、SQLAlchemy 2.0、Alembic

继续使用 PostgreSQL + pgvector。SQLAlchemy 2.0 负责数据库访问，Alembic 管理 schema migration。数据库事务仅用于业务数据、运行记录和 outbox 事件的一致提交；长时间模型推理绝不能在数据库事务中执行。

### 2.4 自研流水线运行时

不引入第三方工作流框架。项目内部实现一个轻量 Pipeline Runtime，负责：

- 将 pipeline 定义编译为 DAG，并校验无环、依赖完整、输出类型可匹配。
- 从数据库声明式领取 ready stage，不依赖进程内内存状态。
- 用 lease（租约）和心跳标识运行中的 stage；worker 崩溃后可回收超时任务。
- 按阶段策略执行重试、退避、超时、取消与失败传播。
- 通过事件表和 SSE/轮询 API 对前端报告进度。
- 基于幂等键复用已有 stage 输出，避免重复计算。

第一版由嵌入 Web 进程的 `PipelineCoordinator` 扫描并推进 `stage_runs`，再把实际 callable 交给 `ResourceScheduler`。数据库只保存流程状态、重试和 artifact，不承担资源 worker 的任务领取；这样无需新增 Redis 或消息队列。未来需要跨进程调度时，可在 runtime 的 dispatcher 接口后替换投递实现。

### 2.5 模型与媒体库

模型能力直接在 Python worker 内调用：

- 转写：优先 Qwen ASR（`Qwen/Qwen3-ASR-1.7B`），并通过 provider 保留 Whisper、SenseVoice 等可替换实现。
- 说话人分离：pyannote.audio。
- 声纹识别：SpeechBrain。
- 文本校正：pycorrector 或 LLM provider。
- embedding / 摘要：复用现有模型能力，但在 Python 侧用统一接口封装。

这些是模型/媒体依赖，不承担工作流编排职责。

## 3. Python 项目架构设计

建议在仓库根目录新增 `backend/`，前端和后端可独立启动、独立测试。

```text
backend/
├── pyproject.toml                 # 依赖、ruff、pyright、pytest 配置
├── alembic.ini
├── migrations/                    # Alembic 数据库迁移
├── src/
│   ├── main.py                    # FastAPI 应用工厂、路由装配、生命周期
│   ├── settings.py                # 根目录 .env 到强类型 Python 配置的适配
│   ├── api/
│   │   ├── router.py              # /api/v1 路由聚合
│   │   ├── recordings.py          # 录音、上传、详情、进度 API
│   │   ├── pipeline.py            # 重试、取消、运行详情 API
│   │   ├── search.py              # 搜索/RAG API
│   │   └── schemas/               # API 的 Pydantic 请求/响应模型
│   ├── domain/
│   │   ├── recordings/            # Recording、Transcript、Utterance 等实体与用例
│   │   ├── pipeline/              # PipelineRun、StageRun、Artifact 领域模型
│   │   └── search/                # 检索计划、证据、回答领域模型
│   ├── pipeline/
│   │   ├── definitions/           # 与业务无关的 DAG 定义模型
│   │   ├── contracts.py           # Stage 泛型协议、输入输出类型、事件类型
│   │   └── registry.py            # 通用 stage registry
│   ├── audio_processing/          # 录音处理领域：图装配、持久化与各具体节点
│   │   ├── definition.py          # recording_processing DAG 定义
│   │   ├── coordinator.py         # 录音 workflow 推进与资源任务提交
│   │   ├── repository.py          # recordings / pipeline_runs / artifacts 持久化
│   │   ├── projections.py         # 录音业务表投影
│   │   └── stages/                # ASR、分离、校正、索引、总结等业务实现
│   ├── providers/
│   │   ├── transcription/         # Whisper、SenseVoice、Qwen ASR 实现
│   │   ├── diarization/           # pyannote 实现
│   │   ├── speaker_identification/
│   │   ├── embedding/
│   │   └── llm/
│   ├── infrastructure/
│   │   ├── db/                    # SQLAlchemy ORM、repository、事务与 outbox
│   │   ├── storage/               # local / S3 对象存储适配
│   │   ├── media/                 # ffmpeg、音频探测与标准化
│   │   └── observability/         # 日志、指标、trace、进度事件
│   └── workers/
│       ├── main.py                # worker 进程入口
│       └── queues.py              # CPU/GPU worker 配置与并发限制
└── tests/
    ├── unit/                      # stage、DAG、provider contract 测试
    ├── integration/               # DB、API、worker 的集成测试
    └── fixtures/                  # 小型音频与模型产物 fixture
```

职责边界：

- `api` 不直接写 SQL 或调用模型，只将请求转给 domain use case。
- `domain` 不依赖 FastAPI、SQLAlchemy、pyannote 等基础设施；定义业务规则与端口接口。
- `pipeline` 不感知录音、模型、业务表或业务节点，只提供可复用的图与 stage 协议。
- `audio_processing` 组合录音领域用例、provider 与具体步骤，并负责其运行记录和投影。
- `providers` 屏蔽模型差异，并把输出转换为项目的标准类型。
- `infrastructure` 提供数据库、存储、媒体和观测实现。

## 4. 核心录音转文字流水线设计

### 4.1 DAG 与阶段定义

录音处理不是固定的 `if/else` 串行链，而是显式 DAG：

```text
create_recording
  │
  ▼
probe_and_normalize
  ├───────────────────────┐
  ▼                       ▼
transcribe             diarize
  │                       │
  └──────► align_and_build_utterances ◄──────┘
                      │
              identify_speakers（可选）
                      │
            correct_text（可配置、可选）
                 ┌────┴─────┐
                 ▼          ▼
          build_index    generate_summary
                 └────┬─────┘
                      ▼
                    finalize
```

- `probe_and_normalize`：校验格式、探测时长、转为统一采样率/声道，输出标准化音频 artifact。
- `transcribe` 与 `diarize`：并行处理同一个标准化音频。
- `align_and_build_utterances`：汇合两个结果，按时间轴给转写片段补全 speaker label，并生成页面与检索使用的 utterance。
- `identify_speakers`：在有已知人声样本时执行，作为可选插件，不影响匿名 speaker 的基本使用。
- `correct_text`：可关闭或替换实现，避免模型不可用时阻断基础转写。
- `build_index` 与 `generate_summary`：在文本最终版本产生后并行。
- `finalize`：聚合所有必需/可选阶段状态，更新录音展示状态。

### 4.2 自研插件系统

可以，并且插件系统适合解决“模型可替换”与“流水线可组合”两个问题。插件不使用动态扫描或任意 Python 代码上传；第一版采用**显式注册**，保证类型检查和可维护性。

插件有两类：

1. **能力插件（provider plugin）**：提供转写、分离、embedding、摘要等能力的多个实现，例如 `faster_whisper`、`qwen_asr`。
2. **阶段插件（stage plugin）**：把能力插件封装为 DAG 节点，例如 `transcribe`、`correct_text`；它声明依赖的输入 artifact 与产出的 artifact。

核心接口示意：

```python
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

class Stage(Protocol, Generic[InputT, OutputT]):
    name: str
    version: str
    input_type: type[InputT]
    output_type: type[OutputT]

    async def run(self, context: StageContext, input: InputT) -> OutputT: ...

class PluginRegistry:
    def register_stage(self, stage: Stage[Any, Any]) -> None: ...
    def get_stage(self, name: str) -> Stage[Any, Any]: ...
```

实际实现中，DAG 定义使用明确的 stage 名称和 typed artifact key；应用启动时将内置插件注册到 `PluginRegistry`。配置只选择已注册的插件名和参数，不能执行任意模块路径。第三方扩展需求出现后，再谨慎增加 Python package entry point 发现机制。

### 4.3 运行记录与状态

现有 `processing_jobs` 不足以记录并行依赖、输入产物和独立重试。新增以下表：

- `pipeline_runs`：一次录音处理或重新处理的总运行，保存 pipeline 版本、整体状态、开始/结束时间。
- `stage_runs`：录音工作流节点状态，保存依赖、attempt、错误、耗时与面向前端的进度；它不是系统 worker 直接领取的队列。
- `ResourceScheduler`：进程内 CPU/GPU 队列、优先级和并发控制；录音协调器和 RAG 工作流向它提交 callable，不单独持久化资源任务。
- `artifacts`：标准化音频、转写、分离结果、utterance、摘要等版本化产物，保存 URI、校验值、模型与参数摘要。
- `pipeline_events`：阶段状态和百分比进度，供 API 查询和 SSE 推送。
- `outbox_events`：在同一个数据库事务中记录待分发事件，避免“录音已创建但任务没有触发”。

状态建议：

- `stage_runs.status`：`pending / running / succeeded / retry_waiting / failed / cancelled / skipped`。
- `pipeline_runs.status`：`queued / running / succeeded / partial_failed / failed / cancelled`。
- `recordings.status`：面向 UI 的简化状态：`uploaded / processing / completed / failed`。

### 4.4 幂等、重试和恢复

- 幂等键为 `(recording_id, stage_name, stage_version, input_fingerprint)`；输入未变时可复用成功 artifact。
- 重试策略由 stage 声明：最大次数、超时、指数退避和哪些错误可重试。
- `PipelineCoordinator` 在节点依赖完成后提交 stage callable；它和 `PipelineRepository`、`PipelineExecutor`、`ArtifactStore` 一起属于 `pipeline.runtime`。资源调度器只负责分配 CPU/GPU，进程重启后由 `stage_runs` 的业务状态重新驱动未完成节点。
- `audio_processing` 只组装录音领域的 `PipelineDefinition`、stage registry 和投影 hooks；它不读取或写入 `pipeline_runs`、`stage_runs`、`artifacts` 等通用运行时表。
- 任一必需阶段失败使其下游等待或跳过；可选阶段失败记录为 `partial_failed`，不阻断可用的转写结果。
- 按 stage 重试时，系统仅重新运行该步骤和确实依赖它的下游步骤。

## 5. 迁移步骤

### 5.1 整体架构搭建

1. 新建 `backend/`，初始化 Python 3.14.4、FastAPI、SQLAlchemy、Alembic、Pydantic、Ruff、Pyright 和 Pytest。
2. 建立配置加载、日志、健康检查和 FastAPI 应用工厂。
3. 接入现有 PostgreSQL，使用 Alembic 管理后续 schema，不再由 Node.js 在运行时修改表结构。
4. 先完成数据库初始化、迁移执行和基础 repository；创建本地存储 adapter，兼容当前 `uploads/` 数据。
5. 建立 worker 进程入口与空的自研 runtime loop，验证 API、数据库和 worker 能独立启动。

验收：新开发环境可通过一条命令初始化数据库并启动 Python API/worker；Pyright strict、Ruff 和基础测试均通过。

### 5.2 录音转文字流水线设计与落地

1. 实现 pipeline DAG、stage contract、插件注册表、`pipeline_runs/stage_runs/artifacts/events/outbox` 数据模型。
2. 实现 PostgreSQL 任务领取、lease、心跳、重试、超时回收、事件投递与进度查询。
3. 迁移 `probe_and_normalize`、`transcribe`、`diarize`，验证转写和分离的并行执行与汇合。
4. 迁移 `align_and_build_utterances`，确保输出与当前详情页展示兼容。
5. 依次接入声纹识别、文本校正、embedding、摘要，并将它们改为可选/可配置插件。
6. 为成功复用、失败重试、worker 崩溃恢复、重复触发和长录音编写集成测试。

验收：单个失败 stage 可独立重试；worker 重启不会丢任务；同一录音的转写和分离并行；成功 artifact 不会因重复请求而重新计算。

### 5.3 API 迁移

1. 在 Python 中迁移录音上传、列表、详情、删除、标题/地点编辑、说话人档案、任务进度和重试 API。
2. 迁移搜索与 RAG API，保持已有响应格式或通过前端 adapter 平滑过渡。
3. Next.js 页面逐页改为请求 Python `/api/v1`，先迁移上传、列表、详情和进度展示。
4. 完成 RAG 页面迁移后，删除 `app/api/**`、`lib/db/**`、Node scheduler、`worker_threads` 与 Node 调 Python 脚本的代码。
5. 进行端到端回归：上传、流水线、失败重试、播放器定位、搜索、流式问答和删除录音。

验收：Next.js 不再包含业务 API、数据库访问和任务调度代码；所有业务请求均由 Python API 提供，前端功能与迁移前保持一致。
