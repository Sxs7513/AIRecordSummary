# Python 后端分层 Monorepo 架构设计

## 1. 文档目标

本文档定义 Python 后端后续重构的目标架构。核心目标是将当前集中在 `backend/src` 和 `backend/scripts` 中的代码，重组为边界清晰、可独立测试、可独立安装的三层 Monorepo：

1. `packages/l1_foundation`：与具体业务无关的基础能力包集合；
2. `packages/l2_core`：录音处理、RAG、评测等核心业务能力包集合；
3. `packages/l3_app`：生产 API、评测 API、训练 API、Worker 和 CLI 等可运行入口集合。

本次重构首先解决代码与依赖边界问题，不要求立即拆分仓库、数据库、机器或 GPU。第一阶段仍然可以：

- 使用同一个 Git 仓库；
- 使用同一个 PostgreSQL 数据库；
- 部署在同一台机器；
- 使用一个 GPU；
- 通过统一调度器串行执行 ASR 评测与训练。

### 1.1 当前实施状态

截至当前版本：

- 已建立 `packages/l1_foundation`、`packages/l2_core` 和 `packages/l3_app` 物理目录；
- L1/L2/L3 共享 `backend/pyproject.toml` 和 `backend/.venv`；
- 只有存在依赖冲突的 Qwen HF Trainer 保留独立 `pyproject.toml` 和 `.venv`；
- 原 `backend/src` 下的实现已全部物理迁入对应的 L1/L2/L3；
- 生产代码统一使用 `l1_foundation.*`、`l2_core.*` 显式分层 namespace；
- 已拆出 production、evaluation、training 三套 FastAPI 入口；
- 已拆出 production worker 和单 GPU ASR compute worker；
- Qwen LoRA Trainer 已从 `backend/scripts` 迁到 `packages/l2_core/trainers/qwen-asr-lora`；
- Trainer 已改用 `Qwen/Qwen3-ASR-1.7B-hf` 和 Transformers 原生训练接口；
- Trainer 使用独立 `.venv`，生产 `qwen-asr` 推理继续使用 `backend/.venv`；
- 评测 HF base/adapter 时，通过常驻子进程协议调用 Trainer 环境，避免每个 case 重复加载模型；
- `npm run dev:python-web` 作为 production API 的兼容命令别名保留。

当前 namespace 状态记录在 `backend/MONOREPO.md`，旧的无层级顶层 import
和临时 `airecord_*` facade 已删除。

## 2. 架构原则

### 2.1 物理分层

`backend/packages` 是 L1、L2、L3 统一的 monorepo 物理根。`l1_foundation`
和 `l2_core` 是主环境中的真实 Python package；`l3_app` 是同一根目录下
的可执行应用集合，不作为业务代码的反向 import 目标。

```text
backend/
└── packages/
    ├── l1_foundation/
    ├── l2_core/
    └── l3_app/
```

`packages` 本身不进入 import 名称，其下一层是顶层 Python package：

```text
目录：packages/l1_foundation/pipeline
import：l1_foundation.pipeline
```

### 2.2 单向依赖

全局依赖方向固定为：

```text
packages/l3_app → l2_core → l1_foundation
```

禁止：

- L1 导入 L2 或 L3；
- L2 导入 L3；
- 业务代码反向导入 FastAPI route；
- 两个 L2 包相互导入形成循环依赖；
- 为了方便而把业务模型、SQL 或重型模型依赖放进 L1。

### 2.3 入口层只负责装配

L3 负责：

- HTTP、CLI、进程启动等协议适配；
- 鉴权和请求上下文建立；
- 依赖注入；
- L2 service 的实例化；
- 事务、队列和运行时的装配；
- 将领域结果转换为 HTTP/CLI 输出。

L3 不负责：

- ASR 数据预处理算法；
- LoRA 训练实现；
- RAG 检索和融合算法；
- CER/WER 计算；
- 业务状态流转规则。

### 2.4 代码 Monorepo 不等于单一 Python 环境

L1/L2/L3 默认共享根 `pyproject.toml` 和虚拟环境。只有存在冲突依赖的
运行时才拥有独立的：

- `pyproject.toml`；
- 受控依赖文件；
- `.venv`；
- Python 启动入口。

当前明确存在的冲突是：

```text
生产 Qwen ASR 推理：
qwen-asr==0.0.6
transformers==4.57.6

Qwen HF LoRA 训练：
Qwen/Qwen3-ASR-1.7B-hf
transformers>=5.13
peft
```

因此不能把生产推理和 HF LoRA 训练强行安装到同一个虚拟环境。

## 3. 目标目录结构

```text
backend/
├── pyproject.toml
│
├── packages/
│   ├── l1_foundation/
│   │   ├── __init__.py
│   │   ├── settings/
│   │   ├── task_runtime/
│   │   ├── pipeline/
│   │   └── infrastructure/
│   │
│   ├── l2_core/
│   │   ├── __init__.py
│   │   ├── access/
│   │   ├── application/
│   │   ├── audio_processing/
│   │   ├── rag/
│   │   ├── evaluation/
│   │   ├── asr_lab/
│   │   ├── auth/
│   │   ├── conversations/
│   │   ├── generation/
│   │   └── trainers/
│   │       └── qwen-asr-lora/
│   │           ├── pyproject.toml
│   │           ├── requirements.lock   # 在目标训练环境验证后生成
│   │           ├── .python-version
│   │           ├── qwen_asr_lora.py
│   │           ├── trainer.py
│   │           ├── inference.py
│   │           ├── contracts.py
│   │           ├── data.py
│   │           └── tests/
│   │
│   └── l3_app/
│       ├── production-api/
│       │   ├── main.py
│       │   └── app_factory.py
│       ├── evaluation-api/
│       │   └── main.py
│       ├── training-api/
│       │   └── main.py
│       └── 每个 API 内部拥有自己的 router、dependencies 和 routes
│
├── migrations/
│   ├── application.sql
│   └── evaluation.sql
│
├── scripts/
│   ├── install/
│   └── database/
│
└── tests/
    ├── architecture/
    ├── integration/
    └── e2e/
```

## 4. l1_foundation

L1 是基础能力包集合，不能包含录音、ASR、RAG、评测等业务语义。

### 4.1 foundation-kernel

包含：

- ID、时间、分页等基础值对象；
- 通用领域异常；
- command、event、result 等基础契约；
- clock、ID generator 等协议；
- 与业务无关的类型辅助。

不应演变为没有边界的 `common` 或 `utils` 目录。

### 4.2 foundation-pipeline

包含：

- Pipeline、Graph、Node、Stage 定义；
- Node 输入输出契约；
- artifact 引用；
- 执行上下文；
- 重试、取消、进度报告协议；
- 资源队列和调度抽象；
- pipeline 结构化日志上下文。

不包含：

- `QwenAsrTranscribeStage`；
- RAG graph 节点；
- 具体录音处理 stage；
- 具体数据库表 SQL。

### 4.3 foundation-persistence

包含：

- SQLAlchemy engine/session 创建；
- transaction 和 unit-of-work 基础能力；
- 数据库健康检查；
- 与业务无关的 repository 基础协议；
- JSON、UUID、时间等通用数据库适配。

各业务表的 SQL 和 repository 实现仍放在对应 L2 包。

### 4.4 foundation-storage

包含：

- ArtifactStore 协议；
- 本地文件存储适配器；
- 对象存储适配器；
- URI、checksum、atomic write 等基础能力。

### 4.5 foundation-observability

包含：

- 结构化日志初始化；
- request、run、pipeline、node 的 trace context；
- metric、trace、audit 的基础协议；
- 日志字段和脱敏规则。

## 5. l2_core

L2 是可复用的核心业务能力层，不包含 HTTP route 或进程启动代码。

### 5.1 recording

负责：

- Recording 聚合和状态；
- 录音元数据；
- 上传完成后的业务命令；
- 录音处理任务创建；
- recording repository。

### 5.2 audio-processing

负责：

- 音频规范化；
- VAD；
- 说话人分离；
- ASR 推理；
- 文本纠错；
- forced alignment；
- utterance 构建；
- embedding indexing；
- 生产录音处理 pipeline 定义。

生产 Qwen ASR 推理依赖属于该包或其独立推理 runtime，不能泄漏到 API 包。

### 5.3 rag

负责：

- route；
- 自适应 plan；
- scope retrieval；
- 混合检索；
- candidate fusion；
- chunk context 扩展；
- evidence 选择；
- answer generation；
- RAG 全链路日志。

### 5.4 dataset-registry

负责：

- source asset；
- 人工标注；
- 标注审核；
- 数据集；
- 数据集冻结版本；
- train/validation/test split；
- 数据版本 checksum。

它是训练与评测共同依赖的数据契约，避免训练包和评测包相互引用。

### 5.5 model-registry

负责：

- 基础模型；
- LoRA adapter；
- model version；
- candidate、validated、approved、retired 状态；
- runtime config；
- 模型产物 URI 和 checksum。

生产、评测、训练都通过该包读取模型身份和版本，不直接依赖彼此。

### 5.6 evaluation

负责：

- evaluation run；
- evaluation case；
- 模型对比；
- CER/WER 和 edit operation；
- micro/macro aggregation；
- case result；
- metric value；
- 评测结果查询。

### 5.7 asr-lab

负责：

- training run 领域状态；
- 训练预设；
- 训练数据 manifest 构造；
- 任务领取、取消和状态流转；
- 训练产物注册；
- 训练完成后自动创建 validation evaluation run；
- 与 trainer 子进程交互的协议。

`asr-lab` 是轻量编排包，不直接安装 Transformers、Torch、PEFT。

## 6. packages/l3_app

### 6.1 production-api

负责生产业务 HTTP API：

- 录音上传和查询；
- pipeline 创建、重试、取消和进度；
- transcript、summary 和 artifact 查询；
- RAG 对话与 evidence 返回。

建议路由前缀：

```text
/api/production/*
```

### 6.2 evaluation-api

负责评测 HTTP API：

- 数据集；
- source asset；
- 区间标注；
- 审核确认；
- 数据集版本冻结；
- 创建和查询评测任务；
- 总体指标和 case diff。

建议路由前缀：

```text
/api/evaluation/*
```

### 6.3 training-api

负责训练 HTTP API：

- 创建训练任务；
- 查询进度和日志；
- 取消任务；
- 训练预设；
- 候选模型；
- 模型审核、发布和废弃。

建议路由前缀：

```text
/api/training/*
```

训练 API 不在 HTTP 请求中加载模型或执行训练，只写入任务并返回 `run_id`。
其 lifespan 在后台线程中持有统一的 `AsrLabWorker`，串行消费评测和训练任务。

### 6.4 内嵌 Worker

`production-api` 的 lifespan 直接启动录音处理 PipelineCoordinator。

`training-api` 的 lifespan 直接启动 `AsrLabWorker`，负责：

- 轮询 evaluation run 和 training run；
- 单 GPU 资源仲裁；
- 串行执行任务；
- 创建 trainer/inference 子进程；
- 采集 stdout、stderr、退出码和结果 manifest；
- 更新任务状态；
- 处理取消、进程关闭、任务回队和异常恢复。

第一阶段调度优先级：

```text
线上高优先级推理 > 手工触发评测 > LoRA 训练
```

具体优先级可配置，但同一时间只允许一个高显存任务占用 GPU。

## 7. API、Worker 与 Trainer 的关系

```text
Frontend
  ├── /api/production/* → production-api
  ├── /api/evaluation/* → evaluation-api
  └── /api/training/*   → training-api

production-api
  → recording / audio-processing / rag
  → lifespan: PipelineCoordinator

evaluation-api
  → dataset-registry / model-registry / evaluation
  → evaluation_runs

training-api
  → dataset-registry / model-registry / asr-lab
  → training_runs
  → lifespan: AsrLabWorker
  ├── evaluation run → inference subprocess
  └── training run   → Qwen HF trainer subprocess
```

三套 API 可以由同一个反向代理或 Next.js BFF 暴露为统一域名。拆分 API 是代码和部署边界，不要求第一阶段引入不同数据库或不同机器。

## 8. Qwen ASR Trainer 独立项目

### 8.1 目录

```text
backend/packages/l2_core/trainers/qwen-asr-lora/
├── pyproject.toml
├── requirements.lock
├── .python-version
├── .venv/
├── qwen_asr_lora.py
├── contracts.py
├── trainer.py
├── inference.py
├── data.py
├── typing_utils.py
└── tests/
```

`.venv` 不提交 Git。

### 8.2 pyproject.toml

示意配置：

```toml
[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[project]
name = "airecord-qwen-asr-trainer"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "accelerate",
  "datasets>=4,<5",
  "librosa",
  "peft",
  "soundfile",
  "torch",
  "transformers>=5.13,<6",
]

[project.scripts]
qwen-asr-lora = "qwen_asr_lora:main"

[project.optional-dependencies]
dev = [
  "pytest",
  "pyright",
  "ruff",
]

[tool.hatch.build.targets.wheel]
only-include = [
  "qwen_asr_lora.py",
  "contracts.py",
  "trainer.py",
  "inference.py",
  "data.py",
  "typing_utils.py",
]
```

实际版本需要在目标 CUDA、PyTorch 和操作系统环境中验证后写入 `requirements.lock`，不能只依赖无上限的浮动版本。锁定文件与目标运行平台相关；如果 macOS 开发环境和 Linux CUDA 训练环境的依赖不同，应分别维护对应的受控依赖文件。

### 8.3 输入输出边界

Trainer 不直接连接业务数据库。

输入：

- `training-manifest.json`；
- train JSONL；
- validation JSONL；
- 裁剪后的音频文件；
- `Qwen/Qwen3-ASR-1.7B-hf` 本地 snapshot；
- LoRA preset。

输出：

- `adapter_config.json`；
- adapter safetensors；
- checkpoints；
- processor/config；
- `training-result.json`；
- 结构化 stdout 日志。

Manifest 示例：

```json
{
  "run_id": "uuid",
  "base_model": "/absolute/model-cache/.../Qwen3-ASR-1.7B-hf/snapshot",
  "train_file": "/absolute/path/train.jsonl",
  "validation_file": "/absolute/path/validation.jsonl",
  "output_dir": "/absolute/path/output",
  "preset": {
    "rank": 16,
    "alpha": 32,
    "dropout": 0.05,
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 0.0002,
    "epochs": 3
  }
}
```

启动方式：

```bash
backend/packages/l2_core/trainers/qwen-asr-lora/.venv/bin/python \
  -m qwen_asr_lora train \
  --manifest /absolute/path/training-manifest.json
```

### 8.4 独立安装

仓库安装脚本使用 Python 标准库 `venv` 和项目现有的 `pip`，为 trainer 创建独立环境：

```bash
python3.12 -m venv backend/packages/l2_core/trainers/qwen-asr-lora/.venv

backend/packages/l2_core/trainers/qwen-asr-lora/.venv/bin/python \
  -m pip install --upgrade pip

backend/packages/l2_core/trainers/qwen-asr-lora/.venv/bin/python \
  -e backend/packages/l2_core/trainers/qwen-asr-lora
```

首次安装从 `pyproject.toml` 解析依赖。目标训练环境通过 smoke test 后生成
`requirements.lock`；后续安装脚本优先安装 lock，再用 `--no-deps -e` 安装
Trainer 本身。

运行时配置：

```env
QWEN_ASR_TRAINER_PYTHON=backend/packages/l2_core/trainers/qwen-asr-lora/.venv/bin/python
QWEN_ASR_TRAINER_MODULE=qwen_asr_lora
QWEN_ASR_TRAINING_MODEL=Qwen/Qwen3-ASR-1.7B-hf
```

安装脚本必须幂等，并明确打印：

- 使用的 Python；
- trainer 环境路径；
- 是否使用受控依赖文件，以及安装状态；
- PyTorch/CUDA 可用状态；
- HF 模型是否已经下载。

### 8.5 Monorepo 本地包安装

不引入额外的 workspace 或依赖管理工具。L1、L2、L3 由根
`backend/pyproject.toml` 统一管理，生产环境只需安装一次：

```bash
backend/.venv/bin/python -m pip install -e 'backend[dev]'
```

L3 应用使用各自根目录下的 `main.py` 作为文件入口，共享同一个 backend 环境。

Trainer 不安装到 `backend/.venv`，也不被 production、evaluation 或 training API
直接 import。它始终使用自己的 `.venv`，由 training-api lifespan 中的
`AsrLabWorker` 通过 subprocess 调用。

`requirements.lock` 在已验证的目标环境中更新。第一阶段可以使用现有 `pip` 生成环境快照，不额外引入锁定工具：

```bash
backend/packages/l2_core/trainers/qwen-asr-lora/.venv/bin/python \
  -m pip freeze \
  --exclude-editable \
  > backend/packages/l2_core/trainers/qwen-asr-lora/requirements.lock
```

依赖升级必须显式执行并经过训练 smoke test，不在普通安装过程中自动刷新锁定文件。

## 9. Trainer 的代码归属

迁移前文件：

```text
backend/scripts/train_qwen_asr_lora.py
```

曾包含完整训练业务实现，不应继续位于 `scripts`，当前已经删除。

迁移后：

```text
backend/packages/l2_core/trainers/qwen-asr-lora/
```

承接：

- 模型加载；
- Processor；
- 训练数据编码；
- data collator；
- PEFT LoRA 配置；
- Trainer；
- checkpoint；
- adapter 保存；
- result manifest。

L3 只保留 API 进程入口；worker 生命周期由对应 API 持有。

`backend/scripts` 后续只保留：

- 环境安装；
- 数据库初始化；
- 本地开发辅助；
- CI/部署辅助。

不再存放生产 API、Worker 或模型训练的核心实现。

## 10. L2 内部依赖关系

允许的主要依赖：

```text
recording            → foundation-*
audio-processing     → recording, foundation-pipeline, foundation-storage
rag                  → recording, foundation-observability
dataset-registry     → recording, foundation-persistence, foundation-storage
model-registry       → foundation-persistence, foundation-storage
evaluation           → dataset-registry, model-registry
asr-lab              → dataset-registry, model-registry, evaluation
qwen-asr-lora        → 独立训练环境，不反向被 API import
```

Trainer 与 ASR Lab 通过文件和进程协议交互，不通过跨虚拟环境 Python import 交互。

## 11. 数据库边界

第一阶段继续使用同一个 PostgreSQL database，但定义表归属：

```text
recording：
  recordings、pipeline_runs、pipeline_nodes、artifacts

rag：
  chats、messages、evidence、embedding/index tables

dataset-registry：
  evaluation_datasets、evaluation_source_assets、
  evaluation_annotations、evaluation_dataset_versions、evaluation_cases

model-registry：
  model_versions

evaluation：
  evaluation_runs、evaluation_run_models、
  evaluation_case_results、evaluation_metric_values

asr-lab：
  training_runs
```

共享数据库不代表任意包都可以直接操作任意表。跨包访问通过公开 repository/service 契约完成。

## 12. 测试与架构约束

每个包拥有自己的 unit tests，仓库级测试负责 integration、E2E 和架构规则。

建议增加自动化架构测试：

- L1 不得 import `audio_processing`、`rag` 等 L2 包；
- L2 不得 import `fastapi` route 或 L3 包；
- `production-api` 不得依赖 trainer；
- `training-api` 不得依赖 Torch、Transformers、PEFT；
- `evaluation` 与 `asr-lab` 不得循环依赖；
- Trainer 不得 import Web API 或业务数据库 adapter。

## 13. 迁移映射

| 当前目录 | 目标目录 |
|---|---|
| `backend/src/pipeline` | `backend/packages/l1_foundation/pipeline` |
| `backend/src/infrastructure` | `backend/packages/l1_foundation/infrastructure` |
| `backend/src/application` | `backend/packages/l2_core/application` |
| `backend/src/audio_processing` | `backend/packages/l2_core/audio_processing` |
| `backend/src/rag` | `backend/packages/l2_core/rag` |
| `backend/src/evaluation` | `backend/packages/l2_core/evaluation` |
| `backend/src/asr_lab` | `backend/packages/l2_core/asr_lab` |
| `backend/scripts/train_qwen_asr_lora.py` | `backend/packages/l2_core/trainers/qwen-asr-lora` |
| `backend/src/api` 生产路由 | `backend/packages/l3_app/production-api` |
| `backend/src/api/routes/asr_lab.py` 评测路由 | `backend/packages/l3_app/evaluation-api` |
| `backend/src/api/routes/asr_lab.py` 训练路由 | `backend/packages/l3_app/training-api` |
| 当前生产 pipeline worker | `backend/packages/l3_app/production-api` lifespan |
| `backend/src/asr_lab/worker.py` | `backend/packages/l3_app/training-api` lifespan |

## 14. 分阶段迁移

### Phase 0：固定边界（已完成）

- 本文档评审通过；
- 建立包命名规范；
- 建立 import 方向规则；
- 明确数据库表归属；
- 暂不移动实现。

### Phase 1：独立 HF Trainer（代码迁移已完成）

- 已创建 `packages/l2_core/trainers/qwen-asr-lora`；
- 已创建独立 `pyproject.toml`；`.venv` 和 `requirements.lock` 在首次安装、验证目标训练环境后生成；
- 使用 `Qwen/Qwen3-ASR-1.7B-hf`；
- 将现有训练脚本逻辑迁入 trainer；
- 安装脚本增加 trainer 环境安装；
- ASR Lab Worker 改为通过 manifest 和 subprocess 调用；
- 删除 `backend/scripts/train_qwen_asr_lora.py`。

这一阶段优先执行，因为它解决当前实际存在的 Transformers 版本冲突。

### Phase 2：拆分三套 API（入口已完成）

- 拆出 production、evaluation、training 三个 composition root；
- 保持原有外部 API 兼容或增加反向代理；
- 共享鉴权协议和 request context；
- 确认三套 API 均不加载 GPU 模型。

### Phase 3：抽取 L1（物理迁移已完成）

- 先迁移纯协议和纯模型；
- 再迁移 pipeline kernel；
- 迁移 persistence、storage、observability；
- 保持业务行为和数据库结构不变。

### Phase 4：迁移 L2（物理迁移和 namespace 收口已完成）

- recording；
- audio-processing；
- rag；
- dataset-registry；
- model-registry；
- evaluation；
- asr-lab。

后续逐包把内部调用切换到 `airecord_*` namespace，并补齐更严格的跨层
import boundary test。

### Phase 5：Worker 与部署收口（入口已完成，生产部署验证待完成）

- production-api lifespan 内嵌 PipelineCoordinator；
- training-api lifespan 内嵌统一 ASR Lab worker；
- 训练和评测任务增加资源锁；
- 完善结构化日志、取消、恢复和超时；
- 生成开发与生产部署命令。

## 15. 最终决策摘要

1. 使用单仓库、多 Python project 的分层 Monorepo；
2. `packages/l1_foundation`、`packages/l2_core` 是显式 Python namespace，`packages/l3_app` 是应用入口目录；
3. L3 API 拆为 production、evaluation、training 三个独立入口；
4. 生产 Worker 与 ASR 计算 Worker 分开；
5. ASR 评测和训练可由一个计算 Worker 串行调度，同一时间只占用一个 GPU；
6. Qwen HF Trainer 属于 L2 核心能力，不放在 `backend/scripts`；
7. Trainer 拥有独立 `pyproject.toml` 和 `.venv`，目标环境验证后提交对应的 `requirements.lock`；
8. Trainer 不直接访问业务数据库，通过 manifest、文件产物和 subprocess 协议与 Worker 交互；
9. 第一阶段只使用现有的 `venv + pip`，共享包通过 editable local package 安装，Trainer 使用独立环境；
10. 重构按 Trainer、API、L1、L2、Worker 的顺序渐进完成，不进行一次性全目录迁移。
