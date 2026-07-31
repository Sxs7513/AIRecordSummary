# Python Backend

## Prerequisites

- Python 3.14.4
- PostgreSQL
- Redis 7.4+
- Kafka 3.9+（KRaft）

The backend reads the existing repository-root `.env` file directly.
`backend/packages/l1_foundation/settings` converts those values
into typed Python settings; it is not a second configuration file. Do not
commit real credentials.

## Install

```bash
cd backend
python3.14 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
cd ..
scripts/install_audio_dependencies.sh
scripts/db/init-python-backend.sh
```

The audio installer installs the aggregate backend project and creates the
isolated Qwen HF Trainer environment under:

```text
backend/packages/l2_core/trainers/qwen-asr-lora/.venv
```

## Run

From the repository root:

```bash
npm run infra:up
npm run dev:production-api
npm run dev:compute-worker
npm run dev:generation-worker
npm run dev:processing-worker
npm run dev:observability-api
npm run dev:observability-worker
npm run dev:evaluation-api
npm run dev:training-api
```

After an intentional Redis reset, stop command processing temporarily and run `npm run infra:rebuild-redis` to replay the compacted Processing, Compute, and Generation state topics.

The Production API, Compute Worker, Observability API, Evaluation API, and Training API listen on ports 8000, 8010, 8003, 8001, and 8002. `npm run dev:python-web` is a
backward-compatible alias for `npm run dev:production-api`.

Start these services in separate terminals. `production-api` publishes Generation
and Processing commands; `processing-worker` advances the recording DAG and submits
atomic Compute commands. The corresponding workers consume them. RAG Span and
model-usage events are projected by `observability-worker` into the PostgreSQL
query model used by `observability-api`. Relevant settings:

```text
OBSERVABILITY_ENABLED=true
KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:9092
PROCESSING_CONSUMER_MAX_POLL_INTERVAL_MS=7200000
REDIS_URL=redis://127.0.0.1:6379/0
NEXT_PUBLIC_OBSERVABILITY_API_ORIGIN=http://localhost:8003
```

The Observability API is query-only. Its browser endpoints use the normal
session cookie and always restrict results to the user's current workspace.

The Uvicorn access logs are disabled so terminal output stays focused on
pipeline stages and model progress.

`production-api` does not run an in-process Pipeline coordinator. Processing and Compute work are both Kafka-driven, with Redis providing live state and resumable streams. `training-api` starts the single-GPU ASR evaluation
and LoRA training worker in a background thread. Both stop before their API
releases database resources; interrupted ASR Lab runs are returned to the
queue and can be resumed by the next process.

Run each API as a single Uvicorn process. Starting multiple `training-api`
workers would create multiple ASR Lab consumers competing for the same GPU.

Run `scripts/install_audio_dependencies.sh` to install production audio-model
dependencies into `backend/.venv` and HF training dependencies into the
Trainer's isolated `.venv`. The legacy root `.venv-audio` is no longer used.

## LLM providers

All text-generation calls go through `l1_foundation.llm`. The default provider
and each use can be selected independently:

```text
LLM_DEFAULT_PROVIDER=gemini
```

Supported values are `gemini`, `local`, and `zhipu`. A use can override the default with
`LLM_CORRECTION_PROVIDER`, `TOPIC_DETECTION_PROVIDER`,
`RECORDING_SUMMARY_PROVIDER`, or `RAG_ANSWER_PROVIDER`. The local provider uses
the single Qwen2.5 7B GGUF model configured by `LOCAL_LLM_MODEL_REPO` and
`LOCAL_LLM_MODEL_FILE`.

Gemini is the project default. Put its credential in the ignored `.env.local`
file:

```text
GEMINI_API_KEY=your-api-key
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
```

To use Zhipu globally instead, put the credential and provider selection in
`.env.local`:

```text
LLM_DEFAULT_PROVIDER=zhipu
ZHIPU_API_KEY=your-api-key
ZHIPU_MODEL=glm-4.5-flash
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
```

Both providers implement streaming, non-streaming, JSON-object, and
provider-adapted JSON-schema requests. Business output is always validated in
L2 after generation.

## Real-audio processing test

The default test suite never loads audio models. To exercise a short local recording end to end, start the infrastructure and workers, then upload the recording through the API or UI:

```bash
npm run infra:up
npm run dev:compute-worker
npm run dev:processing-worker
npm run dev:production-api
```

It exercises `normalize → diarize → Qwen ASR → utterances → correction → summary`. It requires the models referenced by the root `.env` to be available under the repository `model-cache` directory.

旧的数据库 Pipeline E2E 已随运行时表一起删除。新的故障与端到端测试必须启动真实 Kafka/Redis，并验证 Consumer 重放、幂等、retry/DLQ 和 Redis 状态重建。Pipeline Stage 不再按资源队列整体调度；原子 CPU/GPU/在线模型任务由 Compute operation 声明自己的 lane，业务 DAG 只由 `processing-worker` 推进。

## Recording API

All endpoints use the configured `API_PREFIX` (by default `/api`). The
upload endpoint accepts one `multipart/form-data` `audio` file and optional
`title` and `location` fields. On success it writes the file to local storage,
creates `recordings`, and publishes a `recording_processing` command to Kafka.

```text
POST /api/recordings
GET  /api/recordings?status=&page=&page_size=
GET  /api/recordings/{recording_id}
GET  /api/recordings/pipeline-runs/{pipeline_run_id}
POST /api/recordings/{recording_id}/retry
```

`GET /recordings/{recording_id}` returns the materialized transcription,
diarization, utterances, summary, and pipeline-run history. The pipeline-run
endpoint returns each stage's live state, attempt count, and any error.
`retry` is allowed only after the recording reaches `failed`; it creates a new
pipeline run and keeps the failed run intact for diagnosis.

## Checks

```bash
.venv/bin/ruff check packages packages/l2_core/trainers packages/l3_app tests scripts
.venv/bin/pyright
.venv/bin/pytest
```

`scripts/db/init-python-backend.sh` creates the configured database when needed, drops and recreates its `public` schema, then executes `sql/base.sql`. It is intentionally destructive: this refactor does not retain historical data or obsolete runtime tables.
