# AIRecordSummary

Phase 1 follows `docs/technical-architecture.md` and `docs/phase-1-implementation-plan.md`:

- Frontend and API: Next.js App Router + TypeScript
- Database: PostgreSQL, schema in `sql/base.sql`
- Async messaging: Kafka
- Live state and resumable streams: Redis
- Offline orchestration: Kafka-driven Processing Worker with Redis live state
- Atomic model/CPU/GPU compute: Kafka lane consumers with Redis result/stream projections
- Audio pipeline adapters: Whisper, pyannote.audio, SpeechBrain

## Setup

Install Node dependencies:

```bash
npm install
```

Project-wide development configuration is versioned in `.env`. Put
machine-local credentials and tokens in the ignored `.env.local`; its values
override `.env`.

Configure PostgreSQL in `.env` or `.env.local`:

```bash
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=postgres
DB_NAME=ai_record_summary
DB_ADMIN_DATABASE=postgres
```

Speaker diarization uses the gated pyannote model. Set this in `.env.local`
before retrying diarization jobs:

```bash
PYANNOTE_AUTH_TOKEN=your_huggingface_token
```

Reset and initialize the development database manually when needed. This deletes all existing database records:

```bash
npm run db:init
```

The Python web service does not initialize the database on startup; run this command after changing `sql/base.sql`.

Initialize the ASR Lab tables in the same database after the base schema:

```bash
npm run db:init:evaluation
```

Install audio dependencies:

```bash
scripts/install_audio_dependencies.sh
```

The installer creates two isolated Python environments:

```text
backend/.venv
  Production APIs, pipeline workers, qwen-asr inference and shared packages

backend/packages/l2_core/trainers/qwen-asr-lora/.venv
  Qwen3-ASR-1.7B-hf, Transformers 5.13+ and PEFT LoRA training/evaluation
```

The backend can also run this automatically on startup. It is enabled by default in `.env`:

```bash
AUDIO_DEPS_AUTO_INSTALL=true
```

Set it to `false` if you want startup to fail fast instead of installing Python packages.

## Run

Start Kafka and Redis first, then the frontend and Python processes:

```bash
npm run infra:up
npm run dev
npm run dev:production-api
npm run dev:compute-worker
npm run dev:generation-worker
npm run dev:processing-worker
npm run dev:observability-api
npm run dev:observability-worker
npm run dev:evaluation-api
npm run dev:rag-evaluation-worker
npm run dev:training-api
```

If Redis is cleared, stop command processing temporarily and replay the compacted Kafka state topics with `npm run infra:rebuild-redis` before resuming traffic.

The Production API, Compute Worker, Evaluation API, and Training API listen on ports 8000, 8010, 8001, and 8002 respectively.
`npm run dev:python-web` remains as an alias for the production API.

Generation, recording Processing, and atomic Compute work are submitted through Kafka. `generation-worker`, `processing-worker`, and `compute-worker` consume them independently; the Production API does not run an in-process task coordinator.
`training-api` likewise starts the single-GPU ASR evaluation/training worker,
so no separate worker process is required.

Useful pages:

- `http://localhost:3000/recordings`
- `http://localhost:3000/speaker-profiles`
- `http://localhost:3000/asr-lab`

## Phase 1 Routes

- `GET /api/recordings`
- `POST /api/recordings`
- `GET /api/recordings/:id`
- `GET /api/speaker-profiles`
- `POST /api/speaker-profiles`
- `POST /api/speaker-profiles/:id/samples`

## Notes

Uploaded files are stored under `uploads/` in development. PostgreSQL stores business data and final query projections; Redis stores live task state and streams; Kafka stores commands and durable lifecycle events.

The Python backend owns database initialization. Recording processing commands and durable lifecycle events live in Kafka, live progress lives in Redis, and PostgreSQL stores only recording business results and terminal projections.
