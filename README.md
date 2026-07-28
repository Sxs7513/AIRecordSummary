# AIRecordSummary

Phase 1 follows `docs/technical-architecture.md` and `docs/phase-1-implementation-plan.md`:

- Frontend and API: Next.js App Router + TypeScript
- Database: PostgreSQL, schema in `sql/base.sql`
- Offline processing: embedded PostgreSQL-backed worker started with the Next.js backend
- Audio pipeline adapters: Whisper, pyannote.audio, SpeechBrain

## Setup

Install Node dependencies:

```bash
npm install
```

Configure PostgreSQL in `.env`:

```bash
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=postgres
DB_NAME=ai_record_summary
DB_ADMIN_DATABASE=postgres
```

Speaker diarization uses the gated pyannote model. Set this in `.env` before retrying diarization jobs:

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

backend/L2-Core/trainers/qwen-asr-lora/.venv
  Qwen3-ASR-1.7B-hf, Transformers 5.13+ and PEFT LoRA training/evaluation
```

The backend can also run this automatically on startup. It is enabled by default in `.env`:

```bash
AUDIO_DEPS_AUTO_INSTALL=true
```

Set it to `false` if you want startup to fail fast instead of installing Python packages.

## Run

Start the frontend and three Python API entry points:

```bash
npm run dev
npm run dev:production-api
npm run dev:evaluation-api
npm run dev:training-api
```

The APIs listen on ports 8000, 8001, and 8002 respectively.
`npm run dev:python-web` remains as an alias for the production API.

Run the isolated ASR evaluation and LoRA training worker in another terminal:

```bash
npm run worker:asr-lab
```

When `PIPELINE_EMBEDDED_WORKERS_ENABLED=false`, also start the standalone
production pipeline worker:

```bash
npm run worker:production
```

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

Uploaded files are stored under `uploads/` in development. Metadata, transcriptions, speaker diarization segments, target-speaker matches, pipeline runs, and stage runs are persisted in PostgreSQL.

The Python backend owns database initialization and background pipeline execution. Recording processing progress is tracked by `pipeline_runs` and `stage_runs`.
