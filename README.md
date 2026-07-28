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

Install audio dependencies:

```bash
scripts/install_audio_dependencies.sh
```

The backend can also run this automatically on startup. It is enabled by default in `.env`:

```bash
AUDIO_DEPS_AUTO_INSTALL=true
```

Set it to `false` if you want startup to fail fast instead of installing Python packages.

## Run

Start the web app:

```bash
npm run dev
```

Useful pages:

- `http://localhost:3000/recordings`
- `http://localhost:3000/speaker-profiles`

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
