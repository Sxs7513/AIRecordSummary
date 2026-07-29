# Python Backend

## Prerequisites

- Python 3.14.4
- PostgreSQL

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
npm run dev:production-api
npm run dev:evaluation-api
npm run dev:training-api
```

The APIs listen on ports 8000, 8001, and 8002. `npm run dev:python-web` is a
backward-compatible alias for `npm run dev:production-api`.

The Uvicorn access logs are disabled so terminal output stays focused on
pipeline stages and model progress.

`production-api` starts the recording pipeline coordinator and GPU scheduler
as part of its lifespan. `training-api` starts the single-GPU ASR evaluation
and LoRA training worker in a background thread. Both stop before their API
releases database resources; interrupted ASR Lab runs are returned to the
queue and can be resumed by the next process.

Run each API as a single Uvicorn process. Starting multiple `training-api`
workers would create multiple ASR Lab consumers competing for the same GPU.

Run `scripts/install_audio_dependencies.sh` to install production audio-model
dependencies into `backend/.venv` and HF training dependencies into the
Trainer's isolated `.venv`. The legacy root `.venv-audio` is no longer used.

## Real-audio pipeline test

The default test suite never loads audio models. To run the opt-in end-to-end test on a short local recording, use the audio environment and provide an absolute audio path:

```bash
cd backend
RUN_AUDIO_E2E=1 AUDIO_E2E_FILE=/absolute/path/to/short.wav \
  .venv/bin/python -m pytest tests/integration/test_recording_processing_e2e.py
```

It exercises `normalize → diarize → Qwen ASR → utterances → correction → summary`. It requires the models referenced by the root `.env` to be available under the repository `model-cache` directory.

The database-backed E2E additionally verifies the declared `recording_processing` graph, queue workers, artifact bindings and business-table projections. It creates a uniquely named temporary database through the PostgreSQL admin connection configured by the root `.env`, resets its `public` tables, applies `sql/base.sql`, then drops that database after the test. The configured database user must therefore have `CREATE DATABASE` and `CREATE EXTENSION vector` privileges:

```bash
cd backend
RUN_PIPELINE_E2E=1 AUDIO_E2E_FILE=/absolute/path/to/short.wav \
  .venv/bin/python -m pytest tests/integration/test_recording_processing_e2e.py
```

For the repository fixture `tests/files/test1.mp3`, the equivalent shortcut is:

```bash
backend/scripts/test_recording_processing_pipeline_e2e.sh
```

For independent workers, set `PIPELINE_WORKER_QUEUE` in the root `.env` to select the queue:

```text
cpu        # normalize_audio, build_utterances, build_search_chunks
gpu_high   # diarize_pyannote, transcribe_qwen_asr
gpu_normal # correct_text, embedding_indexing, generate_summary
```

## Recording API

All endpoints use the configured `API_PREFIX` (by default `/api`). The
upload endpoint accepts one `multipart/form-data` `audio` file and optional
`title` and `location` fields. On success it writes the file to local storage,
creates `recordings`, and immediately creates a `recording_processing` run.

```text
POST /api/recordings
GET  /api/recordings?status=&page=&page_size=
GET  /api/recordings/{recording_id}
GET  /api/recordings/pipeline-runs/{pipeline_run_id}
POST /api/recordings/{recording_id}/retry
```

`GET /recordings/{recording_id}` returns the materialized transcription,
diarization, utterances, summary, and pipeline-run history. The pipeline-run
endpoint returns each stage's live state, queue, attempt count, and any error.
`retry` is allowed only after the recording reaches `failed`; it creates a new
pipeline run and keeps the failed run intact for diagnosis.

## Checks

```bash
.venv/bin/ruff check packages packages/l2_core/trainers packages/l3_app tests scripts
.venv/bin/pyright
.venv/bin/pytest
```

`scripts/db/init-python-backend.sh` creates the configured database when needed, then executes the idempotent `sql/base.sql`. Tables and indexes use `if not exists`, so missing objects are added while existing tables and their data are preserved.
