from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from l1_foundation.infrastructure.db.session import create_database_engine
from l1_foundation.infrastructure.storage.local import LocalStorage
from l1_foundation.settings import get_settings
from l2_core.asr_lab.worker import AsrLabWorker
from router import training_api_router

logger = logging.getLogger("train")


def _configure_train_logger() -> None:
    """Expose training milestones without enabling noisy global request logs."""
    train_logger = logging.getLogger("train")
    train_logger.setLevel(logging.INFO)
    train_logger.propagate = False
    if train_logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    train_logger.addHandler(handler)


def _configure_evaluation_logger() -> None:
    """Expose evaluation milestones from the ASR Lab worker."""
    evaluation_logger = logging.getLogger("evaluation")
    evaluation_logger.setLevel(logging.INFO)
    evaluation_logger.propagate = False
    if evaluation_logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    evaluation_logger.addHandler(handler)


async def _run_asr_lab_worker(worker: AsrLabWorker) -> None:
    try:
        await asyncio.to_thread(worker.run_forever)
    except Exception:
        logger.exception("ASR Lab worker stopped unexpectedly")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    storage = LocalStorage(app.state.settings.resolved_local_storage_root)
    storage.initialize()
    app.state.storage = storage
    app.state.database_engine = create_database_engine(app.state.settings)
    worker = AsrLabWorker(app.state.database_engine, app.state.settings)
    worker_task = asyncio.create_task(_run_asr_lab_worker(worker), name="asr-lab-worker")
    app.state.asr_lab_worker = worker
    try:
        yield
    finally:
        worker.stop()
        await worker_task
        app.state.database_engine.dispose()


def create_app() -> FastAPI:
    _configure_train_logger()
    _configure_evaluation_logger()
    settings = get_settings()
    app = FastAPI(title=f"{settings.app_name} - Training", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Accept", "Authorization", "Content-Type"],
    )
    app.state.settings = settings
    app.include_router(training_api_router, prefix=settings.api_prefix)
    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8002, access_log=False)
