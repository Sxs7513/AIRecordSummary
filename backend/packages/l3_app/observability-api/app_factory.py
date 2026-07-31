from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from observability_routes import query_router
from repository import ObservabilityRepository
from service import ObservabilityService

from l1_foundation.infrastructure.db.session import create_database_engine
from l1_foundation.settings import Settings, get_settings


def _ready() -> dict[str, str]:
    return {"status": "ready"}


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        app.state.database_engine = create_database_engine(configured)
        repository = ObservabilityRepository(app.state.database_engine)
        repository.abandon_stale_records()
        app.state.observability_service = ObservabilityService(repository)
        try:
            yield
        finally:
            app.state.database_engine.dispose()

    app = FastAPI(title="AI Record Summary Observability API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["Accept", "Content-Type"],
    )
    app.state.settings = configured
    app.include_router(query_router, prefix="/api/observability", tags=["observability"])

    app.add_api_route("/readyz", _ready, methods=["GET"], tags=["health"])

    return app
