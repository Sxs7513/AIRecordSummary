from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from infrastructure.db.health import database_is_healthy
from settings import Settings, get_settings

router = APIRouter()
SettingsDependency = Annotated[Settings, Depends(get_settings)]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["ok", "unavailable"]


@router.get("/health", response_model=HealthResponse)
def health_check(settings: SettingsDependency) -> HealthResponse:
    """Report API liveness and database readiness."""
    if database_is_healthy(settings):
        return HealthResponse(status="ok", database="ok")
    return HealthResponse(status="degraded", database="unavailable")
