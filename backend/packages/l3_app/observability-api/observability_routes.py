from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from service import ObservabilityService

from l2_core.auth.contracts import CurrentUser
from l2_core.auth.service import AuthenticationError, AuthService

query_router = APIRouter()


def get_service(request: Request) -> ObservabilityService:
    return request.app.state.observability_service


def require_current_user(request: Request) -> CurrentUser:
    settings = request.app.state.settings
    try:
        return AuthService(request.app.state.database_engine, settings.session_ttl_days).require_session(
            request.cookies.get(settings.session_cookie_name)
        )
    except AuthenticationError as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required") from error


ServiceDependency = Annotated[ObservabilityService, Depends(get_service)]
CurrentUserDependency = Annotated[CurrentUser, Depends(require_current_user)]


@query_router.get("/overview")
def overview(
    service: ServiceDependency,
    user: CurrentUserDependency,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict[str, object]:
    try:
        return service.overview(user.current_workspace_id, start, end)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@query_router.get("/runs")
def list_runs(
    service: ServiceDependency,
    user: CurrentUserDependency,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[dict[str, object]]:
    try:
        return service.list_runs(user.current_workspace_id, user.id, start, end, limit, offset)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@query_router.get("/runs/{run_id}")
def run_detail(run_id: UUID, service: ServiceDependency, user: CurrentUserDependency) -> dict[str, object]:
    detail = service.run_detail(user.current_workspace_id, run_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="RAG observability run not found")
    return detail


@query_router.get("/runs/{run_id}/conversation")
def run_conversation(run_id: UUID, service: ServiceDependency, user: CurrentUserDependency) -> dict[str, object]:
    conversation = service.run_conversation(user.current_workspace_id, run_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation snapshot not found")
    return conversation
