from __future__ import annotations

from typing import Literal, cast
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from api.dependencies import AuthServiceDependency, CurrentUserDependency
from auth.contracts import CurrentUser
from auth.service import AuthenticationError

router = APIRouter()


class WorkspaceResponse(BaseModel):
    id: UUID
    name: str
    role: Literal["owner", "admin", "member"]


class CurrentUserResponse(BaseModel):
    id: UUID
    email: str
    display_name: str
    current_workspace_id: UUID
    memberships: list[WorkspaceResponse]


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=1024)


class LoginResponse(BaseModel):
    user: CurrentUserResponse


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, response: Response, request: Request, service: AuthServiceDependency) -> LoginResponse:
    try:
        user, token = service.login(str(payload.email), payload.password)
    except AuthenticationError as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password") from error
    settings = request.app.state.settings
    response.set_cookie(
        key=settings.session_cookie_name,
        value=token,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite="lax",
        max_age=settings.session_ttl_days * 24 * 60 * 60,
        path="/",
    )
    return LoginResponse(user=_user_response(user))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response, request: Request, service: AuthServiceDependency) -> None:
    settings = request.app.state.settings
    service.logout(request.cookies.get(settings.session_cookie_name))
    response.delete_cookie(settings.session_cookie_name, path="/")


@router.get("/me", response_model=CurrentUserResponse)
def me(user: CurrentUserDependency) -> CurrentUserResponse:
    return _user_response(user)


def _user_response(user: CurrentUser) -> CurrentUserResponse:
    return CurrentUserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        current_workspace_id=user.current_workspace_id,
        memberships=[
            WorkspaceResponse(id=item.workspace_id, name=item.workspace_name, role=cast(Literal["owner", "admin", "member"], item.role))
            for item in user.memberships
        ],
    )
