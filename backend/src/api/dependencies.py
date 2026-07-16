from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy import Engine

from access.generations import GenerationAccessService
from application.recordings import RecordingService
from audio_processing.stages.summary.regeneration import RecordingSummaryRegenerationService
from auth.contracts import CurrentUser
from auth.service import AuthenticationError, AuthService
from conversations.service import ConversationService
from infrastructure.storage.local import LocalStorage


def get_recording_service(request: Request) -> RecordingService:
    """Build the recording use case from infrastructure owned by this app instance."""
    return RecordingService(
        request.app.state.database_engine,
        request.app.state.storage,
        request.app.state.recording_processing_definition,
    )


RecordingServiceDependency = Annotated[RecordingService, Depends(get_recording_service)]


def get_recording_summary_regeneration_service(request: Request) -> RecordingSummaryRegenerationService:
    return request.app.state.recording_summary_regeneration_service


RecordingSummaryRegenerationServiceDependency = Annotated[RecordingSummaryRegenerationService, Depends(get_recording_summary_regeneration_service)]


def get_auth_service(request: Request) -> AuthService:
    return AuthService(request.app.state.database_engine, request.app.state.settings.session_ttl_days)


AuthServiceDependency = Annotated[AuthService, Depends(get_auth_service)]


def require_current_user(request: Request, service: AuthServiceDependency) -> CurrentUser:
    try:
        return service.require_session(request.cookies.get(request.app.state.settings.session_cookie_name))
    except AuthenticationError as error:
        from fastapi import HTTPException, status

        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required") from error


CurrentUserDependency = Annotated[CurrentUser, Depends(require_current_user)]


def get_database_engine(request: Request) -> Engine:
    return request.app.state.database_engine


def get_storage(request: Request) -> LocalStorage:
    return request.app.state.storage


DatabaseEngineDependency = Annotated[Engine, Depends(get_database_engine)]
StorageDependency = Annotated[LocalStorage, Depends(get_storage)]


def get_generation_access_service(request: Request) -> GenerationAccessService:
    return GenerationAccessService(request.app.state.database_engine)


GenerationAccessServiceDependency = Annotated[GenerationAccessService, Depends(get_generation_access_service)]


def get_conversation_service(request: Request) -> ConversationService:
    return ConversationService(request.app.state.database_engine, request.app.state.generation_service)


ConversationServiceDependency = Annotated[ConversationService, Depends(get_conversation_service)]
