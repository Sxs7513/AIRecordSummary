from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy import Engine

from l1_foundation.files import FileStore
from l1_foundation.streaming import RedisStreamStore
from l2_core.access.generations import GenerationAccessService
from l2_core.application.recordings import RecordingService
from l2_core.audio_processing.stages.summary.regeneration import RecordingSummaryRegenerationService
from l2_core.auth.authorization import AuthorizationService
from l2_core.auth.contracts import CurrentUser
from l2_core.auth.service import AuthenticationError, AuthService
from l2_core.conversations.service import ConversationService
from l2_core.generation.service import GenerationService
from l2_core.rag.queue import GenerationCommandPublisher


def get_recording_service(request: Request) -> RecordingService:
    """Build the recording use case from infrastructure owned by this app instance."""
    return RecordingService(
        request.app.state.database_engine,
        request.app.state.storage,
        request.app.state.recording_processing_definition,
        request.app.state.processing_command_publisher,
        request.app.state.sync_redis_store,
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
        user = service.require_session(request.cookies.get(request.app.state.settings.session_cookie_name))
        return AuthorizationService().require_authenticated(user)
    except AuthenticationError as error:
        from fastapi import HTTPException, status

        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required") from error


CurrentUserDependency = Annotated[CurrentUser, Depends(require_current_user)]


def get_database_engine(request: Request) -> Engine:
    return request.app.state.database_engine


def get_storage(request: Request) -> FileStore:
    return request.app.state.storage


DatabaseEngineDependency = Annotated[Engine, Depends(get_database_engine)]
StorageDependency = Annotated[FileStore, Depends(get_storage)]


def get_generation_access_service(request: Request) -> GenerationAccessService:
    return GenerationAccessService(request.app.state.database_engine, request.app.state.generation_redis_runtime)


GenerationAccessServiceDependency = Annotated[GenerationAccessService, Depends(get_generation_access_service)]


def get_conversation_service(request: Request) -> ConversationService:
    return ConversationService(
        request.app.state.database_engine,
        request.app.state.generation_service,
        request.app.state.conversation_history_store,
        request.app.state.generation_command_publisher,
    )


ConversationServiceDependency = Annotated[ConversationService, Depends(get_conversation_service)]


def get_generation_service(request: Request) -> GenerationService:
    return request.app.state.generation_service


def get_generation_redis_event_store(request: Request) -> RedisStreamStore:
    return request.app.state.async_redis_store


def get_generation_command_publisher(request: Request) -> GenerationCommandPublisher:
    return request.app.state.generation_command_publisher


GenerationServiceDependency = Annotated[GenerationService, Depends(get_generation_service)]
GenerationRedisEventStoreDependency = Annotated[RedisStreamStore, Depends(get_generation_redis_event_store)]
GenerationCommandPublisherDependency = Annotated[GenerationCommandPublisher, Depends(get_generation_command_publisher)]
