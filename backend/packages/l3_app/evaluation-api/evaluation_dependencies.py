from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from l1_foundation.files import FileStore
from l2_core.asr_lab.service import AsrLabService
from l2_core.audio_processing.stages.transcribe_qwen_asr.context import build_qwen_asr_context
from l2_core.auth.authorization import AuthorizationService
from l2_core.auth.contracts import CurrentUser
from l2_core.auth.service import AuthenticationError, AuthService
from l2_core.rag_adjudication_evaluation.service import RagAdjudicationEvaluationService
from l2_core.rag_evaluation.service import RagEvaluationService


def get_auth_service(request: Request) -> AuthService:
    return AuthService(request.app.state.database_engine, request.app.state.settings.session_ttl_days)


AuthServiceDependency = Annotated[AuthService, Depends(get_auth_service)]


def require_current_user(request: Request, service: AuthServiceDependency) -> CurrentUser:
    try:
        user = service.require_session(request.cookies.get(request.app.state.settings.session_cookie_name))
        return AuthorizationService().require_authenticated(user)
    except AuthenticationError as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required") from error


CurrentUserDependency = Annotated[CurrentUser, Depends(require_current_user)]


def get_storage(request: Request) -> FileStore:
    return request.app.state.storage


StorageDependency = Annotated[FileStore, Depends(get_storage)]


def get_asr_lab_service(request: Request) -> AsrLabService:
    settings = request.app.state.settings
    return AsrLabService(
        request.app.state.database_engine,
        request.app.state.storage,
        settings.resolved_asr_lab_project_dataset_root,
        training_workspace_root=settings.resolved_local_storage_root,
        evaluation_context=build_qwen_asr_context(
            settings.resolved_qwen_asr_context_config,
            settings.qwen_asr_max_context_items,
            settings.qwen_asr_context,
        ),
    )


AsrLabServiceDependency = Annotated[AsrLabService, Depends(get_asr_lab_service)]


def get_rag_evaluation_service(request: Request) -> RagEvaluationService:
    return RagEvaluationService(request.app.state.database_engine, request.app.state.settings)


RagEvaluationServiceDependency = Annotated[RagEvaluationService, Depends(get_rag_evaluation_service)]


def get_rag_adjudication_evaluation_service(request: Request) -> RagAdjudicationEvaluationService:
    return RagAdjudicationEvaluationService(request.app.state.database_engine, request.app.state.settings)


RagAdjudicationEvaluationServiceDependency = Annotated[
    RagAdjudicationEvaluationService,
    Depends(get_rag_adjudication_evaluation_service),
]
