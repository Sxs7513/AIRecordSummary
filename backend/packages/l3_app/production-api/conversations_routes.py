from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from generations_routes import (
    _resume_after,  # pyright: ignore[reportPrivateUsage]
    _snapshot_event,  # pyright: ignore[reportPrivateUsage]
    _stream_events,  # pyright: ignore[reportPrivateUsage]
    encode_sse,
)
from pydantic import BaseModel, Field

from l1_foundation.streaming import RedisStreamStore
from l2_core.access.conversations import ConversationAccessDeniedError
from l2_core.conversations.contracts import Conversation, ConversationBusyError, ConversationMessage, ConversationMessagePage, ConversationNotFoundError
from l2_core.generation.contracts import GenerationEvent, GenerationSnapshot, TextBlock
from l2_core.generation.service import GenerationService
from l2_core.rag.adjudication.contracts import ClaimConfirmationDecision
from l2_core.rag.service import RagService
from production_dependencies import (
    ConversationServiceDependency,
    CurrentUserDependency,
    GenerationRedisEventStoreDependency,
    GenerationServiceDependency,
)

router = APIRouter()


class CreateConversationRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class SendConversationMessageRequest(BaseModel):
    content_blocks: list[TextBlock] = Field(min_length=1)
    client_message_id: UUID
    limit: int = Field(default=10, ge=1, le=20)
    force_correction: bool = False


class StartConversationTurnRequest(SendConversationMessageRequest):
    client_conversation_id: UUID


class ResumeConversationGenerationRequest(BaseModel):
    client_request_id: UUID
    mode: Literal["resume", "regenerate"] = "resume"


class ConversationTurnResponse(BaseModel):
    user_message: ConversationMessage
    assistant_message: ConversationMessage
    generation_run_id: UUID


class AdjudicationDecisionRequest(ClaimConfirmationDecision):
    pass


def _rag_service(request: Request) -> RagService:
    return request.app.state.rag_service


@router.get("", response_model=list[Conversation])
def list_conversations(service: ConversationServiceDependency, user: CurrentUserDependency) -> list[Conversation]:
    return service.list(user)


@router.post("", response_model=Conversation, status_code=status.HTTP_201_CREATED)
def create_conversation(payload: CreateConversationRequest, service: ConversationServiceDependency, user: CurrentUserDependency) -> Conversation:
    try:
        return service.create(user, payload.title)
    except PermissionError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Current workspace access denied") from error


@router.post("/turn/events")
async def start_conversation_turn(
    payload: StartConversationTurnRequest,
    request: Request,
    conversation_service: ConversationServiceDependency,
    generation_service: GenerationServiceDependency,
    generation_redis_event_store: GenerationRedisEventStoreDependency,
    user: CurrentUserDependency,
    last_event_id: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    """Create the first turn atomically and stream its generation on this response."""
    try:
        conversation, user_message, assistant_message, _history, _created = conversation_service.create_initial_turn(
            user,
            payload.client_conversation_id,
            payload.content_blocks,
            payload.client_message_id,
            payload.limit,
            _rag_service(request).accessible_recording_ids(user),
            payload.force_correction,
        )
        if assistant_message.generation_run_id is None:
            raise RuntimeError("Assistant message is missing its generation run")
        resume_after = _resume_after(None, last_event_id)
        cursor = resume_after if resume_after is not None else generation_service.cursor(assistant_message.generation_run_id)
        snapshot = generation_service.get(assistant_message.generation_run_id)
        stream = _stream_initial_turn(
            request,
            generation_service,
            generation_redis_event_store,
            conversation,
            payload.client_conversation_id,
            user_message,
            assistant_message,
            snapshot,
            cursor,
            include_snapshot=resume_after is None,
        )
        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )
    except PermissionError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Current workspace access denied") from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


async def _stream_initial_turn(
    request: Request,
    generation_service: GenerationService,
    redis_event_store: RedisStreamStore,
    conversation: Conversation,
    client_conversation_id: UUID,
    user_message: ConversationMessage,
    assistant_message: ConversationMessage,
    snapshot: GenerationSnapshot,
    cursor: str,
    *,
    include_snapshot: bool,
) -> AsyncGenerator[str]:
    ready = GenerationEvent(
        run_id=snapshot.id,
        seq=0,
        type="conversation.ready",
        at=datetime.now(UTC),
        data={
            "client_conversation_id": str(client_conversation_id),
            "conversation": conversation.model_dump(mode="json"),
            "user_message": user_message.model_dump(mode="json"),
            "assistant_message": assistant_message.model_dump(mode="json"),
            "generation_run_id": str(snapshot.id),
        },
    )
    yield encode_sse(ready, cursor)
    if include_snapshot:
        yield encode_sse(_snapshot_event(snapshot), cursor)
        if snapshot.status.is_terminal:
            return
    async for frame in _stream_events(request, generation_service, redis_event_store, snapshot.id, snapshot, cursor):
        yield frame


@router.get("/{conversation_id}/messages", response_model=ConversationMessagePage)
def get_messages(
    conversation_id: UUID,
    service: ConversationServiceDependency,
    user: CurrentUserDependency,
    before: Annotated[int | None, Query(gt=0)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> ConversationMessagePage:
    try:
        return service.messages(user, conversation_id, before, limit)
    except ConversationAccessDeniedError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Conversation access denied") from error


@router.post(
    "/{conversation_id}/generations/{generation_run_id}/resume",
    response_model=ConversationTurnResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def resume_conversation_generation(
    conversation_id: UUID,
    generation_run_id: UUID,
    payload: ResumeConversationGenerationRequest,
    request: Request,
    service: ConversationServiceDependency,
    user: CurrentUserDependency,
) -> ConversationTurnResponse:
    try:
        user_message, assistant_message, _history = service.resume_generation(
            user,
            conversation_id,
            generation_run_id,
            payload.client_request_id,
            reuse_checkpoint=payload.mode == "resume",
            scope_recording_ids=_rag_service(request).accessible_recording_ids(user),
        )
        if assistant_message.generation_run_id is None:
            raise RuntimeError("Assistant message is missing its resumed generation run")
        return ConversationTurnResponse(
            user_message=user_message,
            assistant_message=assistant_message,
            generation_run_id=assistant_message.generation_run_id,
        )
    except ConversationAccessDeniedError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Conversation access denied") from error
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation generation not found") from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.post(
    "/{conversation_id}/generations/{generation_run_id}/adjudication-decisions",
    response_model=ConversationTurnResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_adjudication_decision(
    conversation_id: UUID,
    generation_run_id: UUID,
    payload: AdjudicationDecisionRequest,
    request: Request,
    service: ConversationServiceDependency,
    user: CurrentUserDependency,
) -> ConversationTurnResponse:
    try:
        decision = ClaimConfirmationDecision.model_validate(payload.model_dump())
        user_message, assistant_message, _history = service.submit_adjudication_decision(
            user,
            conversation_id,
            generation_run_id,
            decision,
            _rag_service(request).accessible_recording_ids(user),
        )
        if assistant_message.generation_run_id is None:
            raise RuntimeError("Assistant message is missing its adjudication generation run")
        return ConversationTurnResponse(
            user_message=user_message,
            assistant_message=assistant_message,
            generation_run_id=assistant_message.generation_run_id,
        )
    except ConversationAccessDeniedError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Conversation access denied") from error
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation generation not found") from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_conversation(conversation_id: UUID, service: ConversationServiceDependency, user: CurrentUserDependency) -> Response:
    try:
        service.delete(user, conversation_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except ConversationBusyError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except ConversationAccessDeniedError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the conversation owner can delete it") from error


@router.post("/{conversation_id}/messages", response_model=ConversationTurnResponse, status_code=status.HTTP_202_ACCEPTED)
async def send_message(
    conversation_id: UUID,
    payload: SendConversationMessageRequest,
    request: Request,
    service: ConversationServiceDependency,
    user: CurrentUserDependency,
) -> ConversationTurnResponse:
    try:
        user_message, assistant_message, _history = service.create_turn(
            user,
            conversation_id,
            payload.content_blocks,
            payload.client_message_id,
            payload.limit,
            _rag_service(request).accessible_recording_ids(user),
            payload.force_correction,
        )
        if assistant_message.generation_run_id is None:
            raise RuntimeError("Assistant message is missing its generation run")
        return ConversationTurnResponse(
            user_message=user_message,
            assistant_message=assistant_message,
            generation_run_id=assistant_message.generation_run_id,
        )
    except ConversationBusyError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except ConversationAccessDeniedError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Conversation access denied") from error
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found") from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
