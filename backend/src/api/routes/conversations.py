from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field

from access.conversations import ConversationAccessDeniedError
from api.dependencies import ConversationServiceDependency, CurrentUserDependency
from conversations.contracts import Conversation, ConversationBusyError, ConversationMessage, ConversationMessagePage, ConversationNotFoundError
from generation.contracts import TextBlock
from rag.service import RagService, RagWorkflowRunner

router = APIRouter()


class CreateConversationRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class SendConversationMessageRequest(BaseModel):
    content_blocks: list[TextBlock] = Field(min_length=1)
    client_message_id: UUID
    limit: int = Field(default=10, ge=1, le=20)


class ConversationTurnResponse(BaseModel):
    user_message: ConversationMessage
    assistant_message: ConversationMessage
    generation_run_id: UUID


def _rag_service(request: Request) -> RagService:
    return request.app.state.rag_service


def _rag_runner(request: Request) -> RagWorkflowRunner:
    return request.app.state.rag_workflow_runner


@router.get("", response_model=list[Conversation])
def list_conversations(service: ConversationServiceDependency, user: CurrentUserDependency) -> list[Conversation]:
    return service.list(user)


@router.post("", response_model=Conversation, status_code=status.HTTP_201_CREATED)
def create_conversation(payload: CreateConversationRequest, service: ConversationServiceDependency, user: CurrentUserDependency) -> Conversation:
    try:
        return service.create(user, payload.title)
    except PermissionError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Current workspace access denied") from error


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
        user_message, assistant_message, history = service.create_turn(user, conversation_id, payload.content_blocks, payload.client_message_id, payload.limit)
        if assistant_message.generation_run_id is None:
            raise RuntimeError("Assistant message is missing its generation run")
        rag_service = _rag_service(request)
        _rag_runner(request).submit(
            assistant_message.generation_run_id,
            "".join(block.value for block in payload.content_blocks),
            payload.limit,
            rag_service.accessible_recording_ids(user),
            history,
            service.mark_streaming,
            service.sync_generation,
        )
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
