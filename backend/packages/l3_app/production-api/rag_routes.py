from __future__ import annotations

import logging
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from l2_core.generation.service import GenerationService
from l2_core.rag.queue import GenerationCommandPublisher, RagGenerationWorkItem
from l2_core.rag.service import RagService
from production_dependencies import CurrentUserDependency

router = APIRouter()
logger = logging.getLogger("rag")


class RagQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    limit: int = Field(default=10, ge=1, le=20)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)


class RagGenerationStarted(BaseModel):
    generation_run_id: str


def get_rag_service(request: Request) -> RagService:
    return request.app.state.rag_service


RagServiceDependency = Annotated[RagService, Depends(get_rag_service)]


def get_generation_service(request: Request) -> GenerationService:
    return request.app.state.generation_service


GenerationServiceDependency = Annotated[GenerationService, Depends(get_generation_service)]


def get_generation_command_publisher(request: Request) -> GenerationCommandPublisher:
    return request.app.state.generation_command_publisher


GenerationCommandPublisherDependency = Annotated[GenerationCommandPublisher, Depends(get_generation_command_publisher)]


@router.post("/queries", response_model=RagGenerationStarted, status_code=status.HTTP_202_ACCEPTED)
async def create_rag_query(
    payload: RagQueryRequest,
    service: RagServiceDependency,
    generation_service: GenerationServiceDependency,
    publisher: GenerationCommandPublisherDependency,
    user: CurrentUserDependency,
) -> RagGenerationStarted:
    logger.info("rag query creation received: query_chars=%d limit=%d", len(payload.query.strip()), payload.limit)
    try:
        idempotency_key = payload.idempotency_key or f"rag-answer:{uuid4()}"
        generation = service.create_answer_generation(
            generation_service,
            user,
            payload.query,
            payload.limit,
            idempotency_key,
        )
        try:
            await publisher.submit_rag(
                RagGenerationWorkItem(
                    run_id=generation.id,
                    workspace_id=user.current_workspace_id,
                    query=payload.query,
                    limit=payload.limit,
                    scope_recording_ids=service.accessible_recording_ids(user),
                    generation=generation_service.command(generation.id),
                )
            )
        except Exception as error:
            generation_service.event_sink(generation.id).fail(
                "kafka_unavailable",
                str(error) or type(error).__name__,
                retryable=True,
            )
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Generation queue unavailable") from error
        return RagGenerationStarted(generation_run_id=str(generation.id))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("rag query request failed: query_chars=%d limit=%d", len(payload.query.strip()), payload.limit)
        raise HTTPException(status_code=500, detail="录音问答暂时不可用，请查看 Python 服务日志") from error
