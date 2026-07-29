from __future__ import annotations

import logging
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from dependencies import CurrentUserDependency
from l2_core.generation.service import GenerationService
from l2_core.rag.service import RagService, RagWorkflowRunner

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


def get_rag_workflow_runner(request: Request) -> RagWorkflowRunner:
    return request.app.state.rag_workflow_runner


RagWorkflowRunnerDependency = Annotated[RagWorkflowRunner, Depends(get_rag_workflow_runner)]


@router.post("/queries", response_model=RagGenerationStarted, status_code=status.HTTP_202_ACCEPTED)
async def create_rag_query(
    payload: RagQueryRequest,
    service: RagServiceDependency,
    generation_service: GenerationServiceDependency,
    workflow_runner: RagWorkflowRunnerDependency,
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
        workflow_runner.submit(generation.id, payload.query, payload.limit, service.accessible_recording_ids(user))
        return RagGenerationStarted(generation_run_id=str(generation.id))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("rag query request failed: query_chars=%d limit=%d", len(payload.query.strip()), payload.limit)
        raise HTTPException(status_code=500, detail="录音问答暂时不可用，请查看 Python 服务日志") from error
