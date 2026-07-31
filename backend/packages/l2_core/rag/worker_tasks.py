from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from l1_foundation.task_runtime.resources import ResourceQueue
from l1_foundation.worker import ComputeCommand


class RerankCandidateInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class RerankInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1)
    candidates: list[RerankCandidateInput] = Field(min_length=1)
    max_total_tokens: int = Field(gt=0)


class RerankScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_id: str
    score: float


class RerankResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    scores: list[RerankScore]
    input_tokens: int = Field(ge=0)
    skipped_candidates: int = Field(ge=0)


def rerank_command(query: str, candidates: Sequence[RerankCandidateInput], max_total_tokens: int) -> ComputeCommand[RerankInput]:
    return ComputeCommand(
        task_id=uuid4(),
        operation="rerank.score",
        operation_version="1",
        resource_queue=ResourceQueue.GPU_NORMAL,
        input=RerankInput(query=query, candidates=list(candidates), max_total_tokens=max_total_tokens),
    )
