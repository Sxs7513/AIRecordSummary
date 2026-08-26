from __future__ import annotations

from typing import Any, Literal, Never
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

from evaluation_dependencies import (
    CurrentUserDependency,
    RagAdjudicationEvaluationServiceDependency,
)
from l2_core.rag_evaluation.service import (
    RagEvaluationConflictError,
    RagEvaluationNotFoundError,
    RagEvaluationPermissionError,
)

router = APIRouter()


def _strings() -> list[str]:
    return []


class CreateDatasetRequest(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=2000)


class CreateCaseRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    tags: list[str] = Field(default_factory=_strings, max_length=30)


class UpdateCaseRequest(CreateCaseRequest):
    revision: int = Field(gt=0)


class AddEvidenceRequest(BaseModel):
    chunk_id: UUID
    role: str = Field(pattern="^(target|reference)$")
    position: int = Field(ge=0)


class UpdateEvidenceRequest(BaseModel):
    role: str = Field(pattern="^(target|reference)$")
    position: int = Field(ge=0)


class AddCorrectionRequest(BaseModel):
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    original_expression: str = Field(min_length=1, max_length=300)
    accepted_expressions: list[str] = Field(min_length=1, max_length=10)
    importance: Literal["important", "minor"] = "important"


class RevisionRequest(BaseModel):
    revision: int = Field(gt=0)


class FreezeRequest(BaseModel):
    expected_checksum: str = Field(min_length=64, max_length=64)


class CreateRunRequest(BaseModel):
    dataset_version_id: UUID
    idempotency_key: str = Field(min_length=1, max_length=200)


@router.get("/datasets")
def list_datasets(service: RagAdjudicationEvaluationServiceDependency, user: CurrentUserDependency) -> list[dict[str, Any]]:
    return service.list_datasets(user)


@router.post("/datasets", status_code=status.HTTP_201_CREATED)
def create_dataset(payload: CreateDatasetRequest, service: RagAdjudicationEvaluationServiceDependency, user: CurrentUserDependency) -> dict[str, Any]:
    return _call(service.create_dataset, user, payload.name, payload.description)


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: UUID, service: RagAdjudicationEvaluationServiceDependency, user: CurrentUserDependency) -> dict[str, Any]:
    return _call(service.get_dataset, user, dataset_id)


@router.post("/datasets/{dataset_id}/cases", status_code=status.HTTP_201_CREATED)
def create_case(
    dataset_id: UUID, payload: CreateCaseRequest, service: RagAdjudicationEvaluationServiceDependency, user: CurrentUserDependency
) -> dict[str, Any]:
    return _call(service.create_case, user, dataset_id, payload.query, payload.tags)


@router.delete("/cases/{case_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_case(case_id: UUID, service: RagAdjudicationEvaluationServiceDependency, user: CurrentUserDependency) -> None:
    _call(service.delete_case, user, case_id)


@router.patch("/cases/{case_id}")
def update_case(
    case_id: UUID,
    payload: UpdateCaseRequest,
    service: RagAdjudicationEvaluationServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    return _call(service.update_case, user, case_id, **payload.model_dump())


@router.post("/cases/{case_id}/evidence", status_code=status.HTTP_201_CREATED)
def add_evidence(
    case_id: UUID, payload: AddEvidenceRequest, service: RagAdjudicationEvaluationServiceDependency, user: CurrentUserDependency
) -> dict[str, Any]:
    return _call(service.add_evidence, user, case_id, payload.chunk_id, payload.role, payload.position)


@router.patch("/evidence/{evidence_id}")
def update_evidence(
    evidence_id: UUID, payload: UpdateEvidenceRequest, service: RagAdjudicationEvaluationServiceDependency, user: CurrentUserDependency
) -> dict[str, Any]:
    return _call(service.update_evidence, user, evidence_id, role=payload.role, position=payload.position)


@router.delete("/evidence/{evidence_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_evidence(evidence_id: UUID, service: RagAdjudicationEvaluationServiceDependency, user: CurrentUserDependency) -> None:
    _call(service.delete_evidence, user, evidence_id)


@router.post("/evidence/{evidence_id}/corrections", status_code=status.HTTP_201_CREATED)
def add_correction(
    evidence_id: UUID, payload: AddCorrectionRequest, service: RagAdjudicationEvaluationServiceDependency, user: CurrentUserDependency
) -> dict[str, Any]:
    return _call(service.add_correction, user, evidence_id, **payload.model_dump())


@router.delete("/corrections/{correction_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_correction(correction_id: UUID, service: RagAdjudicationEvaluationServiceDependency, user: CurrentUserDependency) -> None:
    _call(service.delete_correction, user, correction_id)


@router.patch("/corrections/{correction_id}")
def update_correction(
    correction_id: UUID,
    payload: AddCorrectionRequest,
    service: RagAdjudicationEvaluationServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    return _call(service.update_correction, user, correction_id, **payload.model_dump())


@router.post("/cases/{case_id}:{action}")
def transition_case(
    case_id: UUID, action: str, payload: RevisionRequest, service: RagAdjudicationEvaluationServiceDependency, user: CurrentUserDependency
) -> dict[str, Any]:
    if action not in {"review", "approve"}:
        raise HTTPException(status_code=404, detail="Unknown case transition")
    return _call(service.transition_case, user, case_id, payload.revision, action)


@router.get("/chunks")
def search_chunks(
    service: RagAdjudicationEvaluationServiceDependency,
    user: CurrentUserDependency,
    query: str = "",
    recording_id: UUID | None = None,
    limit: int = 30,
    offset: int = 0,
) -> list[dict[str, Any]]:
    return _call(service.search_chunks, user, query=query, recording_id=recording_id, limit=limit, offset=offset)


@router.get("/recordings")
def list_recordings(service: RagAdjudicationEvaluationServiceDependency, user: CurrentUserDependency) -> list[dict[str, Any]]:
    return _call(service.list_recordings, user)


@router.post("/datasets/{dataset_id}/versions:preview")
def preview_version(dataset_id: UUID, service: RagAdjudicationEvaluationServiceDependency, user: CurrentUserDependency) -> dict[str, Any]:
    return _call(service.preview_version, user, dataset_id)


@router.post("/datasets/{dataset_id}/versions:freeze", status_code=status.HTTP_201_CREATED)
def freeze_version(
    dataset_id: UUID, payload: FreezeRequest, service: RagAdjudicationEvaluationServiceDependency, user: CurrentUserDependency
) -> dict[str, Any]:
    return _call(service.freeze_version, user, dataset_id, payload.expected_checksum)


@router.get("/runs")
def list_runs(service: RagAdjudicationEvaluationServiceDependency, user: CurrentUserDependency) -> list[dict[str, Any]]:
    return service.list_runs(user)


@router.post("/runs", status_code=status.HTTP_202_ACCEPTED)
def create_run(payload: CreateRunRequest, service: RagAdjudicationEvaluationServiceDependency, user: CurrentUserDependency) -> dict[str, Any]:
    return _call(service.create_run, user, **payload.model_dump())


@router.get("/runs/{run_id}")
def get_run(run_id: UUID, service: RagAdjudicationEvaluationServiceDependency, user: CurrentUserDependency) -> dict[str, Any]:
    return _call(service.get_run, user, run_id)


@router.post("/runs/{run_id}:cancel")
def cancel_run(run_id: UUID, service: RagAdjudicationEvaluationServiceDependency, user: CurrentUserDependency) -> dict[str, Any]:
    return _call(service.cancel_run, user, run_id)


@router.delete("/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_run(run_id: UUID, service: RagAdjudicationEvaluationServiceDependency, user: CurrentUserDependency) -> None:
    _call(service.delete_run, user, run_id)


def _call(function: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except Exception as error:
        _raise_api_error(error)


def _raise_api_error(error: Exception) -> Never:
    if isinstance(error, RagEvaluationNotFoundError):
        raise HTTPException(status_code=404, detail=str(error)) from error
    if isinstance(error, RagEvaluationPermissionError):
        raise HTTPException(status_code=403, detail=str(error)) from error
    if isinstance(error, (RagEvaluationConflictError, IntegrityError)):
        raise HTTPException(status_code=409, detail=str(error)) from error
    if isinstance(error, ValueError):
        raise HTTPException(status_code=422, detail=str(error)) from error
    raise error
