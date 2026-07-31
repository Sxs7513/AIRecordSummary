from __future__ import annotations

from typing import Any, Never
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

from evaluation_dependencies import CurrentUserDependency, RagEvaluationServiceDependency
from l2_core.rag_evaluation.service import (
    RagEvaluationConflictError,
    RagEvaluationNotFoundError,
    RagEvaluationPermissionError,
)

router = APIRouter()


def _uuid_list() -> list[UUID]:
    return []


def _string_list() -> list[str]:
    return []


class CreateDatasetRequest(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=2000)


class CreateCaseRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    recording_ids: list[UUID] = Field(default_factory=_uuid_list, max_length=100)
    tags: list[str] = Field(default_factory=_string_list, max_length=30)
    group_key: str | None = Field(default=None, max_length=200)


class AddEvidenceRequest(BaseModel):
    chunk_id: UUID
    relevance: int = Field(default=3, ge=1, le=3)


class RevisionRequest(BaseModel):
    revision: int = Field(gt=0)


class FreezeVersionRequest(BaseModel):
    expected_checksum: str = Field(min_length=64, max_length=64)


class CreateRunRequest(BaseModel):
    dataset_version_id: UUID
    idempotency_key: str = Field(min_length=1, max_length=200)
    baseline_run_id: UUID | None = None


@router.get("/datasets")
def list_datasets(service: RagEvaluationServiceDependency, user: CurrentUserDependency) -> list[dict[str, Any]]:
    return service.list_datasets(user)


@router.post("/datasets", status_code=status.HTTP_201_CREATED)
def create_dataset(
    payload: CreateDatasetRequest,
    service: RagEvaluationServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    try:
        return service.create_dataset(user, payload.name, payload.description)
    except Exception as error:
        _raise_api_error(error)


@router.get("/datasets/{dataset_id}")
def get_dataset(
    dataset_id: UUID,
    service: RagEvaluationServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    try:
        return service.get_dataset(user, dataset_id)
    except Exception as error:
        _raise_api_error(error)


@router.post("/datasets/{dataset_id}/cases", status_code=status.HTTP_201_CREATED)
def create_case(
    dataset_id: UUID,
    payload: CreateCaseRequest,
    service: RagEvaluationServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    try:
        return service.create_case(user, dataset_id, **payload.model_dump())
    except Exception as error:
        _raise_api_error(error)


@router.delete("/cases/{case_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_case(
    case_id: UUID,
    service: RagEvaluationServiceDependency,
    user: CurrentUserDependency,
) -> None:
    try:
        service.delete_case(user, case_id)
    except Exception as error:
        _raise_api_error(error)


@router.post("/cases/{case_id}:archive")
def archive_case(
    case_id: UUID,
    service: RagEvaluationServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    try:
        return service.archive_case(user, case_id)
    except Exception as error:
        _raise_api_error(error)


@router.post("/cases/{case_id}/evidence", status_code=status.HTTP_201_CREATED)
def add_evidence(
    case_id: UUID,
    payload: AddEvidenceRequest,
    service: RagEvaluationServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    try:
        return service.add_evidence(user, case_id, payload.chunk_id, payload.relevance)
    except Exception as error:
        _raise_api_error(error)


@router.delete("/evidence/{evidence_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_evidence(
    evidence_id: UUID,
    service: RagEvaluationServiceDependency,
    user: CurrentUserDependency,
) -> None:
    try:
        service.delete_evidence(user, evidence_id)
    except Exception as error:
        _raise_api_error(error)


@router.post("/cases/{case_id}:{action}")
def transition_case(
    case_id: UUID,
    action: str,
    payload: RevisionRequest,
    service: RagEvaluationServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    if action not in {"review", "approve"}:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown case transition")
    try:
        return service.transition_case(user, case_id, payload.revision, action)
    except Exception as error:
        _raise_api_error(error)


@router.get("/chunks")
def search_chunks(
    service: RagEvaluationServiceDependency,
    user: CurrentUserDependency,
    query: str = "",
    recording_id: UUID | None = None,
    limit: int = 30,
    offset: int = 0,
) -> list[dict[str, Any]]:
    try:
        return service.search_chunks(user, query=query, recording_id=recording_id, limit=limit, offset=offset)
    except Exception as error:
        _raise_api_error(error)


@router.get("/recordings")
def list_recordings(
    service: RagEvaluationServiceDependency,
    user: CurrentUserDependency,
) -> list[dict[str, Any]]:
    try:
        return service.list_recordings(user)
    except Exception as error:
        _raise_api_error(error)


@router.post("/datasets/{dataset_id}/versions:preview")
def preview_version(
    dataset_id: UUID,
    service: RagEvaluationServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    try:
        return service.preview_version(user, dataset_id)
    except Exception as error:
        _raise_api_error(error)


@router.post("/datasets/{dataset_id}/versions:freeze", status_code=status.HTTP_201_CREATED)
def freeze_version(
    dataset_id: UUID,
    payload: FreezeVersionRequest,
    service: RagEvaluationServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    try:
        return service.freeze_version(user, dataset_id, payload.expected_checksum)
    except Exception as error:
        _raise_api_error(error)


@router.get("/runs")
def list_runs(service: RagEvaluationServiceDependency, user: CurrentUserDependency) -> list[dict[str, Any]]:
    return service.list_runs(user)


@router.post("/runs", status_code=status.HTTP_202_ACCEPTED)
def create_run(
    payload: CreateRunRequest,
    service: RagEvaluationServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    try:
        return service.create_run(user, **payload.model_dump())
    except Exception as error:
        _raise_api_error(error)


@router.get("/runs/{run_id}")
def get_run(
    run_id: UUID,
    service: RagEvaluationServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    try:
        return service.get_run(user, run_id)
    except Exception as error:
        _raise_api_error(error)


@router.post("/runs/{run_id}:cancel")
def cancel_run(
    run_id: UUID,
    service: RagEvaluationServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    try:
        return service.cancel_run(user, run_id)
    except Exception as error:
        _raise_api_error(error)


@router.delete("/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_run(
    run_id: UUID,
    service: RagEvaluationServiceDependency,
    user: CurrentUserDependency,
) -> None:
    try:
        service.delete_run(user, run_id)
    except Exception as error:
        _raise_api_error(error)


def _raise_api_error(error: Exception) -> Never:
    if isinstance(error, RagEvaluationNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    if isinstance(error, RagEvaluationPermissionError):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(error)) from error
    if isinstance(error, (RagEvaluationConflictError, IntegrityError)):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    if isinstance(error, ValueError):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    raise error
