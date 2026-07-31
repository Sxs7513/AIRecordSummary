from __future__ import annotations

from typing import Any, Never
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.exc import IntegrityError

from l2_core.asr_lab.service import AsrLabConflictError, AsrLabNotFoundError, AsrLabPermissionError
from training_dependencies import AsrLabServiceDependency, CurrentUserDependency

training_router = APIRouter()
model_router = APIRouter()


class CreateTrainingRunRequest(BaseModel):
    dataset_id: UUID | None = None
    dataset_version_id: UUID | None = None
    base_model_version_id: UUID
    preset_name: str = Field(default="lora_safe_v1", min_length=1, max_length=160)
    candidate_model_name: str = Field(min_length=1, max_length=160)
    run_validation: bool = False
    idempotency_key: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def require_one_dataset_reference(self) -> CreateTrainingRunRequest:
        if (self.dataset_id is None) == (self.dataset_version_id is None):
            raise ValueError("Provide exactly one of dataset_id or dataset_version_id")
        return self


@training_router.get("")
def list_training_runs(service: AsrLabServiceDependency, user: CurrentUserDependency) -> list[dict[str, Any]]:
    return service.list_training_runs(user)


@training_router.post("", status_code=status.HTTP_202_ACCEPTED)
def create_training_run(
    payload: CreateTrainingRunRequest,
    service: AsrLabServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    try:
        return service.create_training_run(user, **payload.model_dump())
    except Exception as error:
        _raise_api_error(error)


@training_router.post("/{run_id}:cancel")
def cancel_training_run(run_id: UUID, service: AsrLabServiceDependency, user: CurrentUserDependency) -> dict[str, Any]:
    try:
        return service.cancel_training_run(user, run_id)
    except Exception as error:
        _raise_api_error(error)


@training_router.delete("/{run_id}")
def delete_training_run(run_id: UUID, service: AsrLabServiceDependency, user: CurrentUserDependency) -> dict[str, Any]:
    try:
        return service.delete_training_run(user, run_id)
    except Exception as error:
        _raise_api_error(error)


@model_router.get("")
def list_model_versions(service: AsrLabServiceDependency, user: CurrentUserDependency) -> list[dict[str, Any]]:
    return service.list_models(user)


@model_router.post("/{model_id}:approve")
def approve_model_version(model_id: UUID, service: AsrLabServiceDependency, user: CurrentUserDependency) -> dict[str, Any]:
    try:
        return service.update_model_status(user, model_id, "approved")
    except Exception as error:
        _raise_api_error(error)


@model_router.post("/{model_id}:retire")
def retire_model_version(model_id: UUID, service: AsrLabServiceDependency, user: CurrentUserDependency) -> dict[str, Any]:
    try:
        return service.update_model_status(user, model_id, "retired")
    except Exception as error:
        _raise_api_error(error)


def _raise_api_error(error: Exception) -> Never:
    if isinstance(error, AsrLabNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    if isinstance(error, AsrLabPermissionError):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(error)) from error
    if isinstance(error, AsrLabConflictError | IntegrityError):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    if isinstance(error, ValueError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    raise error
