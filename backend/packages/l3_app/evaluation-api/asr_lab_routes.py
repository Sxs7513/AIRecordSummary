from __future__ import annotations

from dataclasses import asdict
from typing import Annotated, Any, Literal, Never
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

from evaluation_dependencies import AsrLabServiceDependency, CurrentUserDependency, StorageDependency
from l2_core.asr_lab.service import AsrLabConflictError, AsrLabNotFoundError, AsrLabPermissionError

evaluation_router = APIRouter()
model_router = APIRouter()


class CreateDatasetRequest(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=2000)


class ImportRecordingRequest(BaseModel):
    recording_id: UUID


class AnnotationRequest(BaseModel):
    source_asset_id: UUID
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    reference_text: str = Field(min_length=1)
    language: str | None = Field(default="zh", max_length=40)
    train_allowed: bool = True
    evaluation_allowed: bool = True
    contains_sensitive_data: bool = False
    project_persistence_password: str | None = Field(default=None, min_length=8, max_length=256)


class UpdateAnnotationRequest(AnnotationRequest):
    revision: int = Field(gt=0)


class RevisionRequest(BaseModel):
    revision: int = Field(gt=0)


class ImportEncryptedProjectDatasetRequest(BaseModel):
    password: str = Field(min_length=8, max_length=256)


class DatasetVersionRequest(BaseModel):
    normalization_name: str = "zh_asr"
    normalization_version: str = "v1"
    seed: str = Field(default="asr-lab-v1", min_length=1, max_length=160)


class FreezeDatasetVersionRequest(DatasetVersionRequest):
    expected_checksum: str = Field(min_length=64, max_length=64)


class CreateEvaluationRunRequest(BaseModel):
    dataset_version_id: UUID
    split: Literal["validation", "test"] = "test"
    model_version_ids: list[UUID] = Field(min_length=1, max_length=8)
    normalization_name: str = "zh_asr"
    normalization_version: str = "v1"
    idempotency_key: str = Field(min_length=1, max_length=200)


@evaluation_router.get("/datasets")
def list_datasets(service: AsrLabServiceDependency, user: CurrentUserDependency) -> list[dict[str, Any]]:
    return service.list_datasets(user)


@evaluation_router.get("/project-datasets")
def list_encrypted_project_datasets(
    service: AsrLabServiceDependency,
    user: CurrentUserDependency,
) -> list[dict[str, Any]]:
    del user
    return service.list_encrypted_project_datasets()


@evaluation_router.post("/project-datasets/{package_id}:import", status_code=status.HTTP_201_CREATED)
def import_encrypted_project_dataset(
    package_id: str,
    payload: ImportEncryptedProjectDatasetRequest,
    service: AsrLabServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    try:
        return service.import_encrypted_project_dataset(user, package_id, payload.password)
    except Exception as error:
        _raise_api_error(error)


@evaluation_router.post("/datasets", status_code=status.HTTP_201_CREATED)
def create_dataset(payload: CreateDatasetRequest, service: AsrLabServiceDependency, user: CurrentUserDependency) -> dict[str, Any]:
    try:
        return service.create_dataset(user, payload.name, payload.description)
    except Exception as error:
        _raise_api_error(error)


@evaluation_router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: UUID, service: AsrLabServiceDependency, user: CurrentUserDependency) -> dict[str, Any]:
    try:
        return service.get_dataset(user, dataset_id)
    except Exception as error:
        _raise_api_error(error)


@evaluation_router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: UUID, service: AsrLabServiceDependency, user: CurrentUserDependency) -> dict[str, Any]:
    try:
        return service.delete_dataset(user, dataset_id)
    except Exception as error:
        _raise_api_error(error)


@evaluation_router.post("/datasets/{dataset_id}/samples", status_code=status.HTTP_201_CREATED)
async def create_sample(
    dataset_id: UUID,
    service: AsrLabServiceDependency,
    user: CurrentUserDependency,
    audio: Annotated[UploadFile, File(description="Original audio kept only for this request")],
    start_ms: Annotated[int, Form(ge=0)],
    end_ms: Annotated[int, Form(gt=0)],
    reference_text: Annotated[str, Form(min_length=1)],
    language: Annotated[str | None, Form(max_length=40)] = "zh",
    train_allowed: Annotated[bool, Form()] = True,
    evaluation_allowed: Annotated[bool, Form()] = True,
    contains_sensitive_data: Annotated[bool, Form()] = False,
    project_persistence_password: Annotated[str | None, Form(min_length=8, max_length=256)] = None,
) -> dict[str, Any]:
    try:
        return await service.create_sample(
            user,
            dataset_id,
            audio_upload=audio,
            start_ms=start_ms,
            end_ms=end_ms,
            reference_text=reference_text,
            language=language,
            train_allowed=train_allowed,
            evaluation_allowed=evaluation_allowed,
            contains_sensitive_data=contains_sensitive_data,
            project_persistence_password=project_persistence_password,
        )
    except Exception as error:
        _raise_api_error(error)


@evaluation_router.post("/datasets/{dataset_id}/assets:import-recording", status_code=status.HTTP_201_CREATED)
def import_recording(
    dataset_id: UUID,
    payload: ImportRecordingRequest,
    service: AsrLabServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    try:
        return service.import_recording(user, dataset_id, payload.recording_id)
    except Exception as error:
        _raise_api_error(error)


@evaluation_router.get("/assets/{asset_id}/audio")
def get_asset_audio(
    asset_id: UUID,
    service: AsrLabServiceDependency,
    storage: StorageDependency,
    user: CurrentUserDependency,
) -> FileResponse:
    try:
        asset = service.get_asset_audio(user, asset_id)
        path = storage.resolve(str(asset["storage_path"]))
        if not path.is_file():
            raise AsrLabNotFoundError("Audio file not found")
        return FileResponse(path, media_type=str(asset["mime_type"]), filename=str(asset["file_name"]))
    except Exception as error:
        _raise_api_error(error)


@evaluation_router.delete("/assets/{asset_id}")
def delete_asset(
    asset_id: UUID,
    service: AsrLabServiceDependency,
    user: CurrentUserDependency,
    delete_annotations: bool = False,
) -> dict[str, Any]:
    try:
        return service.delete_asset(user, asset_id, delete_annotations=delete_annotations)
    except Exception as error:
        _raise_api_error(error)


@evaluation_router.post("/datasets/{dataset_id}/annotations", status_code=status.HTTP_201_CREATED)
def create_annotation(
    dataset_id: UUID,
    payload: AnnotationRequest,
    service: AsrLabServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    try:
        return service.create_annotation(user, dataset_id, **payload.model_dump())
    except Exception as error:
        _raise_api_error(error)


@evaluation_router.patch("/annotations/{annotation_id}")
def update_annotation(
    annotation_id: UUID,
    payload: UpdateAnnotationRequest,
    service: AsrLabServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    try:
        return service.update_annotation(user, annotation_id, **payload.model_dump(exclude={"source_asset_id"}))
    except Exception as error:
        _raise_api_error(error)


@evaluation_router.post("/annotations/{annotation_id}:review")
def review_annotation(
    annotation_id: UUID,
    payload: RevisionRequest,
    service: AsrLabServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    try:
        return service.review_annotation(user, annotation_id, payload.revision)
    except Exception as error:
        _raise_api_error(error)


@evaluation_router.post("/annotations/{annotation_id}:approve")
def approve_annotation(
    annotation_id: UUID,
    payload: RevisionRequest,
    service: AsrLabServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    try:
        return service.approve_annotation(user, annotation_id, payload.revision)
    except Exception as error:
        _raise_api_error(error)


@evaluation_router.delete("/annotations/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_annotation(
    annotation_id: UUID,
    revision: int,
    service: AsrLabServiceDependency,
    user: CurrentUserDependency,
) -> None:
    try:
        service.delete_annotation(user, annotation_id, revision)
    except Exception as error:
        _raise_api_error(error)


@evaluation_router.post("/datasets/{dataset_id}/versions:preview")
def preview_dataset_version(
    dataset_id: UUID,
    payload: DatasetVersionRequest,
    service: AsrLabServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    try:
        return asdict(service.preview_dataset_version(user, dataset_id, **payload.model_dump()))
    except Exception as error:
        _raise_api_error(error)


@evaluation_router.post("/datasets/{dataset_id}/versions:freeze", status_code=status.HTTP_201_CREATED)
def freeze_dataset_version(
    dataset_id: UUID,
    payload: FreezeDatasetVersionRequest,
    service: AsrLabServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    try:
        return service.freeze_dataset_version(user, dataset_id, **payload.model_dump())
    except Exception as error:
        _raise_api_error(error)


@evaluation_router.get("/runs")
def list_evaluation_runs(service: AsrLabServiceDependency, user: CurrentUserDependency) -> list[dict[str, Any]]:
    return service.list_evaluation_runs(user)


@evaluation_router.post("/runs", status_code=status.HTTP_202_ACCEPTED)
def create_evaluation_run(
    payload: CreateEvaluationRunRequest,
    service: AsrLabServiceDependency,
    user: CurrentUserDependency,
) -> dict[str, Any]:
    try:
        return service.create_evaluation_run(user, **payload.model_dump())
    except Exception as error:
        _raise_api_error(error)


@evaluation_router.get("/runs/{run_id}")
def get_evaluation_run(run_id: UUID, service: AsrLabServiceDependency, user: CurrentUserDependency) -> dict[str, Any]:
    try:
        return service.get_evaluation_run(user, run_id)
    except Exception as error:
        _raise_api_error(error)


@evaluation_router.post("/runs/{run_id}:cancel")
def cancel_evaluation_run(run_id: UUID, service: AsrLabServiceDependency, user: CurrentUserDependency) -> dict[str, Any]:
    try:
        return service.cancel_evaluation_run(user, run_id)
    except Exception as error:
        _raise_api_error(error)


@evaluation_router.delete("/runs/{run_id}")
def delete_evaluation_run(run_id: UUID, service: AsrLabServiceDependency, user: CurrentUserDependency) -> dict[str, Any]:
    try:
        return service.delete_evaluation_run(user, run_id)
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
