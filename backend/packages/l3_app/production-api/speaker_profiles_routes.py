from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy import text

from production_dependencies import CurrentUserDependency, DatabaseEngineDependency, StorageDependency

router = APIRouter()
_SUPPORTED_SUFFIXES = {".mp3", ".m4a", ".wav", ".mp4"}


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class SpeakerProfileSampleResponse(ApiModel):
    id: UUID
    speaker_profile_id: UUID
    file_name: str
    storage_path: str
    mime_type: str
    file_size_bytes: int
    duration_seconds: int | None
    status: Literal["uploaded", "processing", "completed", "failed"]
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class SpeakerProfileResponse(ApiModel):
    id: UUID
    display_name: str
    status: Literal["active", "inactive"]
    notes: str | None
    created_at: datetime
    updated_at: datetime
    samples: list[SpeakerProfileSampleResponse] = []


@router.get("", response_model=list[SpeakerProfileResponse])
def list_speaker_profiles(engine: DatabaseEngineDependency, _: CurrentUserDependency) -> list[SpeakerProfileResponse]:
    with engine.connect() as connection:
        profiles = [dict(row) for row in connection.execute(text("select * from speaker_profiles order by created_at desc")).mappings()]
        samples = [dict(row) for row in connection.execute(text("select * from speaker_profile_samples order by created_at desc")).mappings()]
    return [
        SpeakerProfileResponse(
            **profile,
            samples=[SpeakerProfileSampleResponse.model_validate(sample) for sample in samples if sample["speaker_profile_id"] == profile["id"]],
        )
        for profile in profiles
    ]


@router.post("", response_model=SpeakerProfileResponse, status_code=status.HTTP_201_CREATED)
def create_speaker_profile(
    request: Request,
    engine: DatabaseEngineDependency,
    _: CurrentUserDependency,
    display_name: Annotated[str, Form()],
    profile_status: Annotated[Literal["active", "inactive"], Form(alias="status")] = "active",
    notes: Annotated[str | None, Form()] = None,
) -> SpeakerProfileResponse | RedirectResponse:
    if not display_name.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="目标人物名称不能为空")
    with engine.begin() as connection:
        row = (
            connection.execute(
                text("insert into speaker_profiles (display_name, status, notes) values (:display_name, :status, :notes) returning *"),
                {"display_name": display_name.strip(), "status": profile_status, "notes": notes.strip() if notes and notes.strip() else None},
            )
            .mappings()
            .one()
        )
    if _is_html_form(request):
        return RedirectResponse("/speaker-profiles", status_code=status.HTTP_303_SEE_OTHER)
    return SpeakerProfileResponse(**dict(row), samples=[])


@router.post("/{speaker_profile_id}/samples", response_model=SpeakerProfileSampleResponse, status_code=status.HTTP_201_CREATED)
async def create_speaker_profile_sample(
    speaker_profile_id: UUID,
    request: Request,
    engine: DatabaseEngineDependency,
    storage: StorageDependency,
    _: CurrentUserDependency,
    audio: Annotated[UploadFile, File()],
) -> SpeakerProfileSampleResponse | RedirectResponse:
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES and not (audio.content_type or "").startswith("audio/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请选择 mp3、m4a、wav 或 mp4 格式的参考音频")
    file_name = Path(audio.filename or "sample").name
    key = f"speaker-samples/{speaker_profile_id}/{uuid4().hex}{suffix or '.audio'}"
    destination = storage.resolve(key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with destination.open("wb") as output:
            while chunk := await audio.read(1024 * 1024):
                size += len(chunk)
                output.write(chunk)
        if size == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请选择非空的参考音频")
        with engine.begin() as connection:
            exists = connection.execute(text("select 1 from speaker_profiles where id = :id"), {"id": speaker_profile_id}).scalar_one_or_none()
            if exists is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Speaker profile not found")
            row = (
                connection.execute(
                    text(
                        """insert into speaker_profile_samples (speaker_profile_id, file_name, storage_path, mime_type, file_size_bytes, status)
                    values (:speaker_profile_id, :file_name, :storage_path, :mime_type, :file_size_bytes, 'completed') returning *"""
                    ),
                    {
                        "speaker_profile_id": speaker_profile_id,
                        "file_name": file_name,
                        "storage_path": key,
                        "mime_type": audio.content_type or "application/octet-stream",
                        "file_size_bytes": size,
                    },
                )
                .mappings()
                .one()
            )
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await audio.close()
    if _is_html_form(request):
        return RedirectResponse("/speaker-profiles", status_code=status.HTTP_303_SEE_OTHER)
    return SpeakerProfileSampleResponse(**dict(row))


@router.post("/{speaker_profile_id}/delete", response_model=None)
def delete_speaker_profile(
    speaker_profile_id: UUID, request: Request, engine: DatabaseEngineDependency, storage: StorageDependency, _: CurrentUserDependency
) -> RedirectResponse | dict[str, bool]:
    with engine.begin() as connection:
        paths = [
            str(row)
            for row in connection.execute(
                text("select storage_path from speaker_profile_samples where speaker_profile_id = :id"), {"id": speaker_profile_id}
            ).scalars()
        ]
        deleted = connection.execute(text("delete from speaker_profiles where id = :id returning id"), {"id": speaker_profile_id}).scalar_one_or_none()
    if deleted is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Speaker profile not found")
    for path in paths:
        storage.resolve(path).unlink(missing_ok=True)
    if _is_html_form(request):
        return RedirectResponse("/speaker-profiles", status_code=status.HTTP_303_SEE_OTHER)
    return {"deleted": True}


def _is_html_form(request: Request) -> bool:
    return "text/html" in request.headers.get("accept", "")
