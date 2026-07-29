from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from dependencies import (
    CurrentUserDependency,
    RecordingServiceDependency,
    RecordingSummaryRegenerationServiceDependency,
    StorageDependency,
)
from l2_core.access.recordings import RecordingAccessDeniedError
from l2_core.application.recordings import RecordingNotFoundError, RecordingNotRetryableError, RecordingStageNotRetryableError
from l2_core.audio_processing.stages.summary.regeneration import RecordingSummaryNotReadyError

router = APIRouter()


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class RecordingResponse(ApiModel):
    id: UUID
    title: str
    file_name: str
    location: str | None
    mime_type: str
    file_size_bytes: int
    duration_seconds: int | None
    status: Literal["uploaded", "processing", "completed", "failed"]
    error_message: str | None
    uploaded_at: datetime
    created_at: datetime
    updated_at: datetime


class PipelineRunResponse(ApiModel):
    id: UUID
    recording_id: UUID
    pipeline_name: str
    pipeline_version: str
    status: Literal["queued", "running", "succeeded", "partial_failed", "failed", "cancelled"]
    started_at: datetime | None
    finished_at: datetime | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class StageRunResponse(ApiModel):
    id: UUID
    pipeline_run_id: UUID
    recording_id: UUID
    node_name: str
    stage_name: str
    stage_version: str
    required: bool
    resource_queue: Literal["cpu", "gpu_normal", "gpu_high"]
    status: Literal["pending", "running", "succeeded", "retry_waiting", "failed", "cancelled", "skipped"]
    attempt_count: int
    max_attempts: int | None
    progress_percent: int | None
    progress_message: str | None
    progress_updated_at: datetime | None
    generation_run_id: UUID | None
    error_message: str | None
    available_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime


class TranscriptionResponse(ApiModel):
    id: UUID
    recording_id: UUID
    language: str | None
    model_name: str
    full_text: str
    segment_count: int
    created_at: datetime
    updated_at: datetime


class RecordingSummaryResponse(ApiModel):
    id: UUID
    recording_id: UUID
    provider: str
    model_name: str
    summary_text: str
    created_at: datetime
    updated_at: datetime


class TranscriptionSegmentResponse(ApiModel):
    id: UUID
    recording_id: UUID
    transcription_id: UUID
    segment_index: int
    start_ms: int
    end_ms: int
    text: str
    speaker_label: str | None
    speaker_cluster_id: str | None
    speaker_confidence: Decimal | None
    is_target_person: bool
    target_person_confidence: Decimal | None
    diarization_segment_id: UUID | None
    matched_speaker_profile_id: UUID | None
    created_at: datetime


class TranscriptionTokenResponse(ApiModel):
    id: UUID
    recording_id: UUID
    transcription_id: UUID
    token_index: int
    source_window_index: int
    text: str
    start_ms: int
    end_ms: int
    speaker_cluster_id: str | None
    speaker_label: str | None
    attribution_status: str


class DiarizationSegmentResponse(ApiModel):
    id: UUID
    recording_id: UUID
    speaker_cluster_id: str
    speaker_label: str
    start_ms: int
    end_ms: int
    confidence: Decimal | None
    is_target_person: bool
    target_person_confidence: Decimal | None
    matched_speaker_profile_id: UUID | None
    created_at: datetime


class UtteranceSegmentResponse(ApiModel):
    id: UUID
    recording_id: UUID
    utterance_index: int
    start_ms: int
    end_ms: int
    text: str
    speaker_label: str | None
    speaker_cluster_id: str | None
    source_transcription_segment_ids: list[UUID]
    is_target_person: bool
    target_person_confidence: Decimal | None
    matched_speaker_profile_id: UUID | None
    merge_reason: str
    created_at: datetime


class CreateRecordingResponse(ApiModel):
    recording: RecordingResponse
    pipeline_run_id: UUID


class RecordingListResponse(ApiModel):
    items: list[RecordingResponse]
    total: int
    page: int
    page_size: int
    stats: dict[str, int]


class RecordingDetailResponse(ApiModel):
    recording: RecordingResponse
    summary: RecordingSummaryResponse | None
    transcription: TranscriptionResponse | None
    transcription_segments: list[TranscriptionSegmentResponse]
    transcription_tokens: list[TranscriptionTokenResponse]
    speaker_diarization_segments: list[DiarizationSegmentResponse]
    utterance_segments: list[UtteranceSegmentResponse]
    pipeline_runs: list[PipelineRunResponse]


class PipelineRunDetailResponse(ApiModel):
    run: PipelineRunResponse
    stages: list[StageRunResponse]


class RetryRecordingResponse(ApiModel):
    pipeline_run_id: UUID


class RetryRecordingStageResponse(ApiModel):
    stage_run_id: UUID


class SummaryRegenerationResponse(ApiModel):
    generation_run_id: UUID


class UpdateRecordingRequest(ApiModel):
    title: str | None = None
    location: str | None = None


class SpeakerMapping(ApiModel):
    speaker_cluster_id: str = Field(min_length=1, max_length=160)
    display_name: str = Field(min_length=1, max_length=160)
    speaker_profile_id: UUID | None = None


class UpdateSpeakerMappingsRequest(ApiModel):
    mappings: list[SpeakerMapping]


@router.get("/{recording_id}/audio")
def get_recording_audio(recording_id: UUID, service: RecordingServiceDependency, storage: StorageDependency, user: CurrentUserDependency) -> FileResponse:
    """Return audio only after the same recording-level view authorization as the detail API."""
    try:
        recording = service.get_recording_audio(user, recording_id)
    except RecordingNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found") from error
    except RecordingAccessDeniedError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Recording access denied") from error
    path = storage.resolve(str(recording["storage_path"]))
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording audio not found")
    return FileResponse(path, media_type=str(recording["mime_type"]), filename=str(recording["file_name"]))


@router.post("", response_model=CreateRecordingResponse, status_code=status.HTTP_201_CREATED)
async def create_recording(
    service: RecordingServiceDependency,
    user: CurrentUserDependency,
    audio: Annotated[UploadFile, File(description="The audio recording to process")],
    title: Annotated[str | None, Form()] = None,
    location: Annotated[str | None, Form()] = None,
) -> CreateRecordingResponse:
    """Store one audio file and enqueue its recording_processing run."""
    try:
        recording, run_id = await service.create_from_upload(user, audio, title, location)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    return CreateRecordingResponse(recording=RecordingResponse.model_validate(recording), pipeline_run_id=run_id)


@router.get("", response_model=RecordingListResponse)
def list_recordings(
    service: RecordingServiceDependency,
    user: CurrentUserDependency,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> RecordingListResponse:
    """List recordings and their current high-level processing status."""
    items, total, stats = service.list_recordings(user, status_filter, page, page_size)
    return RecordingListResponse(items=[RecordingResponse.model_validate(item) for item in items], total=total, page=page, page_size=page_size, stats=stats)


@router.get("/{recording_id}", response_model=RecordingDetailResponse)
def get_recording(recording_id: UUID, service: RecordingServiceDependency, user: CurrentUserDependency) -> RecordingDetailResponse:
    """Return the current materialized result and the history of pipeline runs."""
    try:
        detail = service.get_recording_detail(user, recording_id)
    except RecordingNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found") from error
    except RecordingAccessDeniedError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Recording access denied") from error
    return RecordingDetailResponse(
        recording=RecordingResponse.model_validate(detail["recording"]),
        summary=_optional_model(RecordingSummaryResponse, detail["summary"]),
        transcription=_optional_model(TranscriptionResponse, detail["transcription"]),
        transcription_segments=[TranscriptionSegmentResponse.model_validate(item) for item in detail["transcription_segments"]],
        transcription_tokens=[TranscriptionTokenResponse.model_validate(item) for item in detail["transcription_tokens"]],
        speaker_diarization_segments=[DiarizationSegmentResponse.model_validate(item) for item in detail["speaker_diarization_segments"]],
        utterance_segments=[UtteranceSegmentResponse.model_validate(item) for item in detail["utterance_segments"]],
        pipeline_runs=[PipelineRunResponse.model_validate(item) for item in detail["pipeline_runs"]],
    )


@router.patch("/{recording_id}", response_model=RecordingResponse)
def update_recording(
    recording_id: UUID, payload: UpdateRecordingRequest, service: RecordingServiceDependency, user: CurrentUserDependency
) -> RecordingResponse:
    """Update a recording title and/or location."""
    try:
        recording = service.update_recording(
            user,
            recording_id,
            title=payload.title if "title" in payload.model_fields_set else None,
            location=payload.location,
            update_location="location" in payload.model_fields_set,
        )
    except RecordingNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found") from error
    except RecordingAccessDeniedError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Recording access denied") from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    return RecordingResponse.model_validate(recording)


@router.patch("/{recording_id}/speaker-mappings", status_code=status.HTTP_204_NO_CONTENT)
def update_speaker_mappings(
    recording_id: UUID,
    payload: UpdateSpeakerMappingsRequest,
    service: RecordingServiceDependency,
    user: CurrentUserDependency,
) -> None:
    """Update recording-local speaker identities without rerunning the pipeline."""
    try:
        service.update_speaker_mappings(
            user,
            recording_id,
            [(mapping.speaker_cluster_id, mapping.display_name, mapping.speaker_profile_id) for mapping in payload.mappings],
        )
    except RecordingNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found") from error
    except RecordingAccessDeniedError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Recording access denied") from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@router.delete("/{recording_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_recording(recording_id: UUID, service: RecordingServiceDependency, user: CurrentUserDependency) -> None:
    """Permanently remove one recording and its associated files and results."""
    try:
        service.delete_recording(user, recording_id)
    except RecordingNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found") from error
    except RecordingAccessDeniedError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Recording access denied") from error


@router.get("/pipeline-runs/{run_id}", response_model=PipelineRunDetailResponse)
def get_pipeline_run(run_id: UUID, service: RecordingServiceDependency, user: CurrentUserDependency) -> PipelineRunDetailResponse:
    """Return the live execution state of one pipeline run and all its stages."""
    try:
        result = service.get_pipeline_run(user, run_id)
    except RecordingNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pipeline run not found") from error
    except RecordingAccessDeniedError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Recording access denied") from error
    return PipelineRunDetailResponse(
        run=PipelineRunResponse.model_validate(result["run"]),
        stages=[StageRunResponse.model_validate(item) for item in result["stages"]],
    )


@router.post("/{recording_id}/retry", response_model=RetryRecordingResponse, status_code=status.HTTP_201_CREATED)
def retry_recording(recording_id: UUID, service: RecordingServiceDependency, user: CurrentUserDependency) -> RetryRecordingResponse:
    """Create a new run for a failed recording without overwriting the failed run's history."""
    try:
        run_id = service.retry_failed_recording(user, recording_id)
    except RecordingNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording not found") from error
    except RecordingAccessDeniedError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Recording access denied") from error
    except RecordingNotRetryableError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    return RetryRecordingResponse(pipeline_run_id=run_id)


@router.post(
    "/{recording_id}/stages/embedding_indexing/retry",
    response_model=RetryRecordingStageResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def retry_recording_embedding_index(
    recording_id: UUID,
    service: RecordingServiceDependency,
    user: CurrentUserDependency,
) -> RetryRecordingStageResponse:
    """Requeue the recording's embedding node using its existing search-chunk artifact."""
    try:
        stage_run_id = service.retry_embedding_indexing(user, recording_id)
    except RecordingAccessDeniedError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Recording access denied") from error
    except RecordingStageNotRetryableError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    return RetryRecordingStageResponse(stage_run_id=stage_run_id)


@router.post("/{recording_id}/summary/regenerate", response_model=SummaryRegenerationResponse, status_code=status.HTTP_202_ACCEPTED)
async def regenerate_recording_summary(
    recording_id: UUID,
    service: RecordingSummaryRegenerationServiceDependency,
    user: CurrentUserDependency,
) -> SummaryRegenerationResponse:
    """Regenerate the summary from the recording's current corrected utterances."""
    try:
        generation_run_id = await service.regenerate(user, recording_id)
    except RecordingSummaryNotReadyError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except RecordingAccessDeniedError as error:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Recording access denied") from error
    return SummaryRegenerationResponse(generation_run_id=generation_run_id)


def _optional_model(model: type[ApiModel], value: Any) -> Any:
    return None if value is None else model.model_validate(value)
