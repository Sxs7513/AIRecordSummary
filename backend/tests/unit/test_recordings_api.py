from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from httpx import Response

from app_factory import create_app
from dependencies import get_recording_service, get_recording_summary_regeneration_service, require_current_user
from l1_foundation.settings import Settings
from l2_core.application.recordings import RecordingNotFoundError, RecordingNotRetryableError
from l2_core.auth.contracts import CurrentUser


def _settings() -> Settings:
    return Settings.model_validate(
        {
            "DB_HOST": "localhost",
            "DB_PORT": 5432,
            "DB_USER": "postgres",
            "DB_PASSWORD": "postgres",
            "DB_NAME": "ai_record_summary",
            "DB_ADMIN_DATABASE": "postgres",
            "DB_SSL": False,
            "LOCAL_STORAGE_ROOT": "uploads-test",
        }
    )


def _recording(recording_id: UUID, status: str = "processing") -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "id": recording_id,
        "title": "meeting.mp3",
        "file_name": "meeting.mp3",
        "storage_path": f"recordings/{recording_id}/audio.mp3",
        "location": None,
        "mime_type": "audio/mpeg",
        "file_size_bytes": 5,
        "duration_seconds": None,
        "status": status,
        "error_message": None,
        "uploaded_at": now,
        "created_at": now,
        "updated_at": now,
    }


class FakeRecordingService:
    def __init__(self) -> None:
        self.recording_id = uuid4()
        self.run_id = uuid4()

    async def create_from_upload(self, _user: CurrentUser, _audio: object, title: str | None, _location: str | None) -> tuple[dict[str, Any], UUID]:
        recording = _recording(self.recording_id)
        recording["title"] = title or recording["title"]
        return recording, self.run_id

    def list_recordings(self, _user: CurrentUser, _status: str | None, page: int, _page_size: int) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
        return [_recording(self.recording_id)], 1, {"uploaded": 0, "processing": 1, "completed": 0, "failed": 0}

    def get_recording_detail(self, _user: CurrentUser, recording_id: UUID) -> dict[str, Any]:
        if recording_id != self.recording_id:
            raise RecordingNotFoundError(str(recording_id))
        return {
            "recording": _recording(recording_id),
            "summary": None,
            "transcription": None,
            "transcription_segments": [],
            "transcription_tokens": [],
            "speaker_diarization_segments": [],
            "utterance_segments": [],
            "pipeline_runs": [self._pipeline_run()],
        }

    def get_pipeline_run(self, _user: CurrentUser, run_id: UUID) -> dict[str, Any]:
        if run_id != self.run_id:
            raise RecordingNotFoundError(str(run_id))
        return {"run": self._pipeline_run(), "stages": []}

    def _pipeline_run(self) -> dict[str, Any]:
        now = datetime.now(UTC)
        return {
            "id": self.run_id,
            "recording_id": self.recording_id,
            "pipeline_name": "recording_processing",
            "pipeline_version": "4",
            "status": "queued",
            "started_at": None,
            "finished_at": None,
            "error_message": None,
            "created_at": now,
            "updated_at": now,
        }

    def retry_failed_recording(self, _user: CurrentUser, recording_id: UUID) -> UUID:
        if recording_id != self.recording_id:
            raise RecordingNotFoundError(str(recording_id))
        return self.run_id

    def retry_embedding_indexing(self, _user: CurrentUser, recording_id: UUID) -> UUID:
        if recording_id != self.recording_id:
            raise RecordingNotFoundError(str(recording_id))
        return self.run_id

    def update_recording(
        self, _user: CurrentUser, recording_id: UUID, *, title: str | None = None, location: str | None = None, update_location: bool = False
    ) -> dict[str, Any]:
        if recording_id != self.recording_id:
            raise RecordingNotFoundError(str(recording_id))
        recording = _recording(recording_id)
        if title is not None:
            recording["title"] = title
        if update_location:
            recording["location"] = location
        return recording

    def update_speaker_mappings(
        self,
        _user: CurrentUser,
        recording_id: UUID,
        _mappings: list[tuple[str, str, UUID | None]],
    ) -> None:
        if recording_id != self.recording_id:
            raise RecordingNotFoundError(str(recording_id))

    def delete_recording(self, _user: CurrentUser, recording_id: UUID) -> None:
        if recording_id != self.recording_id:
            raise RecordingNotFoundError(str(recording_id))


class FakeSummaryRegenerationService:
    def __init__(self, recording_id: UUID, generation_run_id: UUID) -> None:
        self._recording_id = recording_id
        self._generation_run_id = generation_run_id

    async def regenerate(self, _user: CurrentUser, recording_id: UUID) -> UUID:
        if recording_id != self._recording_id:
            raise RecordingNotFoundError(str(recording_id))
        return self._generation_run_id


def _client(service: FakeRecordingService) -> TestClient:
    app = create_app(_settings(), start_pipeline_worker=False)
    app.dependency_overrides[get_recording_service] = lambda: service
    app.dependency_overrides[get_recording_summary_regeneration_service] = lambda: FakeSummaryRegenerationService(service.recording_id, service.run_id)
    app.dependency_overrides[require_current_user] = lambda: CurrentUser(uuid4(), "test@example.com", "Test", uuid4(), ())
    return TestClient(app)


def test_create_recording_enqueues_a_pipeline_run() -> None:
    service = FakeRecordingService()
    with _client(service) as client:
        response = cast(
            Response,
            cast(Any, client).post(
                "/api/recordings",
                data={"title": "weekly sync"},
                files={"audio": ("meeting.mp3", b"audio", "audio/mpeg")},
            ),
        )

    assert response.status_code == 201
    assert response.json()["recording"]["title"] == "weekly sync"
    assert response.json()["pipeline_run_id"] == str(service.run_id)


def test_recording_read_endpoints_return_current_results_and_execution_state() -> None:
    service = FakeRecordingService()
    with _client(service) as client:
        client_any = cast(Any, client)
        listing = cast(Response, client_any.get("/api/recordings?page=1&page_size=10"))
        detail = cast(Response, client_any.get(f"/api/recordings/{service.recording_id}"))
        run = cast(Response, client_any.get(f"/api/recordings/pipeline-runs/{service.run_id}"))

    assert listing.status_code == 200
    assert listing.json()["total"] == 1
    assert detail.status_code == 200
    assert detail.json()["recording"]["id"] == str(service.recording_id)
    assert detail.json()["pipeline_runs"][0]["recording_id"] == str(service.recording_id)
    assert run.status_code == 200
    assert run.json()["run"]["status"] == "queued"


def test_retry_failure_is_reported_as_a_conflict() -> None:
    service = FakeRecordingService()
    service.retry_failed_recording = lambda _user, _recording_id: (_ for _ in ()).throw(RecordingNotRetryableError("not failed"))  # type: ignore[method-assign]
    with _client(service) as client:
        response = cast(Response, cast(Any, client).post(f"/api/recordings/{service.recording_id}/retry"))

    assert response.status_code == 409


def test_summary_regeneration_starts_a_generation_stream() -> None:
    service = FakeRecordingService()
    with _client(service) as client:
        response = cast(Response, cast(Any, client).post(f"/api/recordings/{service.recording_id}/summary/regenerate"))

    assert response.status_code == 202
    assert response.json() == {"generation_run_id": str(service.run_id)}


def test_embedding_indexing_retry_requeues_only_that_stage() -> None:
    service = FakeRecordingService()
    with _client(service) as client:
        response = cast(
            Response,
            cast(Any, client).post(f"/api/recordings/{service.recording_id}/stages/embedding_indexing/retry"),
        )

    assert response.status_code == 202
    assert response.json() == {"stage_run_id": str(service.run_id)}


def test_recording_mutation_endpoints_use_the_python_service() -> None:
    service = FakeRecordingService()
    with _client(service) as client:
        client_any = cast(Any, client)
        metadata = cast(
            Response,
            client_any.patch(f"/api/recordings/{service.recording_id}", json={"title": "renamed", "location": "会议室 A"}),
        )
        labels = cast(
            Response,
            client_any.patch(
                f"/api/recordings/{service.recording_id}/speaker-mappings",
                json={"mappings": [{"speaker_cluster_id": "SPEAKER_00", "display_name": "Alice"}]},
            ),
        )
        deleted = cast(Response, client_any.delete(f"/api/recordings/{service.recording_id}"))

    assert metadata.status_code == 200
    assert metadata.json()["title"] == "renamed"
    assert metadata.json()["location"] == "会议室 A"
    assert labels.status_code == 204
    assert deleted.status_code == 204
