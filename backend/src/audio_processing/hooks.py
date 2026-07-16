from __future__ import annotations

from uuid import UUID

from sqlalchemy import Engine, text

from audio_processing.contracts import RecordingId
from audio_processing.projections import RecordingProjectionService


class RecordingProcessingHooks:
    """Recording-domain reactions to generic pipeline lifecycle events."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._projections = RecordingProjectionService(engine)

    def stage_succeeded(self, subject_id: UUID, stage_name: str, output: object) -> None:
        self._projections.project(RecordingId(subject_id), stage_name, output)

    def run_state_changed(self, subject_id: UUID, status: str, error_message: str | None) -> None:
        if status not in {"succeeded", "partial_failed", "failed", "cancelled"}:
            return
        recording_status = "completed" if status in {"succeeded", "partial_failed"} else "failed"
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    update recordings set status = :status, error_message = :error_message, updated_at = now()
                    where id = :recording_id
                    """
                ),
                {"recording_id": subject_id, "status": recording_status, "error_message": error_message},
            )
