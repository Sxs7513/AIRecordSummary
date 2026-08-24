from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4, uuid5

from fastapi import UploadFile
from sqlalchemy import Connection, Engine, RowMapping, text

from l1_foundation.infrastructure.storage.local import LocalStorage
from l1_foundation.pipeline.contracts import ArtifactRef, PipelineRunId, StageRunId
from l1_foundation.pipeline.definitions.graph import PipelineDefinition
from l1_foundation.streaming import SyncRedisStreamStore
from l2_core.access.recordings import RecordingAccessService
from l2_core.application.processing_queue import ProcessingCommandPublisher, queued_processing_state, stable_recording_processing_id
from l2_core.audio_processing.contracts import RecordingId
from l2_core.audio_processing.definition import recording_processing
from l2_core.auth.contracts import CurrentUser

type DatabaseRow = dict[str, Any]
logger = logging.getLogger("audio_processing")


class RecordingNotFoundError(LookupError):
    """Raised when a requested recording does not exist."""


class RecordingNotRetryableError(ValueError):
    """Raised when a recording is not currently in a failed state."""


class RecordingStageNotRetryableError(ValueError):
    """Raised when a requested recording stage cannot be requeued."""


class RecordingService:
    """HTTP-facing use cases for recording creation, read models, and reruns."""

    def __init__(
        self,
        engine: Engine,
        storage: LocalStorage,
        processing_definition: PipelineDefinition = recording_processing,
        processing_publisher: ProcessingCommandPublisher | None = None,
        processing_state_store: SyncRedisStreamStore | None = None,
    ) -> None:
        self._engine = engine
        self._storage = storage
        self._access = RecordingAccessService(engine)
        self._processing_definition = processing_definition
        self._processing_publisher = processing_publisher
        self._processing_state_store = processing_state_store

    async def create_from_upload(self, user: CurrentUser, upload: UploadFile, title: str | None, location: str | None) -> tuple[DatabaseRow, PipelineRunId]:
        """Persist an uploaded file, create its recording, then enqueue the declared pipeline."""
        file_name = self._safe_file_name(upload.filename)
        recording_id = RecordingId(uuid4())
        storage_path = self._storage_key(recording_id, file_name)
        destination = self._storage.resolve(storage_path)
        file_size_bytes, content_md5 = await self._write_upload(upload, destination)
        processing_id = stable_recording_processing_id(
            user.current_workspace_id,
            user.id,
            self._processing_definition.name,
            self._processing_definition.version,
            content_md5,
        )
        recording_created = False
        try:
            with self._engine.begin() as connection:
                recording, recording_created = self._insert_recording(
                    connection=connection,
                    recording_id=recording_id,
                    workspace_id=user.current_workspace_id,
                    owner_user_id=user.id,
                    title=title.strip() if title and title.strip() else file_name,
                    file_name=file_name,
                    storage_path=storage_path,
                    location=location.strip() if location and location.strip() else None,
                    mime_type=upload.content_type or "application/octet-stream",
                    file_size_bytes=file_size_bytes,
                    content_md5=content_md5,
                    processing_id=processing_id,
                    pipeline_name=self._processing_definition.name,
                    pipeline_version=self._processing_definition.version,
                )
                if recording_created:
                    if self._processing_publisher is None:
                        raise RuntimeError("Processing outbox publisher is unavailable")
                    self._processing_publisher.enqueue_recording(
                        connection,
                        recording_id,
                        self._processing_definition.name,
                        self._processing_definition.version,
                        self._source_audio_artifact(recording),
                        processing_id=processing_id,
                        workspace_id=user.current_workspace_id,
                    )
            if not recording_created:
                self._storage.remove_tree(f"recordings/{recording_id}")
                return recording, PipelineRunId(processing_id)
            run_id = PipelineRunId(processing_id)
            self._remember_processing(recording_id, run_id)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()
        return recording, run_id

    def _remember_processing(self, recording_id: UUID, processing_id: UUID) -> None:
        if self._processing_state_store is None:
            return
        self._processing_state_store.set_state_if_absent(
            f"processing:{processing_id}:state",
            queued_processing_state(
                processing_id,
                recording_id,
                self._processing_definition.name,
                self._processing_definition.version,
            ),
        )
        self._processing_state_store.set_state(
            f"recording:{recording_id}:processing",
            {"processing_id": str(processing_id)},
        )

    def list_recordings(self, user: CurrentUser, status: str | None, page: int, page_size: int) -> tuple[list[DatabaseRow], int, dict[str, int]]:
        normalized_status = status if status and status != "all" else None
        where = f"where ({self._access.accessible_predicate()})"
        params: dict[str, Any] = {"limit": page_size, "offset": (page - 1) * page_size, "current_user_id": user.id}
        if normalized_status:
            where += " and status = :status"
            params["status"] = normalized_status
        with self._engine.connect() as connection:
            total = cast(int, connection.execute(text(f"select count(*) from recordings {where}"), params).scalar_one())
            rows = [
                dict(row)
                for row in connection.execute(text(f"select * from recordings {where} order by created_at desc limit :limit offset :offset"), params)
                .mappings()
                .all()
            ]
            stats_rows = connection.execute(text(f"select status, count(*) as count from recordings {where} group by status"), params).mappings().all()
        stats = {"uploaded": 0, "processing": 0, "completed": 0, "failed": 0}
        for row in stats_rows:
            stats[cast(str, row["status"])] = cast(int, row["count"])
        return rows, total, stats

    def get_recording_detail(self, user: CurrentUser, recording_id: UUID) -> dict[str, Any]:
        self._access.require_view(recording_id, user)
        with self._engine.connect() as connection:
            recording_row = (
                connection.execute(text("select * from recordings where id = :recording_id"), {"recording_id": recording_id}).mappings().one_or_none()
            )
            recording = None if recording_row is None else dict(recording_row)
            if recording is None:
                raise RecordingNotFoundError(str(recording_id))
            return {
                "recording": recording,
                "summary": self._optional_row(
                    connection.execute(text("select * from recording_summaries where recording_id = :recording_id"), {"recording_id": recording_id})
                    .mappings()
                    .one_or_none()
                ),
                "transcription": self._optional_row(
                    connection.execute(text("select * from transcriptions where recording_id = :recording_id"), {"recording_id": recording_id})
                    .mappings()
                    .one_or_none()
                ),
                "transcription_segments": [
                    dict(row)
                    for row in connection.execute(
                        text(
                            """
                            select segments.*,
                                   coalesce(profiles.display_name, mappings.display_name, segments.speaker_label) as speaker_label,
                                   coalesce(mappings.speaker_profile_id, segments.matched_speaker_profile_id) as matched_speaker_profile_id,
                                   (mappings.speaker_profile_id is not null or segments.is_target_person) as is_target_person
                            from transcription_segments segments
                            left join recording_speaker_mappings mappings
                              on mappings.recording_id = segments.recording_id
                             and mappings.speaker_cluster_id = segments.speaker_cluster_id
                            left join speaker_profiles profiles on profiles.id = mappings.speaker_profile_id
                            where segments.recording_id = :recording_id
                            order by segments.segment_index
                            """
                        ),
                        {"recording_id": recording_id},
                    )
                    .mappings()
                    .all()
                ],
                "transcription_tokens": [
                    dict(row)
                    for row in connection.execute(
                        text(
                            """
                            select tokens.*,
                                   coalesce(profiles.display_name, mappings.display_name, tokens.speaker_label) as speaker_label
                            from transcription_tokens tokens
                            left join recording_speaker_mappings mappings
                              on mappings.recording_id = tokens.recording_id
                             and mappings.speaker_cluster_id = tokens.speaker_cluster_id
                            left join speaker_profiles profiles on profiles.id = mappings.speaker_profile_id
                            where tokens.recording_id = :recording_id
                            order by tokens.token_index
                            """
                        ),
                        {"recording_id": recording_id},
                    )
                    .mappings()
                    .all()
                ],
                "speaker_diarization_segments": [
                    dict(row)
                    for row in connection.execute(
                        text(
                            """
                            select segments.*,
                                   coalesce(profiles.display_name, mappings.display_name, segments.speaker_label) as speaker_label,
                                   coalesce(mappings.speaker_profile_id, segments.matched_speaker_profile_id) as matched_speaker_profile_id,
                                   (mappings.speaker_profile_id is not null or segments.is_target_person) as is_target_person
                            from speaker_diarization_segments segments
                            left join recording_speaker_mappings mappings
                              on mappings.recording_id = segments.recording_id
                             and mappings.speaker_cluster_id = segments.speaker_cluster_id
                            left join speaker_profiles profiles on profiles.id = mappings.speaker_profile_id
                            where segments.recording_id = :recording_id
                            order by segments.start_ms, segments.end_ms
                            """
                        ),
                        {"recording_id": recording_id},
                    )
                    .mappings()
                    .all()
                ],
                "utterance_segments": [
                    dict(row)
                    for row in connection.execute(
                        text(
                            """
                            select segments.*,
                                   coalesce(profiles.display_name, mappings.display_name, segments.speaker_label) as speaker_label,
                                   coalesce(mappings.speaker_profile_id, segments.matched_speaker_profile_id) as matched_speaker_profile_id,
                                   (mappings.speaker_profile_id is not null or segments.is_target_person) as is_target_person
                            from utterance_segments segments
                            left join recording_speaker_mappings mappings
                              on mappings.recording_id = segments.recording_id
                             and mappings.speaker_cluster_id = segments.speaker_cluster_id
                            left join speaker_profiles profiles on profiles.id = mappings.speaker_profile_id
                            where segments.recording_id = :recording_id
                            order by segments.utterance_index
                            """
                        ),
                        {"recording_id": recording_id},
                    )
                    .mappings()
                    .all()
                ],
                "pipeline_runs": self._runtime_pipeline_runs(recording),
            }

    def get_recording_audio(self, user: CurrentUser, recording_id: UUID) -> DatabaseRow:
        """Resolve file metadata after the same view authorization without loading the full read model."""
        self._access.require_view(recording_id, user)
        with self._engine.connect() as connection:
            row = (
                connection.execute(text("select storage_path, mime_type, file_name from recordings where id = :recording_id"), {"recording_id": recording_id})
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise RecordingNotFoundError(str(recording_id))
        return dict(row)

    def get_pipeline_run(self, user: CurrentUser, run_id: UUID) -> dict[str, Any]:
        if self._processing_state_store is None:
            raise RecordingNotFoundError(str(run_id))
        state = self._processing_state_store.get_state(f"processing:{run_id}:state")
        if state is None:
            raise RecordingNotFoundError(str(run_id))
        recording_id = UUID(str(state["subject_id"]))
        self._access.require_view(recording_id, user)
        return {"run": self._runtime_run(state), "stages": self._runtime_stages(state)}

    def _runtime_pipeline_runs(self, recording: DatabaseRow) -> list[DatabaseRow]:
        if self._processing_state_store is not None:
            mapping = self._processing_state_store.get_state(f"recording:{recording['id']}:processing")
            if mapping is not None and mapping.get("processing_id") is not None:
                state = self._processing_state_store.get_state(f"processing:{mapping['processing_id']}:state")
                if state is not None:
                    return [self._runtime_run(state)]
        return [self._database_pipeline_run(recording)]

    @staticmethod
    def _database_pipeline_run(recording: DatabaseRow) -> DatabaseRow:
        status = {
            "uploaded": "queued",
            "processing": "running",
            "completed": "succeeded",
            "failed": "failed",
        }[cast(str, recording["status"])]
        terminal = status in {"succeeded", "failed"}
        return {
            "id": UUID(str(recording["processing_id"])),
            "recording_id": UUID(str(recording["id"])),
            "pipeline_name": recording["processing_pipeline_name"],
            "pipeline_version": recording["processing_pipeline_version"],
            "status": status,
            "started_at": None if status == "queued" else recording["created_at"],
            "finished_at": recording["updated_at"] if terminal else None,
            "error_message": recording.get("error_message"),
            "created_at": recording["created_at"],
            "updated_at": recording["updated_at"],
        }

    @staticmethod
    def _runtime_run(state: dict[str, Any]) -> DatabaseRow:
        started = state.get("started_at") or state["updated_at"]
        return {
            "id": UUID(str(state["processing_id"])),
            "recording_id": UUID(str(state["subject_id"])),
            "pipeline_name": state["pipeline_name"],
            "pipeline_version": state["pipeline_version"],
            "status": state["status"],
            "started_at": state.get("started_at"),
            "finished_at": state.get("finished_at"),
            "error_message": state.get("error_message"),
            "created_at": started,
            "updated_at": state["updated_at"],
        }

    def _runtime_stages(self, state: dict[str, Any]) -> list[DatabaseRow]:
        raw_stages = cast(dict[str, dict[str, Any]], state.get("stages", {}))
        now = state["updated_at"]
        processing_id = UUID(str(state["processing_id"]))
        recording_id = UUID(str(state["subject_id"]))
        rows: list[DatabaseRow] = []
        for node in self._processing_definition.topologically_sorted_nodes():
            stage = raw_stages.get(node.name, {})
            rows.append(
                {
                    "id": uuid5(processing_id, node.name),
                    "pipeline_run_id": processing_id,
                    "recording_id": recording_id,
                    "node_name": node.name,
                    "stage_name": node.stage_name,
                    "stage_version": node.stage_version,
                    "required": node.required,
                    "status": stage.get("status", "pending"),
                    "attempt_count": stage.get("attempt", 0),
                    "max_attempts": node.retry_policy.max_attempts,
                    "progress_percent": stage.get("progress_percent"),
                    "progress_message": stage.get("progress_message"),
                    "progress_updated_at": stage.get("progress_updated_at"),
                    "generation_run_id": None,
                    "error_message": stage.get("error"),
                    "available_at": now,
                    "started_at": None,
                    "finished_at": now if stage.get("status") in {"succeeded", "failed"} else None,
                    "created_at": state.get("started_at") or now,
                    "updated_at": now,
                }
            )
        return rows

    async def retry_failed_recording(self, user: CurrentUser, recording_id: UUID) -> PipelineRunId:
        self._access.require_edit(recording_id, user)
        processing_id = uuid4()
        with self._engine.begin() as connection:
            row = (
                connection.execute(
                    text("select status, storage_path, file_name, mime_type from recordings where id = :recording_id for update"),
                    {"recording_id": recording_id},
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                raise RecordingNotFoundError(str(recording_id))
            status = cast(str, row["status"])
            if status != "failed":
                raise RecordingNotRetryableError(f"Recording {recording_id} has status {status!r}, not 'failed'")
            if self._processing_publisher is None:
                raise RuntimeError("Processing outbox publisher is unavailable")
            connection.execute(
                text("update recordings set status = 'processing', error_message = null, updated_at = now() where id = :recording_id"),
                {"recording_id": recording_id},
            )
            self._processing_publisher.enqueue_recording(
                connection,
                recording_id,
                self._processing_definition.name,
                self._processing_definition.version,
                self._source_audio_artifact(dict(row)),
                processing_id=processing_id,
                workspace_id=user.current_workspace_id,
            )
        self._remember_processing(recording_id, processing_id)
        return PipelineRunId(processing_id)

    async def retry_embedding_indexing(self, user: CurrentUser, recording_id: UUID) -> StageRunId:
        """Re-run only embedding indexing from the current run's successful search-chunk artifact."""
        self._access.require_edit(recording_id, user)
        if self._processing_publisher is None:
            raise RecordingStageNotRetryableError("Processing Kafka publisher is unavailable")
        if self._processing_state_store is None:
            raise RecordingStageNotRetryableError("Processing state is unavailable")
        mapping = self._processing_state_store.get_state(f"recording:{recording_id}:processing")
        if mapping is None or mapping.get("processing_id") is None:
            raise RecordingStageNotRetryableError("Recording has no reusable processing run")
        processing_id = UUID(str(mapping["processing_id"]))
        state = self._processing_state_store.get_state(f"processing:{processing_id}:state")
        if state is None:
            raise RecordingStageNotRetryableError("Processing state has expired; search chunks cannot be reused")
        stages = cast(dict[str, dict[str, Any]], state.get("stages", {}))
        chunk_stage = stages.get("build_search_chunks")
        if chunk_stage is None or chunk_stage.get("status") != "succeeded":
            raise RecordingStageNotRetryableError("Search chunks were not generated successfully")
        chunk_ref = next(
            (
                ArtifactRef.model_validate(raw)
                for raw in cast(list[dict[str, Any]], chunk_stage.get("artifacts", []))
                if raw.get("artifact_type") == "search.chunks"
            ),
            None,
        )
        if chunk_ref is None or not self._storage.resolve(chunk_ref.uri).is_file():
            raise RecordingStageNotRetryableError("Search chunk artifact is no longer available")
        with self._engine.begin() as connection:
            self._processing_publisher.enqueue_embedding_retry(connection, processing_id, recording_id, chunk_ref)
        return StageRunId(uuid5(processing_id, "embedding_indexing"))

    def update_recording(
        self,
        user: CurrentUser,
        recording_id: UUID,
        *,
        title: str | None = None,
        location: str | None = None,
        update_location: bool = False,
    ) -> DatabaseRow:
        """Update user-editable recording metadata while retaining the processing result."""
        self._access.require_edit(recording_id, user)
        if title is not None and not title.strip():
            raise ValueError("Title is required")
        assignments: list[str] = []
        params: dict[str, Any] = {"recording_id": recording_id}
        if title is not None:
            assignments.append("title = :title")
            params["title"] = title.strip()
        if update_location:
            assignments.append("location = :location")
            params["location"] = location.strip() if location and location.strip() else None
        if not assignments:
            raise ValueError("At least one field must be provided")
        assignments.append("updated_at = now()")
        with self._engine.begin() as connection:
            row = (
                connection.execute(text(f"update recordings set {', '.join(assignments)} where id = :recording_id returning *"), params)
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise RecordingNotFoundError(str(recording_id))
        return dict(row)

    def update_speaker_mappings(
        self,
        user: CurrentUser,
        recording_id: UUID,
        mappings: list[tuple[str, str, UUID | None]],
    ) -> None:
        """Upsert recording-local cluster identities without rewriting pipeline output."""
        self._access.require_edit(recording_id, user)
        cleaned = [
            (cluster_id.strip(), display_name.strip(), profile_id)
            for cluster_id, display_name, profile_id in mappings
            if cluster_id.strip() and display_name.strip()
        ]
        with self._engine.begin() as connection:
            exists = connection.execute(
                text("select 1 from recordings where id = :recording_id for update"), {"recording_id": recording_id}
            ).scalar_one_or_none()
            if exists is None:
                raise RecordingNotFoundError(str(recording_id))
            known_clusters = {
                str(value)
                for value in connection.execute(
                    text(
                        """
                        select speaker_cluster_id from speaker_diarization_segments where recording_id = :recording_id
                        union
                        select speaker_cluster_id from transcription_segments
                         where recording_id = :recording_id and speaker_cluster_id is not null
                        union
                        select speaker_cluster_id from utterance_segments
                         where recording_id = :recording_id and speaker_cluster_id is not null
                        """
                    ),
                    {"recording_id": recording_id},
                ).scalars()
            }
            unknown = sorted({cluster_id for cluster_id, _display_name, _profile_id in cleaned} - known_clusters)
            if unknown:
                raise ValueError(f"Unknown speaker cluster: {', '.join(unknown)}")
            for cluster_id, display_name, profile_id in cleaned:
                connection.execute(
                    text(
                        """
                        insert into recording_speaker_mappings (
                            recording_id, speaker_cluster_id, display_name, speaker_profile_id
                        ) values (
                            :recording_id, :speaker_cluster_id, :display_name, :speaker_profile_id
                        )
                        on conflict (recording_id, speaker_cluster_id) do update set
                            display_name = excluded.display_name,
                            speaker_profile_id = excluded.speaker_profile_id,
                            updated_at = now()
                        """
                    ),
                    {
                        "recording_id": recording_id,
                        "speaker_cluster_id": cluster_id,
                        "display_name": display_name,
                        "speaker_profile_id": profile_id,
                    },
                )
            connection.execute(text("update recordings set updated_at = now() where id = :recording_id"), {"recording_id": recording_id})

    async def delete_recording(self, user: CurrentUser, recording_id: UUID) -> None:
        """Delete a recording, its database results, and all recording-owned storage trees."""
        self._access.require_edit(recording_id, user)
        with self._engine.begin() as connection:
            if self._processing_publisher is None:
                raise RuntimeError("Processing outbox publisher is unavailable")
            self._processing_publisher.enqueue_cancel(connection, recording_id)
            exists = connection.execute(
                text("delete from recordings where id = :recording_id returning id"), {"recording_id": recording_id}
            ).scalar_one_or_none()
        if exists is None:
            raise RecordingNotFoundError(str(recording_id))
        self._storage.remove_tree(f"recordings/{recording_id}")
        self._storage.remove_tree(f"normalized/{recording_id}")
        self._storage.remove_tree(f"artifacts/{recording_id}")

    def _insert_recording(
        self,
        connection: Connection,
        recording_id: RecordingId,
        workspace_id: UUID,
        owner_user_id: UUID,
        title: str,
        file_name: str,
        storage_path: str,
        location: str | None,
        mime_type: str,
        file_size_bytes: int,
        content_md5: str,
        processing_id: UUID,
        pipeline_name: str,
        pipeline_version: str,
    ) -> tuple[DatabaseRow, bool]:
        inserted = (
            connection.execute(
                text(
                    """
                    insert into recordings (
                        id, workspace_id, owner_user_id, title, file_name, storage_path, location,
                        mime_type, file_size_bytes, content_md5, processing_id,
                        processing_pipeline_name, processing_pipeline_version, status
                    ) values (
                        :id, :workspace_id, :owner_user_id, :title, :file_name, :storage_path, :location,
                        :mime_type, :file_size_bytes, :content_md5, :processing_id,
                        :pipeline_name, :pipeline_version, 'processing'
                    )
                    on conflict (processing_id) do nothing
                    returning *
                    """
                ),
                {
                    "id": recording_id,
                    "workspace_id": workspace_id,
                    "owner_user_id": owner_user_id,
                    "title": title,
                    "file_name": file_name,
                    "storage_path": storage_path,
                    "location": location,
                    "mime_type": mime_type,
                    "file_size_bytes": file_size_bytes,
                    "content_md5": content_md5,
                    "processing_id": processing_id,
                    "pipeline_name": pipeline_name,
                    "pipeline_version": pipeline_version,
                },
            )
            .mappings()
            .one_or_none()
        )
        if inserted is not None:
            return dict(inserted), True
        existing = (
            connection.execute(
                text("select * from recordings where processing_id = :processing_id"),
                {"processing_id": processing_id},
            )
            .mappings()
            .one()
        )
        return dict(existing), False

    @staticmethod
    def _source_audio_artifact(recording: DatabaseRow) -> ArtifactRef:
        return ArtifactRef(
            artifact_type="audio.source",
            artifact_version="1",
            uri=str(recording["storage_path"]),
            metadata={"file_name": str(recording["file_name"]), "mime_type": str(recording["mime_type"])},
        )

    @staticmethod
    def _safe_file_name(file_name: str | None) -> str:
        name = Path(file_name or "recording").name
        if name in {"", ".", ".."}:
            return "recording"
        return name

    @staticmethod
    def _storage_key(recording_id: RecordingId, file_name: str) -> str:
        suffix = Path(file_name).suffix.lower()
        return f"recordings/{recording_id}/{uuid4().hex}{suffix}"

    @staticmethod
    async def _write_upload(upload: UploadFile, destination: Path) -> tuple[int, str]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        digest = hashlib.md5(usedforsecurity=False)
        with destination.open("wb") as file:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                digest.update(chunk)
                file.write(chunk)
        if total == 0:
            destination.unlink(missing_ok=True)
            raise ValueError("Uploaded audio file is empty")
        return total, digest.hexdigest()

    @staticmethod
    def _optional_row(row: RowMapping | None) -> DatabaseRow | None:
        return None if row is None else dict(row)

    @staticmethod
    def order_stage_rows(pipeline_name: str, stages: list[DatabaseRow], definition: PipelineDefinition = recording_processing) -> list[DatabaseRow]:
        """Return stage read models in the graph's declared topological display order."""
        if pipeline_name != definition.name:
            return stages
        node_order = {node.name: index for index, node in enumerate(definition.topologically_sorted_nodes())}
        selected_asr_name = "transcribe_funasr_nano" if "transcribe_funasr_nano" in node_order else "transcribe_qwen_asr"
        asr_index = node_order[selected_asr_name]
        node_order.setdefault("transcribe_qwen_asr", asr_index)
        node_order.setdefault("transcribe_funasr_nano", asr_index)
        return sorted(stages, key=lambda stage: (node_order.get(cast(str, stage["node_name"]), len(node_order)), cast(str, stage["node_name"])))
