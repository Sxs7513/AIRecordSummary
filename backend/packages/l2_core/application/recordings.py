from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

from fastapi import UploadFile
from sqlalchemy import Engine, RowMapping, text

from l1_foundation.infrastructure.storage.local import LocalStorage
from l1_foundation.pipeline.contracts import ArtifactPayload, ArtifactRef, PipelineRunId, StageRunId
from l1_foundation.pipeline.definitions.graph import PipelineDefinition
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore
from l1_foundation.pipeline.runtime.repository import PipelineRepository
from l2_core.access.recordings import RecordingAccessService
from l2_core.application.recording_processing import StartRecordingProcessing
from l2_core.audio_processing.contracts import RecordingId
from l2_core.audio_processing.definition import recording_processing
from l2_core.audio_processing.stages.recording_models import SearchChunksOutput
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

    def __init__(self, engine: Engine, storage: LocalStorage, processing_definition: PipelineDefinition = recording_processing) -> None:
        self._engine = engine
        self._storage = storage
        self._pipeline_repository = PipelineRepository(engine)
        self._access = RecordingAccessService(engine)
        self._processing_definition = processing_definition

    async def create_from_upload(self, user: CurrentUser, upload: UploadFile, title: str | None, location: str | None) -> tuple[DatabaseRow, PipelineRunId]:
        """Persist an uploaded file, create its recording, then enqueue the declared pipeline."""
        file_name = self._safe_file_name(upload.filename)
        recording_id = RecordingId(uuid4())
        storage_path = self._storage_key(recording_id, file_name)
        destination = self._storage.resolve(storage_path)
        file_size_bytes = await self._write_upload(upload, destination)
        try:
            recording = self._insert_recording(
                recording_id=recording_id,
                workspace_id=user.current_workspace_id,
                owner_user_id=user.id,
                title=title.strip() if title and title.strip() else file_name,
                file_name=file_name,
                storage_path=storage_path,
                location=location.strip() if location and location.strip() else None,
                mime_type=upload.content_type or "application/octet-stream",
                file_size_bytes=file_size_bytes,
            )
            run_id = StartRecordingProcessing(self._pipeline_repository, self._processing_definition).execute(
                recording_id, self._source_audio_artifact(recording)
            )
        except Exception:
            self._delete_recording(recording_id)
            destination.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()
        return recording, run_id

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
                "pipeline_runs": self._pipeline_runs(connection, recording_id),
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
        with self._engine.connect() as connection:
            run_row = connection.execute(text("select * from pipeline_runs where id = :run_id"), {"run_id": run_id}).mappings().one_or_none()
            if run_row is None:
                raise RecordingNotFoundError(str(run_id))
            run = dict(run_row)
            if run["subject_type"] != "recording":
                raise RecordingNotFoundError(str(run_id))
            self._access.require_view(UUID(str(run["subject_id"])), user)
            stages = self._pipeline_stage_rows(connection, run_id)
        return {
            "run": self._as_recording_pipeline_run(run),
            "stages": self.order_stage_rows(cast(str, run["pipeline_name"]), stages, self._processing_definition),
        }

    @staticmethod
    def _pipeline_stage_rows(connection: Any, run_id: UUID) -> list[DatabaseRow]:
        """Keep recording details readable while an existing database awaits a schema reset."""
        generation_table_exists = connection.execute(text("select to_regclass('public.generation_runs')")).scalar_one() is not None
        generation_column = (
            """
            , (
                select generation_runs.id from generation_runs
                where generation_runs.parent_type = 'stage_run' and generation_runs.parent_id = stage_runs.id::text
                order by generation_runs.created_at desc limit 1
            ) as generation_run_id
            """
            if generation_table_exists
            else ", null::uuid as generation_run_id"
        )
        return [
            dict(row)
            for row in connection.execute(
                text(
                    f"""
                    select stage_runs.*, stage_runs.subject_id as recording_id {generation_column}
                    from stage_runs where pipeline_run_id = :run_id
                    order by created_at, id
                    """
                ),
                {"run_id": run_id},
            )
            .mappings()
            .all()
        ]

    def retry_failed_recording(self, user: CurrentUser, recording_id: UUID) -> PipelineRunId:
        self._access.require_edit(recording_id, user)
        with self._engine.connect() as connection:
            row = (
                connection.execute(
                    text("select status, storage_path, file_name, mime_type from recordings where id = :recording_id"), {"recording_id": recording_id}
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise RecordingNotFoundError(str(recording_id))
        status = cast(str, row["status"])
        if status != "failed":
            raise RecordingNotRetryableError(f"Recording {recording_id} has status {status!r}, not 'failed'")
        return StartRecordingProcessing(self._pipeline_repository, self._processing_definition).execute(
            RecordingId(recording_id), self._source_audio_artifact(dict(row))
        )

    def retry_embedding_indexing(self, user: CurrentUser, recording_id: UUID) -> StageRunId:
        """Requeue only the embedding node; the pipeline coordinator performs the execution."""
        self._access.require_edit(recording_id, user)
        stage_run_id, stage_status = self._restore_embedding_retry_input(recording_id)
        if stage_status == "retry_waiting":
            self._pipeline_repository.resume_retry_stage(stage_run_id)
            return stage_run_id
        if stage_status in {"pending", "running"}:
            return stage_run_id
        try:
            return self._pipeline_repository.requeue_stage("recording", recording_id, "embedding_indexing")
        except (LookupError, ValueError) as error:
            raise RecordingStageNotRetryableError(str(error)) from error

    def _restore_embedding_retry_input(self, recording_id: UUID) -> tuple[StageRunId, str]:
        """Restore only search.chunks from its durable stage output when startup cleanup removed the file."""
        with self._engine.connect() as connection:
            row = (
                connection.execute(
                    text(
                        """
                        select embedding_stage.id as embedding_stage_run_id,
                               embedding_stage.status as embedding_stage_status,
                               producer.id as producer_stage_run_id,
                               producer.pipeline_run_id,
                               producer.output_payload,
                               artifacts.uri,
                               artifacts.artifact_version
                        from stage_runs embedding_stage
                        join pipeline_runs on pipeline_runs.id = embedding_stage.pipeline_run_id
                        join stage_run_dependencies dependencies on dependencies.stage_run_id = embedding_stage.id
                        join stage_runs producer on producer.id = dependencies.depends_on_stage_run_id
                        join artifacts on artifacts.stage_run_id = producer.id
                        where pipeline_runs.subject_type = 'recording'
                          and pipeline_runs.subject_id = :recording_id
                          and embedding_stage.node_name = 'embedding_indexing'
                          and producer.node_name = 'build_search_chunks'
                          and artifacts.artifact_type = 'search.chunks'
                        order by pipeline_runs.created_at desc, artifacts.created_at desc
                        limit 1
                        """
                    ),
                    {"recording_id": recording_id},
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise RecordingStageNotRetryableError("找不到 embedding_indexing 所需的 search.chunks 记录")

        artifact_path = self._storage.resolve(cast(str, row["uri"]))
        if not artifact_path.is_file():
            output_payload = row["output_payload"]
            if not isinstance(output_payload, dict) or not isinstance(output_payload.get("output"), dict):
                raise RecordingStageNotRetryableError("build_search_chunks 没有可用于恢复 search.chunks 的持久化输出")
            try:
                chunks = SearchChunksOutput.model_validate(output_payload["output"])
            except ValueError as error:
                raise RecordingStageNotRetryableError("build_search_chunks 的持久化输出格式无效") from error
            restored = ArtifactStore(self._storage.resolve("")).write_json(
                RecordingId(recording_id),
                PipelineRunId(cast(UUID, row["pipeline_run_id"])),
                StageRunId(cast(UUID, row["producer_stage_run_id"])),
                "build_search_chunks",
                ArtifactPayload(
                    artifact_type="search.chunks",
                    artifact_version=cast(str, row["artifact_version"]),
                    data=chunks.model_dump(mode="json"),
                ),
            )
            if restored.uri != row["uri"]:
                raise RecordingStageNotRetryableError("search.chunks 的恢复路径与原 artifact 记录不一致")
            logger.info(
                "录音索引：已从 build_search_chunks 持久化输出恢复输入 artifact，recording_id=%s uri=%s",
                recording_id,
                restored.uri,
            )

        return StageRunId(cast(UUID, row["embedding_stage_run_id"])), cast(str, row["embedding_stage_status"])

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

    def delete_recording(self, user: CurrentUser, recording_id: UUID) -> None:
        """Delete a recording, its database results, and all recording-owned storage trees."""
        self._access.require_edit(recording_id, user)
        with self._engine.begin() as connection:
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
        recording_id: RecordingId,
        workspace_id: UUID,
        owner_user_id: UUID,
        title: str,
        file_name: str,
        storage_path: str,
        location: str | None,
        mime_type: str,
        file_size_bytes: int,
    ) -> DatabaseRow:
        with self._engine.begin() as connection:
            return dict(
                connection.execute(
                    text(
                        """
                    insert into recordings (id, workspace_id, owner_user_id, title, file_name, storage_path, location, mime_type, file_size_bytes, status)
                    values (:id, :workspace_id, :owner_user_id, :title, :file_name, :storage_path, :location, :mime_type, :file_size_bytes, 'uploaded')
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
                    },
                )
                .mappings()
                .one()
            )

    def _delete_recording(self, recording_id: RecordingId) -> None:
        with self._engine.begin() as connection:
            connection.execute(text("delete from recordings where id = :recording_id"), {"recording_id": recording_id})

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
    async def _write_upload(upload: UploadFile, destination: Path) -> int:
        destination.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with destination.open("wb") as file:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                file.write(chunk)
        if total == 0:
            destination.unlink(missing_ok=True)
            raise ValueError("Uploaded audio file is empty")
        return total

    @staticmethod
    def _pipeline_runs(connection: Any, recording_id: UUID) -> list[DatabaseRow]:
        return [
            dict(row)
            for row in connection.execute(
                text(
                    "select pipeline_runs.*, pipeline_runs.subject_id as recording_id "
                    "from pipeline_runs where subject_type = 'recording' and subject_id = :recording_id order by created_at desc"
                ),
                {"recording_id": recording_id},
            )
            .mappings()
            .all()
        ]

    @staticmethod
    def _as_recording_pipeline_run(run: DatabaseRow) -> DatabaseRow:
        """Keep the recording HTTP contract independent of generic runtime subject naming."""
        return {**run, "recording_id": run["subject_id"]}

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
