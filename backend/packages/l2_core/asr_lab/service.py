from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast
from uuid import UUID, uuid4

from fastapi import UploadFile
from sqlalchemy import Connection, Engine, text
from sqlalchemy.exc import IntegrityError

from l1_foundation.infrastructure.storage.local import LocalStorage
from l2_core.access.recordings import RecordingAccessService
from l2_core.asr_lab.encrypted_datasets import EncryptedDatasetStore, crop_audio_to_flac
from l2_core.auth.contracts import CurrentUser
from l2_core.evaluation.contracts import ApprovedAnnotation, DatasetVersionPreview
from l2_core.evaluation.datasets import build_dataset_preview

type DatabaseRow = dict[str, Any]
train_logger = logging.getLogger("train")
evaluation_logger = logging.getLogger("evaluation")


class AsrLabNotFoundError(LookupError):
    """Raised when a workspace-scoped ASR Lab resource does not exist."""


class AsrLabConflictError(ValueError):
    """Raised when a requested state transition is invalid or stale."""


class AsrLabPermissionError(PermissionError):
    """Raised when a workspace member lacks the required management role."""


class AsrLabService:
    """Workspace-scoped use cases for annotation, datasets, runs, and models."""

    def __init__(
        self,
        engine: Engine,
        storage: LocalStorage,
        project_dataset_root: Path,
        *,
        evaluation_context: str = "",
    ) -> None:
        self._engine = engine
        self._storage = storage
        self._recording_access = RecordingAccessService(engine)
        self._encrypted_datasets = EncryptedDatasetStore(project_dataset_root)
        self._evaluation_context = evaluation_context

    def list_datasets(self, user: CurrentUser) -> list[DatabaseRow]:
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    select datasets.*,
                           count(distinct annotations.id) filter (where annotations.status = 'draft') as draft_count,
                           count(distinct annotations.id) filter (where annotations.status = 'reviewed') as reviewed_count,
                           count(distinct annotations.id) filter (where annotations.status = 'approved') as approved_count,
                           count(distinct assets.id) as asset_count,
                           max(versions.version_number) filter (where versions.status = 'frozen') as latest_version_number
                    from evaluation_datasets datasets
                    left join evaluation_source_assets assets
                      on assets.dataset_id = datasets.id
                     and assets.archived_at is null
                    left join evaluation_annotations annotations
                      on annotations.dataset_id = datasets.id
                     and annotations.source_asset_id = assets.id
                    left join evaluation_dataset_versions versions on versions.dataset_id = datasets.id
                    where datasets.workspace_id = :workspace_id
                      and datasets.status = 'active'
                    group by datasets.id
                    order by datasets.updated_at desc
                    """
                ),
                {"workspace_id": user.current_workspace_id},
            ).mappings()
        return [dict(row) for row in rows]

    def create_dataset(self, user: CurrentUser, name: str, description: str | None) -> DatabaseRow:
        cleaned_name = name.strip()
        if not cleaned_name:
            raise ValueError("Dataset name is required")
        with self._engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    insert into evaluation_datasets (
                        workspace_id, name, description, task_type, created_by_user_id
                    )
                    values (:workspace_id, :name, :description, 'asr', :user_id)
                    returning *
                    """
                ),
                {
                    "workspace_id": user.current_workspace_id,
                    "name": cleaned_name,
                    "description": description.strip() if description and description.strip() else None,
                    "user_id": user.id,
                },
            ).mappings().one()
        return dict(row)

    def delete_dataset(self, user: CurrentUser, dataset_id: UUID) -> DatabaseRow:
        self._require_manager(user)
        artifact_uris: list[str] = []
        with self._engine.begin() as connection:
            self._dataset_row(connection, user, dataset_id)
            connection.execute(
                text("select id from evaluation_datasets where id = :dataset_id for update"),
                {"dataset_id": dataset_id},
            )
            version_count = cast(
                int,
                connection.execute(
                    text("select count(*) from evaluation_dataset_versions where dataset_id = :dataset_id"),
                    {"dataset_id": dataset_id},
                ).scalar_one(),
            )
            if version_count > 0:
                connection.execute(
                    text(
                        """
                        update evaluation_datasets
                        set status = 'archived', archived_at = now(), updated_at = now()
                        where id = :dataset_id
                        """
                    ),
                    {"dataset_id": dataset_id},
                )
                return {
                    "id": dataset_id,
                    "mode": "archived",
                    "retained_version_count": version_count,
                }

            artifact_uris = [
                cast(str, value)
                for value in connection.execute(
                    text(
                        """
                        select artifact_uri
                        from evaluation_source_assets
                        where dataset_id = :dataset_id and artifact_uri is not null
                        """
                    ),
                    {"dataset_id": dataset_id},
                ).scalars()
            ]
            connection.execute(
                text("delete from evaluation_annotations where dataset_id = :dataset_id"),
                {"dataset_id": dataset_id},
            )
            connection.execute(
                text("delete from evaluation_source_assets where dataset_id = :dataset_id"),
                {"dataset_id": dataset_id},
            )
            connection.execute(
                text("delete from evaluation_datasets where id = :dataset_id"),
                {"dataset_id": dataset_id},
            )

        for artifact_uri in artifact_uris:
            self._storage.remove_tree(str(Path(artifact_uri).parent))
        return {
            "id": dataset_id,
            "mode": "deleted",
            "retained_version_count": 0,
        }

    def get_dataset(self, user: CurrentUser, dataset_id: UUID) -> DatabaseRow:
        with self._engine.connect() as connection:
            dataset = self._dataset_row(connection, user, dataset_id)
            assets = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select assets.*,
                               count(annotations.id) as annotation_count,
                               count(annotations.id) filter (where annotations.status = 'approved') as approved_count
                        from evaluation_source_assets assets
                        left join evaluation_annotations annotations
                          on annotations.source_asset_id = assets.id
                         and annotations.dataset_id = :dataset_id
                        where assets.workspace_id = :workspace_id
                          and assets.dataset_id = :dataset_id
                          and assets.archived_at is null
                        group by assets.id
                        order by assets.created_at desc
                        """
                    ),
                    {"dataset_id": dataset_id, "workspace_id": user.current_workspace_id},
                ).mappings()
            ]
            annotations = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select annotations.*
                        from evaluation_annotations annotations
                        join evaluation_source_assets assets on assets.id = annotations.source_asset_id
                        where annotations.dataset_id = :dataset_id
                          and assets.archived_at is null
                        order by annotations.source_asset_id, annotations.start_ms, annotations.created_at
                        """
                    ),
                    {"dataset_id": dataset_id},
                ).mappings()
            ]
            versions = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select *
                        from evaluation_dataset_versions
                        where dataset_id = :dataset_id
                        order by version_number desc
                        """
                    ),
                    {"dataset_id": dataset_id},
                ).mappings()
            ]
        return {
            "dataset": dataset,
            "assets": assets,
            "annotations": annotations,
            "versions": versions,
        }

    async def create_sample(
        self,
        user: CurrentUser,
        dataset_id: UUID,
        *,
        audio_upload: UploadFile,
        start_ms: int,
        end_ms: int,
        reference_text: str,
        language: str | None,
        train_allowed: bool,
        evaluation_allowed: bool,
        contains_sensitive_data: bool,
        project_persistence_password: str | None = None,
    ) -> DatabaseRow:
        cleaned_text = reference_text.strip()
        if not cleaned_text:
            raise ValueError("Reference text is required")
        self._require_dataset(user, dataset_id)
        if project_persistence_password is not None:
            self._encrypted_datasets.verify_password(str(dataset_id), project_persistence_password)

        source_file_name = self._safe_file_name(audio_upload.filename)
        try:
            with TemporaryDirectory(prefix="asr-lab-source-") as temporary_directory:
                source = Path(temporary_directory) / source_file_name
                await self._write_upload(audio_upload, source)
                if start_ms < 0 or end_ms <= start_ms:
                    raise ValueError("Audio interval must satisfy 0 <= start_ms < end_ms")
                audio = await asyncio.to_thread(crop_audio_to_flac, source, start_ms, end_ms)
        finally:
            await audio_upload.close()

        asset_id = uuid4()
        annotation_id = uuid4()
        file_name = f"{asset_id}.flac"
        storage_key = f"asr-lab/samples/{asset_id}/{file_name}"
        destination = self._storage.resolve(storage_key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(audio)
        sample_duration_ms = self._probe_duration_ms_sync(destination)
        checksum = hashlib.sha256(audio).hexdigest()
        try:
            with self._engine.begin() as connection:
                self._dataset_row(connection, user, dataset_id)
                connection.execute(
                    text(
                        """
                        insert into evaluation_source_assets (
                            id, workspace_id, dataset_id, artifact_uri, checksum, file_name, mime_type,
                            file_size_bytes, duration_ms, created_by_user_id
                        )
                        values (
                            :id, :workspace_id, :dataset_id, :artifact_uri, :checksum, :file_name,
                            'audio/flac', :file_size_bytes, :duration_ms, :user_id
                        )
                        """
                    ),
                    {
                        "id": asset_id,
                        "workspace_id": user.current_workspace_id,
                        "dataset_id": dataset_id,
                        "artifact_uri": storage_key,
                        "checksum": checksum,
                        "file_name": file_name,
                        "file_size_bytes": len(audio),
                        "duration_ms": sample_duration_ms,
                        "user_id": user.id,
                    },
                )
                row = connection.execute(
                    text(
                        """
                        insert into evaluation_annotations (
                            id, dataset_id, source_asset_id, start_ms, end_ms, reference_text,
                            language, train_allowed, evaluation_allowed, contains_sensitive_data,
                            group_key, created_by_user_id
                        )
                        values (
                            :id, :dataset_id, :source_asset_id, 0, :end_ms, :reference_text,
                            :language, :train_allowed, :evaluation_allowed, :contains_sensitive_data,
                            :group_key, :user_id
                        )
                        returning *
                        """
                    ),
                    {
                        "id": annotation_id,
                        "dataset_id": dataset_id,
                        "source_asset_id": asset_id,
                        "end_ms": sample_duration_ms,
                        "reference_text": cleaned_text,
                        "language": language.strip() if language and language.strip() else None,
                        "train_allowed": train_allowed,
                        "evaluation_allowed": evaluation_allowed,
                        "contains_sensitive_data": contains_sensitive_data,
                        "group_key": str(asset_id),
                        "user_id": user.id,
                    },
                ).mappings().one()
        except Exception:
            self._storage.remove_tree(str(Path(storage_key).parent))
            raise

        result = dict(row)
        if project_persistence_password is not None:
            self._encrypted_datasets.persist_audio_sample(
                package_id=str(dataset_id),
                sample_id=str(annotation_id),
                password=project_persistence_password,
                audio=audio,
                text=cleaned_text,
            )
            result["project_persisted"] = True
        return result

    def import_recording(self, user: CurrentUser, dataset_id: UUID, recording_id: UUID) -> DatabaseRow:
        self._require_dataset(user, dataset_id)
        self._recording_access.require_view(recording_id, user)
        with self._engine.connect() as connection:
            recording_row = connection.execute(
                text(
                    """
                    select id, workspace_id, storage_path, file_name, mime_type,
                           file_size_bytes, duration_seconds
                    from recordings
                    where id = :recording_id
                    """
                ),
                {"recording_id": recording_id},
            ).mappings().one_or_none()
        if recording_row is None:
            raise AsrLabNotFoundError(str(recording_id))
        recording = dict(recording_row)
        source_path = self._storage.resolve(cast(str, recording["storage_path"]))
        if not source_path.is_file():
            raise AsrLabNotFoundError("Recording audio not found")
        checksum = self._file_checksum(source_path)
        duration_seconds = recording["duration_seconds"]
        duration_ms = self._probe_duration_ms_sync(source_path) if duration_seconds is None else int(duration_seconds) * 1000
        with self._engine.begin() as connection:
            existing = connection.execute(
                text(
                    """
                    select *
                    from evaluation_source_assets
                    where workspace_id = :workspace_id
                      and dataset_id = :dataset_id
                      and recording_id = :recording_id
                      and archived_at is null
                    order by created_at
                    limit 1
                    """
                ),
                {"workspace_id": user.current_workspace_id, "dataset_id": dataset_id, "recording_id": recording_id},
            ).mappings().one_or_none()
            if existing is not None:
                result = dict(existing)
            else:
                result = dict(
                    connection.execute(
                        text(
                            """
                            insert into evaluation_source_assets (
                                workspace_id, dataset_id, recording_id, checksum, file_name, mime_type,
                                file_size_bytes, duration_ms, created_by_user_id
                            )
                            values (
                                :workspace_id, :dataset_id, :recording_id, :checksum, :file_name, :mime_type,
                                :file_size_bytes, :duration_ms, :user_id
                            )
                            returning *
                            """
                        ),
                        {
                            "workspace_id": user.current_workspace_id,
                            "dataset_id": dataset_id,
                            "recording_id": recording_id,
                            "checksum": checksum,
                            "file_name": recording["file_name"],
                            "mime_type": recording["mime_type"],
                            "file_size_bytes": recording["file_size_bytes"],
                            "duration_ms": duration_ms,
                            "user_id": user.id,
                        },
                    ).mappings().one()
                )
        return result

    def get_asset_audio(self, user: CurrentUser, asset_id: UUID) -> DatabaseRow:
        with self._engine.connect() as connection:
            row = connection.execute(
                text(
                    """
                    select assets.*, recordings.storage_path as recording_storage_path
                    from evaluation_source_assets assets
                    left join recordings on recordings.id = assets.recording_id
                    where assets.id = :asset_id
                      and assets.workspace_id = :workspace_id
                      and assets.archived_at is null
                    """
                ),
                {"asset_id": asset_id, "workspace_id": user.current_workspace_id},
            ).mappings().one_or_none()
        if row is None:
            raise AsrLabNotFoundError(str(asset_id))
        result = dict(row)
        result["storage_path"] = result["artifact_uri"] or result["recording_storage_path"]
        return result

    def delete_asset(self, user: CurrentUser, asset_id: UUID, *, delete_annotations: bool) -> DatabaseRow:
        with self._engine.begin() as connection:
            asset_row = connection.execute(
                text(
                    """
                    select *
                    from evaluation_source_assets
                    where id = :asset_id
                      and workspace_id = :workspace_id
                      and archived_at is null
                    for update
                    """
                ),
                {"asset_id": asset_id, "workspace_id": user.current_workspace_id},
            ).mappings().one_or_none()
            if asset_row is None:
                raise AsrLabNotFoundError(str(asset_id))
            asset = dict(asset_row)
            annotation_count = cast(
                int,
                connection.execute(
                    text("select count(*) from evaluation_annotations where source_asset_id = :asset_id"),
                    {"asset_id": asset_id},
                ).scalar_one(),
            )
            if annotation_count and not delete_annotations:
                raise AsrLabConflictError(f"录音包含 {annotation_count} 条标注，请确认同时删除这些标注")
            snapshot_reference_count = cast(
                int,
                connection.execute(
                    text("select count(*) from evaluation_cases where source_asset_id = :asset_id"),
                    {"asset_id": asset_id},
                ).scalar_one(),
            )
            if snapshot_reference_count:
                connection.execute(
                    text("update evaluation_source_assets set archived_at = now() where id = :asset_id"),
                    {"asset_id": asset_id},
                )
                return {
                    "id": asset_id,
                    "mode": "archived",
                    "deleted_annotation_count": 0,
                    "retained_snapshot_reference_count": snapshot_reference_count,
                }

            deleted_annotation_count = connection.execute(
                text("delete from evaluation_annotations where source_asset_id = :asset_id"),
                {"asset_id": asset_id},
            ).rowcount
            connection.execute(
                text("update evaluation_source_assets set archived_at = now() where id = :asset_id"),
                {"asset_id": asset_id},
            )

        artifact_uri = asset.get("artifact_uri")
        if isinstance(artifact_uri, str):
            self._storage.remove_tree(str(Path(artifact_uri).parent))
        return {
            "id": asset_id,
            "mode": "deleted",
            "deleted_annotation_count": deleted_annotation_count,
            "retained_snapshot_reference_count": 0,
        }

    def create_annotation(
        self,
        user: CurrentUser,
        dataset_id: UUID,
        *,
        source_asset_id: UUID,
        start_ms: int,
        end_ms: int,
        reference_text: str,
        language: str | None,
        train_allowed: bool,
        evaluation_allowed: bool,
        contains_sensitive_data: bool,
        project_persistence_password: str | None = None,
    ) -> DatabaseRow:
        cleaned_text = reference_text.strip()
        if not cleaned_text:
            raise ValueError("Reference text is required")
        if project_persistence_password is not None:
            self._require_dataset(user, dataset_id)
            self._encrypted_datasets.verify_password(str(dataset_id), project_persistence_password)
        annotation_id = uuid4()
        with self._engine.begin() as connection:
            self._dataset_row(connection, user, dataset_id)
            duration_ms = self._asset_duration(connection, user, dataset_id, source_asset_id)
            self._validate_interval(start_ms, end_ms, duration_ms)
            row = connection.execute(
                text(
                    """
                    insert into evaluation_annotations (
                        id, dataset_id, source_asset_id, start_ms, end_ms, reference_text,
                        language, train_allowed, evaluation_allowed, contains_sensitive_data,
                        group_key, created_by_user_id
                    )
                    values (
                        :id, :dataset_id, :source_asset_id, :start_ms, :end_ms, :reference_text,
                        :language, :train_allowed, :evaluation_allowed, :contains_sensitive_data,
                        :group_key, :user_id
                    )
                    returning *
                    """
                ),
                {
                    "id": annotation_id,
                    "dataset_id": dataset_id,
                    "source_asset_id": source_asset_id,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "reference_text": cleaned_text,
                    "language": language.strip() if language and language.strip() else None,
                    "train_allowed": train_allowed,
                    "evaluation_allowed": evaluation_allowed,
                    "contains_sensitive_data": contains_sensitive_data,
                    "group_key": str(source_asset_id),
                    "user_id": user.id,
                },
            ).mappings().one()
        result = dict(row)
        if project_persistence_password is not None:
            self._persist_annotation_sample(user, result, project_persistence_password)
            result["project_persisted"] = True
        return result

    def update_annotation(
        self,
        user: CurrentUser,
        annotation_id: UUID,
        *,
        revision: int,
        start_ms: int,
        end_ms: int,
        reference_text: str,
        language: str | None,
        train_allowed: bool,
        evaluation_allowed: bool,
        contains_sensitive_data: bool,
        project_persistence_password: str | None = None,
    ) -> DatabaseRow:
        cleaned_text = reference_text.strip()
        if not cleaned_text:
            raise ValueError("Reference text is required")
        with self._engine.begin() as connection:
            existing = self._annotation_row(connection, user, annotation_id)
            if project_persistence_password is not None:
                self._encrypted_datasets.verify_password(str(existing["dataset_id"]), project_persistence_password)
            duration_ms = self._asset_duration(
                connection,
                user,
                UUID(str(existing["dataset_id"])),
                UUID(str(existing["source_asset_id"])),
            )
            self._validate_interval(start_ms, end_ms, duration_ms)
            row = connection.execute(
                text(
                    """
                    update evaluation_annotations
                    set start_ms = :start_ms,
                        end_ms = :end_ms,
                        reference_text = :reference_text,
                        language = :language,
                        train_allowed = :train_allowed,
                        evaluation_allowed = :evaluation_allowed,
                        contains_sensitive_data = :contains_sensitive_data,
                        status = 'draft',
                        reviewed_by_user_id = null,
                        reviewed_at = null,
                        approved_by_user_id = null,
                        approved_at = null,
                        revision = revision + 1,
                        updated_at = now()
                    where id = :annotation_id
                      and revision = :revision
                    returning *
                    """
                ),
                {
                    "annotation_id": annotation_id,
                    "revision": revision,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "reference_text": cleaned_text,
                    "language": language.strip() if language and language.strip() else None,
                    "train_allowed": train_allowed,
                    "evaluation_allowed": evaluation_allowed,
                    "contains_sensitive_data": contains_sensitive_data,
                },
            ).mappings().one_or_none()
        if row is None:
            raise AsrLabConflictError("Annotation was changed by another user; reload and retry")
        result = dict(row)
        if project_persistence_password is not None:
            self._persist_annotation_sample(user, result, project_persistence_password)
            result["project_persisted"] = True
        return result

    def list_encrypted_project_datasets(self) -> list[DatabaseRow]:
        return self._encrypted_datasets.list_packages()

    def import_encrypted_project_dataset(self, user: CurrentUser, package_id: str, password: str) -> DatabaseRow:
        samples = self._encrypted_datasets.load(package_id, password)
        if not samples:
            raise ValueError("加密数据集没有可导入的样本")

        dataset_id = uuid4()
        prepared_assets: list[dict[str, Any]] = []
        written_paths: list[Path] = []
        try:
            for sample in samples:
                asset_id = uuid4()
                file_name = f"{sample.sample_id}.flac"
                storage_key = f"asr-lab/assets/{asset_id}/{file_name}"
                destination = self._storage.resolve(storage_key)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(sample.audio)
                written_paths.append(destination)
                prepared_assets.append(
                    {
                        "asset_id": asset_id,
                        "storage_key": storage_key,
                        "file_name": file_name,
                        "file_size_bytes": len(sample.audio),
                        "checksum": hashlib.sha256(sample.audio).hexdigest(),
                        "duration_ms": self._probe_duration_ms_sync(destination),
                        "reference_text": sample.text,
                    }
                )

            with self._engine.begin() as connection:
                dataset = connection.execute(
                    text(
                        """
                        insert into evaluation_datasets (
                            id, workspace_id, name, description, task_type, created_by_user_id
                        )
                        values (:id, :workspace_id, :name, null, 'asr', :user_id)
                        returning *
                        """
                    ),
                    {
                        "id": dataset_id,
                        "workspace_id": user.current_workspace_id,
                        "name": f"项目加密数据集 {package_id[:8]}",
                        "user_id": user.id,
                    },
                ).mappings().one()
                for prepared in prepared_assets:
                    connection.execute(
                        text(
                            """
                            insert into evaluation_source_assets (
                                id, workspace_id, dataset_id, artifact_uri, checksum, file_name, mime_type,
                                file_size_bytes, duration_ms, created_by_user_id
                            )
                            values (
                                :asset_id, :workspace_id, :dataset_id, :storage_key, :checksum, :file_name,
                                'audio/flac', :file_size_bytes, :duration_ms, :user_id
                            )
                            """
                        ),
                        {
                            **prepared,
                            "workspace_id": user.current_workspace_id,
                            "dataset_id": dataset_id,
                            "user_id": user.id,
                        },
                    )
                    connection.execute(
                        text(
                            """
                            insert into evaluation_annotations (
                                dataset_id, source_asset_id, start_ms, end_ms, reference_text,
                                language, train_allowed, evaluation_allowed, contains_sensitive_data,
                                group_key, created_by_user_id
                            )
                            values (
                                :dataset_id, :asset_id, 0, :duration_ms, :reference_text,
                                'zh', true, true, true, :group_key, :user_id
                            )
                            """
                        ),
                        {
                            **prepared,
                            "dataset_id": dataset_id,
                            "group_key": str(prepared["asset_id"]),
                            "user_id": user.id,
                        },
                    )
            result = dict(dataset)
            result["imported_sample_count"] = len(prepared_assets)
            return result
        except Exception:
            for path in written_paths:
                path.unlink(missing_ok=True)
                with suppress(OSError):
                    path.parent.rmdir()
            raise

    def _persist_annotation_sample(self, user: CurrentUser, annotation: DatabaseRow, password: str) -> None:
        asset = self.get_asset_audio(user, UUID(str(annotation["source_asset_id"])))
        source = self._storage.resolve(cast(str, asset["storage_path"]))
        if not source.is_file():
            raise AsrLabNotFoundError("Audio file not found")
        self._encrypted_datasets.persist_sample(
            package_id=str(annotation["dataset_id"]),
            sample_id=str(annotation["id"]),
            password=password,
            source=source,
            start_ms=cast(int, annotation["start_ms"]),
            end_ms=cast(int, annotation["end_ms"]),
            text=cast(str, annotation["reference_text"]),
        )

    def review_annotation(self, user: CurrentUser, annotation_id: UUID, revision: int) -> DatabaseRow:
        self._require_manager(user)
        return self._transition_annotation(user, annotation_id, revision, "draft", "reviewed")

    def approve_annotation(self, user: CurrentUser, annotation_id: UUID, revision: int) -> DatabaseRow:
        self._require_manager(user)
        return self._transition_annotation(user, annotation_id, revision, "reviewed", "approved")

    def delete_annotation(self, user: CurrentUser, annotation_id: UUID, revision: int) -> None:
        with self._engine.begin() as connection:
            self._annotation_row(connection, user, annotation_id)
            try:
                deleted = connection.execute(
                    text(
                        """
                        delete from evaluation_annotations
                        where id = :annotation_id
                          and revision = :revision
                        returning id
                        """
                    ),
                    {"annotation_id": annotation_id, "revision": revision},
                ).scalar_one_or_none()
            except IntegrityError as error:
                raise AsrLabConflictError("Annotation is referenced by a frozen dataset version") from error
        if deleted is None:
            raise AsrLabConflictError("Annotation was changed by another user; reload and retry")

    def preview_dataset_version(
        self,
        user: CurrentUser,
        dataset_id: UUID,
        *,
        normalization_name: str,
        normalization_version: str,
        seed: str,
        split_strategy_name: Literal["deterministic_group_hash_v1", "all_train_v1"] = "deterministic_group_hash_v1",
    ) -> DatasetVersionPreview:
        with self._engine.connect() as connection:
            self._dataset_row(connection, user, dataset_id)
            rows = connection.execute(
                text(
                    """
                    select annotations.*, assets.checksum as source_checksum
                    from evaluation_annotations annotations
                    join evaluation_source_assets assets on assets.id = annotations.source_asset_id
                    where annotations.dataset_id = :dataset_id
                      and annotations.status = 'approved'
                      and assets.archived_at is null
                      and (
                          (:training_only and annotations.train_allowed)
                          or (
                              not :training_only
                              and (annotations.train_allowed or annotations.evaluation_allowed)
                          )
                      )
                    order by annotations.group_key, annotations.start_ms, annotations.id
                    """
                ),
                {
                    "dataset_id": dataset_id,
                    "training_only": split_strategy_name == "all_train_v1",
                },
            ).mappings().all()
            total_count = cast(
                int,
                connection.execute(
                    text(
                        """
                        select count(*)
                        from evaluation_annotations annotations
                        join evaluation_source_assets assets on assets.id = annotations.source_asset_id
                        where annotations.dataset_id = :dataset_id
                          and assets.archived_at is null
                        """
                    ),
                    {"dataset_id": dataset_id},
                ).scalar_one(),
            )
        annotations = [
            ApprovedAnnotation(
                id=UUID(str(row["id"])),
                source_asset_id=UUID(str(row["source_asset_id"])),
                source_checksum=cast(str, row["source_checksum"]),
                start_ms=cast(int, row["start_ms"]),
                end_ms=cast(int, row["end_ms"]),
                reference_text=cast(str, row["reference_text"]),
                language=cast(str | None, row["language"]),
                group_key=cast(str, row["group_key"]),
                train_allowed=cast(bool, row["train_allowed"]),
                evaluation_allowed=cast(bool, row["evaluation_allowed"]),
            )
            for row in rows
        ]
        preview = build_dataset_preview(
            annotations,
            normalization_name=normalization_name,
            normalization_version=normalization_version,
            seed=seed,
            split_strategy_name=split_strategy_name,
            excluded_count=total_count - len(annotations),
        )
        evaluation_logger.info(
            "ASR 评测：数据版本预览 dataset_id=%s strategy=%s train=%d validation=%d test=%d excluded=%d checksum=%s",
            dataset_id,
            split_strategy_name,
            preview.train.case_count,
            preview.validation.case_count,
            preview.test.case_count,
            preview.excluded_count,
            preview.checksum,
        )
        return preview

    def freeze_dataset_version(
        self,
        user: CurrentUser,
        dataset_id: UUID,
        *,
        normalization_name: str,
        normalization_version: str,
        seed: str,
        split_strategy_name: Literal["deterministic_group_hash_v1", "all_train_v1"] = "deterministic_group_hash_v1",
        expected_checksum: str | None = None,
    ) -> DatabaseRow:
        self._require_manager(user)
        preview = self.preview_dataset_version(
            user,
            dataset_id,
            normalization_name=normalization_name,
            normalization_version=normalization_version,
            seed=seed,
            split_strategy_name=split_strategy_name,
        )
        if expected_checksum is not None and preview.checksum != expected_checksum:
            raise AsrLabConflictError("数据集切片已发生变化，请重新预览并确认后再冻结")
        ratios = (
            {"train": 100, "validation": 0, "test": 0}
            if split_strategy_name == "all_train_v1"
            else {"train": 80, "validation": 10, "test": 10}
        )
        strategy = json.dumps(
            {"name": split_strategy_name, "seed": seed, "ratios": ratios},
            separators=(",", ":"),
        )
        with self._engine.begin() as connection:
            self._dataset_row(connection, user, dataset_id)
            connection.execute(text("select pg_advisory_xact_lock(hashtext(:lock_key))"), {"lock_key": f"evaluation-dataset:{dataset_id}"})
            version_number = cast(
                int,
                connection.execute(
                    text("select coalesce(max(version_number), 0) + 1 from evaluation_dataset_versions where dataset_id = :dataset_id"),
                    {"dataset_id": dataset_id},
                ).scalar_one(),
            )
            version_id = UUID(
                str(
                    connection.execute(
                        text(
                            """
                            insert into evaluation_dataset_versions (
                                dataset_id, version_number, normalization_name, normalization_version,
                                split_strategy, case_count, created_by_user_id
                            )
                            values (
                                :dataset_id, :version_number, :normalization_name, :normalization_version,
                                cast(:split_strategy as jsonb), :case_count, :user_id
                            )
                            returning id
                            """
                        ),
                        {
                            "dataset_id": dataset_id,
                            "version_number": version_number,
                            "normalization_name": normalization_name,
                            "normalization_version": normalization_version,
                            "split_strategy": strategy,
                            "case_count": len(preview.cases),
                            "user_id": user.id,
                        },
                    ).scalar_one()
                )
            )
            for case in preview.cases:
                annotation = case.annotation
                connection.execute(
                    text(
                        """
                        insert into evaluation_cases (
                            dataset_version_id, source_annotation_id, source_asset_id,
                            start_ms, end_ms, reference_text_raw, reference_text_normalized,
                            language, split, group_key, train_allowed, evaluation_allowed
                        )
                        values (
                            :dataset_version_id, :source_annotation_id, :source_asset_id,
                            :start_ms, :end_ms, :reference_text_raw, :reference_text_normalized,
                            :language, :split, :group_key, :train_allowed, :evaluation_allowed
                        )
                        """
                    ),
                    {
                        "dataset_version_id": version_id,
                        "source_annotation_id": annotation.id,
                        "source_asset_id": annotation.source_asset_id,
                        "start_ms": annotation.start_ms,
                        "end_ms": annotation.end_ms,
                        "reference_text_raw": annotation.reference_text,
                        "reference_text_normalized": case.normalized_reference_text,
                        "language": annotation.language,
                        "split": case.split.value,
                        "group_key": annotation.group_key,
                        "train_allowed": annotation.train_allowed,
                        "evaluation_allowed": annotation.evaluation_allowed,
                    },
                )
            row = connection.execute(
                text(
                    """
                    update evaluation_dataset_versions
                    set status = 'frozen', checksum = :checksum, frozen_at = now()
                    where id = :version_id and status = 'building'
                    returning *
                    """
                ),
                {"version_id": version_id, "checksum": preview.checksum},
            ).mappings().one()
        evaluation_logger.info(
            "ASR 评测：数据版本冻结完成 dataset_id=%s version_id=%s version_number=%d strategy=%s train=%d validation=%d test=%d checksum=%s",
            dataset_id,
            version_id,
            version_number,
            split_strategy_name,
            preview.train.case_count,
            preview.validation.case_count,
            preview.test.case_count,
            preview.checksum,
        )
        return dict(row)

    def list_models(self, user: CurrentUser) -> list[DatabaseRow]:
        self._ensure_base_model(user)
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    select *
                    from model_versions
                    where workspace_id = :workspace_id
                      and model_family = 'qwen3_asr'
                    order by
                        case when status = 'deployed' then 0 when status = 'approved' then 1 else 2 end,
                        created_at desc
                    """
                ),
                {"workspace_id": user.current_workspace_id},
            ).mappings()
        return [dict(row) for row in rows]

    def update_model_status(self, user: CurrentUser, model_id: UUID, target_status: str) -> DatabaseRow:
        self._require_manager(user)
        if target_status not in {"approved", "retired"}:
            raise ValueError("Model status transition must target approved or retired")
        with self._engine.begin() as connection:
            current = self._model_row(connection, user, model_id)
            current_status = cast(str, current["status"])
            allowed = {
                "approved": {"candidate", "validated"},
                "retired": {"candidate", "validated", "approved", "deployed"},
            }
            if current_status not in allowed[target_status]:
                raise AsrLabConflictError(f"Model cannot transition from {current_status} to {target_status}")
            row = connection.execute(
                text("update model_versions set status = :status, updated_at = now() where id = :model_id returning *"),
                {"status": target_status, "model_id": model_id},
            ).mappings().one()
        return dict(row)

    def create_training_run(
        self,
        user: CurrentUser,
        *,
        dataset_id: UUID | None,
        dataset_version_id: UUID | None,
        base_model_version_id: UUID,
        preset_name: str,
        candidate_model_name: str,
        run_validation: bool,
        idempotency_key: str,
    ) -> DatabaseRow:
        self._require_manager(user)
        if (dataset_id is None) == (dataset_version_id is None):
            raise ValueError("Provide exactly one of dataset_id or dataset_version_id")
        resolved_version_id = (
            self._get_or_create_training_dataset_version(user, dataset_id, run_validation=run_validation)
            if dataset_id is not None
            else cast(UUID, dataset_version_id)
        )
        with self._engine.begin() as connection:
            self._require_frozen_version(connection, user, resolved_version_id)
            train_count = cast(
                int,
                connection.execute(
                    text("select count(*) from evaluation_cases where dataset_version_id = :version_id and split = 'train'"),
                    {"version_id": resolved_version_id},
                ).scalar_one(),
            )
            if train_count == 0:
                raise AsrLabConflictError("Frozen dataset version has no training cases")
            base_model = self._model_row(connection, user, base_model_version_id)
            runtime_config = base_model.get("runtime_config")
            provider = cast(dict[str, object], runtime_config).get("provider") if isinstance(runtime_config, dict) else None
            if provider != "qwen_hf":
                raise AsrLabConflictError("LoRA training requires a Qwen3-ASR Hugging Face base model")
            if base_model.get("adapter_uri") is not None:
                raise AsrLabConflictError("This training preset currently requires a base checkpoint without an existing adapter")
            row = connection.execute(
                text(
                    """
                    insert into training_runs (
                        workspace_id, dataset_version_id, base_model_version_id,
                        training_method, preset_name, candidate_model_name,
                        idempotency_key, config_snapshot, created_by_user_id
                    )
                    values (
                        :workspace_id, :dataset_version_id, :base_model_version_id,
                        'lora', :preset_name, :candidate_model_name,
                        :idempotency_key, cast(:config_snapshot as jsonb), :user_id
                    )
                    on conflict (workspace_id, idempotency_key)
                    do update set idempotency_key = excluded.idempotency_key
                    returning *
                    """
                ),
                {
                    "workspace_id": user.current_workspace_id,
                    "dataset_version_id": resolved_version_id,
                    "base_model_version_id": base_model_version_id,
                    "preset_name": preset_name.strip(),
                    "candidate_model_name": candidate_model_name.strip(),
                    "idempotency_key": idempotency_key.strip(),
                    "config_snapshot": json.dumps(
                        {
                            "preset_name": preset_name.strip(),
                            "training_method": "lora",
                            "run_validation": run_validation,
                        }
                    ),
                    "user_id": user.id,
                },
            ).mappings().one()
        return dict(row)

    def _get_or_create_training_dataset_version(
        self,
        user: CurrentUser,
        dataset_id: UUID,
        *,
        run_validation: bool,
    ) -> UUID:
        normalization_name = "zh_asr"
        normalization_version = "v1"
        seed = "asr-lab-v1"
        split_strategy_name: Literal["deterministic_group_hash_v1", "all_train_v1"] = (
            "deterministic_group_hash_v1" if run_validation else "all_train_v1"
        )
        preview = self.preview_dataset_version(
            user,
            dataset_id,
            normalization_name=normalization_name,
            normalization_version=normalization_version,
            seed=seed,
            split_strategy_name=split_strategy_name,
        )
        if preview.train.case_count == 0:
            raise AsrLabConflictError("数据集中没有可用于训练的已确认样本")
        with self._engine.connect() as connection:
            existing = connection.execute(
                text(
                    """
                    select versions.id
                    from evaluation_dataset_versions versions
                    join evaluation_datasets datasets on datasets.id = versions.dataset_id
                    where versions.dataset_id = :dataset_id
                      and versions.status = 'frozen'
                      and versions.checksum = :checksum
                      and datasets.workspace_id = :workspace_id
                      and datasets.status = 'active'
                    """
                ),
                {
                    "dataset_id": dataset_id,
                    "checksum": preview.checksum,
                    "workspace_id": user.current_workspace_id,
                },
            ).scalar_one_or_none()
        if existing is not None:
            return UUID(str(existing))
        try:
            version = self.freeze_dataset_version(
                user,
                dataset_id,
                normalization_name=normalization_name,
                normalization_version=normalization_version,
                seed=seed,
                split_strategy_name=split_strategy_name,
            )
            return UUID(str(version["id"]))
        except IntegrityError:
            with self._engine.connect() as connection:
                raced = connection.execute(
                    text(
                        """
                        select id
                        from evaluation_dataset_versions
                        where dataset_id = :dataset_id
                          and status = 'frozen'
                          and checksum = :checksum
                        """
                    ),
                    {"dataset_id": dataset_id, "checksum": preview.checksum},
                ).scalar_one_or_none()
            if raced is None:
                raise
            return UUID(str(raced))

    def list_training_runs(self, user: CurrentUser) -> list[DatabaseRow]:
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    select runs.*,
                           versions.version_number as dataset_version_number,
                           datasets.name as dataset_name,
                           models.name as base_model_name
                    from training_runs runs
                    join evaluation_dataset_versions versions on versions.id = runs.dataset_version_id
                    join evaluation_datasets datasets on datasets.id = versions.dataset_id
                    join model_versions models on models.id = runs.base_model_version_id
                    where runs.workspace_id = :workspace_id
                    order by runs.created_at desc
                    """
                ),
                {"workspace_id": user.current_workspace_id},
            ).mappings()
        return [dict(row) for row in rows]

    def cancel_training_run(self, user: CurrentUser, run_id: UUID) -> DatabaseRow:
        self._require_manager(user)
        return self._cancel_run(user, "training_runs", run_id)

    def delete_training_run(self, user: CurrentUser, run_id: UUID) -> DatabaseRow:
        self._require_manager(user)
        with self._engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    select *
                    from training_runs
                    where id = :run_id and workspace_id = :workspace_id
                    for update
                    """
                ),
                {"run_id": run_id, "workspace_id": user.current_workspace_id},
            ).mappings().one_or_none()
            if row is None:
                raise AsrLabNotFoundError(str(run_id))
            run = dict(row)
            run_status = cast(str, run["status"])
            if run_status in {"preparing", "training", "validating"}:
                raise AsrLabConflictError("训练任务正在运行，请先取消并等待任务进入 cancelled 状态")
            produced_model_count = cast(
                int,
                connection.execute(
                    text("select count(*) from model_versions where training_run_id = :run_id"),
                    {"run_id": run_id},
                ).scalar_one(),
            )
            if run_status == "succeeded" or produced_model_count > 0:
                raise AsrLabConflictError("成功的训练任务已关联候选模型，不能直接删除")
            connection.execute(
                text("delete from training_runs where id = :run_id"),
                {"run_id": run_id},
            )

        self._storage.remove_tree(f"asr-lab/training/{run_id}")
        train_logger.info("ASR LoRA：训练任务已删除 run_id=%s previous_status=%s", run_id, run_status)
        return {
            "id": run_id,
            "status": run_status,
            "deleted": True,
        }

    def create_evaluation_run(
        self,
        user: CurrentUser,
        *,
        dataset_version_id: UUID,
        split: str,
        model_version_ids: list[UUID],
        normalization_name: str,
        normalization_version: str,
        idempotency_key: str,
    ) -> DatabaseRow:
        if split not in {"validation", "test"}:
            raise ValueError("Evaluation split must be validation or test")
        if len(model_version_ids) < 1 or len(set(model_version_ids)) != len(model_version_ids):
            raise ValueError("At least one distinct model version is required")
        with self._engine.begin() as connection:
            self._require_frozen_version(connection, user, dataset_version_id)
            case_count = cast(
                int,
                connection.execute(
                    text("select count(*) from evaluation_cases where dataset_version_id = :version_id and split = :split"),
                    {"version_id": dataset_version_id, "split": split},
                ).scalar_one(),
            )
            if case_count == 0:
                raise AsrLabConflictError(f"Frozen dataset version has no {split} cases")
            for model_id in model_version_ids:
                self._model_row(connection, user, model_id)
            run = connection.execute(
                text(
                    """
                    insert into evaluation_runs (
                        workspace_id, dataset_version_id, evaluator_type, split,
                        idempotency_key, config_snapshot, total_case_count, created_by_user_id
                    )
                    values (
                        :workspace_id, :dataset_version_id, 'asr', :split,
                        :idempotency_key, cast(:config_snapshot as jsonb), :total_case_count, :user_id
                    )
                    on conflict (workspace_id, idempotency_key)
                    do update set idempotency_key = excluded.idempotency_key
                    returning *
                    """
                ),
                {
                    "workspace_id": user.current_workspace_id,
                    "dataset_version_id": dataset_version_id,
                    "split": split,
                    "idempotency_key": idempotency_key.strip(),
                    "config_snapshot": json.dumps(
                        {
                            "normalization_name": normalization_name,
                            "normalization_version": normalization_version,
                            "asr_context": self._evaluation_context,
                        },
                        separators=(",", ":"),
                    ),
                    "total_case_count": case_count * len(model_version_ids),
                    "user_id": user.id,
                },
            ).mappings().one()
            run_id = UUID(str(run["id"]))
            existing_count = cast(
                int,
                connection.execute(
                    text("select count(*) from evaluation_run_models where evaluation_run_id = :run_id"),
                    {"run_id": run_id},
                ).scalar_one(),
            )
            if existing_count == 0:
                for position, model_id in enumerate(model_version_ids):
                    connection.execute(
                        text(
                            """
                            insert into evaluation_run_models (
                                evaluation_run_id, model_version_id, role, position
                            )
                            values (:run_id, :model_id, :role, :position)
                            """
                        ),
                        {"run_id": run_id, "model_id": model_id, "role": "baseline" if position == 0 else "candidate", "position": position},
                    )
        evaluation_logger.info(
            "ASR 评测：任务已创建 run_id=%s dataset_version_id=%s split=%s models=%s case_count=%d total_model_cases=%d",
            run_id,
            dataset_version_id,
            split,
            ",".join(str(model_id) for model_id in model_version_ids),
            case_count,
            case_count * len(model_version_ids),
        )
        return dict(run)

    def list_evaluation_runs(self, user: CurrentUser) -> list[DatabaseRow]:
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    select runs.*,
                           datasets.name as dataset_name,
                           versions.version_number as dataset_version_number,
                           coalesce(
                               jsonb_agg(
                                   jsonb_build_object(
                                       'id', models.id,
                                       'name', models.name,
                                       'version', models.version,
                                       'role', run_models.role,
                                       'position', run_models.position
                                   )
                                   order by run_models.position
                               ) filter (where models.id is not null),
                               '[]'::jsonb
                           ) as models
                    from evaluation_runs runs
                    join evaluation_dataset_versions versions on versions.id = runs.dataset_version_id
                    join evaluation_datasets datasets on datasets.id = versions.dataset_id
                    left join evaluation_run_models run_models on run_models.evaluation_run_id = runs.id
                    left join model_versions models on models.id = run_models.model_version_id
                    where runs.workspace_id = :workspace_id
                    group by runs.id, datasets.name, versions.version_number
                    order by runs.created_at desc
                    """
                ),
                {"workspace_id": user.current_workspace_id},
            ).mappings()
        return [dict(row) for row in rows]

    def get_evaluation_run(self, user: CurrentUser, run_id: UUID) -> DatabaseRow:
        with self._engine.connect() as connection:
            run_row = connection.execute(
                text("select * from evaluation_runs where id = :run_id and workspace_id = :workspace_id"),
                {"run_id": run_id, "workspace_id": user.current_workspace_id},
            ).mappings().one_or_none()
            if run_row is None:
                raise AsrLabNotFoundError(str(run_id))
            models = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select models.*, run_models.role, run_models.position
                        from evaluation_run_models run_models
                        join model_versions models on models.id = run_models.model_version_id
                        where run_models.evaluation_run_id = :run_id
                        order by run_models.position
                        """
                    ),
                    {"run_id": run_id},
                ).mappings()
            ]
            metrics = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select *
                        from evaluation_metric_values
                        where evaluation_run_id = :run_id
                          and evaluation_case_id is null
                        order by model_version_id nulls first, metric_name
                        """
                    ),
                    {"run_id": run_id},
                ).mappings()
            ]
            cases = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select results.*, cases.source_asset_id, cases.start_ms, cases.end_ms,
                               cases.reference_text_raw, cases.reference_text_normalized,
                               assets.file_name
                        from evaluation_case_results results
                        join evaluation_cases cases on cases.id = results.evaluation_case_id
                        join evaluation_source_assets assets on assets.id = cases.source_asset_id
                        where results.evaluation_run_id = :run_id
                        order by cases.source_asset_id, cases.start_ms, results.model_version_id
                        """
                    ),
                    {"run_id": run_id},
                ).mappings()
            ]
        return {"run": dict(run_row), "models": models, "metrics": metrics, "case_results": cases}

    def cancel_evaluation_run(self, user: CurrentUser, run_id: UUID) -> DatabaseRow:
        return self._cancel_run(user, "evaluation_runs", run_id)

    def delete_evaluation_run(self, user: CurrentUser, run_id: UUID) -> DatabaseRow:
        self._require_manager(user)
        with self._engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    select status
                    from evaluation_runs
                    where id = :run_id and workspace_id = :workspace_id
                    for update
                    """
                ),
                {"run_id": run_id, "workspace_id": user.current_workspace_id},
            ).mappings().one_or_none()
            if row is None:
                raise AsrLabNotFoundError(str(run_id))
            run_status = cast(str, row["status"])
            if run_status == "running":
                raise AsrLabConflictError("评测任务正在运行，请先取消并等待任务进入 cancelled 状态")
            connection.execute(
                text("delete from evaluation_runs where id = :run_id"),
                {"run_id": run_id},
            )
        evaluation_logger.info("ASR 评测：任务已删除 run_id=%s previous_status=%s", run_id, run_status)
        return {
            "id": run_id,
            "status": run_status,
            "deleted": True,
        }

    def _transition_annotation(
        self,
        user: CurrentUser,
        annotation_id: UUID,
        revision: int,
        expected_status: str,
        target_status: str,
    ) -> DatabaseRow:
        reviewer_assignments = (
            "reviewed_by_user_id = :user_id, reviewed_at = now()"
            if target_status == "reviewed"
            else "approved_by_user_id = :user_id, approved_at = now()"
        )
        with self._engine.begin() as connection:
            self._annotation_row(connection, user, annotation_id)
            row = connection.execute(
                text(
                    f"""
                    update evaluation_annotations
                    set status = :target_status,
                        {reviewer_assignments},
                        revision = revision + 1,
                        updated_at = now()
                    where id = :annotation_id
                      and revision = :revision
                      and status = :expected_status
                    returning *
                    """
                ),
                {
                    "annotation_id": annotation_id,
                    "revision": revision,
                    "expected_status": expected_status,
                    "target_status": target_status,
                    "user_id": user.id,
                },
            ).mappings().one_or_none()
        if row is None:
            raise AsrLabConflictError(f"Annotation must be current and {expected_status} before it can become {target_status}")
        return dict(row)

    def _cancel_run(self, user: CurrentUser, table_name: str, run_id: UUID) -> DatabaseRow:
        if table_name not in {"training_runs", "evaluation_runs"}:
            raise ValueError("Unsupported run table")
        with self._engine.begin() as connection:
            row = connection.execute(
                text(
                    f"""
                    update {table_name}
                    set cancel_requested = true,
                        status = case when status = 'queued' then 'cancelled' else status end,
                        finished_at = case when status = 'queued' then now() else finished_at end,
                        updated_at = now()
                    where id = :run_id
                      and workspace_id = :workspace_id
                      and status not in ('succeeded', 'failed', 'cancelled')
                    returning *
                    """
                ),
                {"run_id": run_id, "workspace_id": user.current_workspace_id},
            ).mappings().one_or_none()
        if row is None:
            raise AsrLabConflictError("Run is missing or already finished")
        return dict(row)

    def _ensure_base_model(self, user: CurrentUser) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    insert into model_versions (
                        workspace_id, model_family, name, version, base_model_name,
                        status, runtime_config, metadata, created_by_user_id
                    )
                    values (
                        :workspace_id, 'qwen3_asr', 'Qwen3-ASR-1.7B-HF', 'base',
                        'Qwen/Qwen3-ASR-1.7B-hf', 'approved',
                        '{"provider":"qwen_hf"}'::jsonb,
                        '{"source":"built_in","purpose":"lora_training_and_evaluation"}'::jsonb,
                        :user_id
                    )
                    on conflict (workspace_id, model_family, name, version) do nothing
                    """
                ),
                {"workspace_id": user.current_workspace_id, "user_id": user.id},
            )

    def _require_dataset(self, user: CurrentUser, dataset_id: UUID) -> None:
        with self._engine.connect() as connection:
            self._dataset_row(connection, user, dataset_id)

    @staticmethod
    def _dataset_row(connection: Connection, user: CurrentUser, dataset_id: UUID) -> DatabaseRow:
        row = connection.execute(
            text(
                """
                select *
                from evaluation_datasets
                where id = :dataset_id
                  and workspace_id = :workspace_id
                  and status = 'active'
                """
            ),
            {"dataset_id": dataset_id, "workspace_id": user.current_workspace_id},
        ).mappings().one_or_none()
        if row is None:
            raise AsrLabNotFoundError(str(dataset_id))
        return dict(row)

    @staticmethod
    def _annotation_row(connection: Connection, user: CurrentUser, annotation_id: UUID) -> DatabaseRow:
        row = connection.execute(
            text(
                """
                select annotations.*
                from evaluation_annotations annotations
                join evaluation_datasets datasets on datasets.id = annotations.dataset_id
                where annotations.id = :annotation_id
                  and datasets.workspace_id = :workspace_id
                  and datasets.status = 'active'
                """
            ),
            {"annotation_id": annotation_id, "workspace_id": user.current_workspace_id},
        ).mappings().one_or_none()
        if row is None:
            raise AsrLabNotFoundError(str(annotation_id))
        return dict(row)

    @staticmethod
    def _asset_duration(connection: Connection, user: CurrentUser, dataset_id: UUID, asset_id: UUID) -> int:
        value = connection.execute(
            text(
                """
                select duration_ms
                from evaluation_source_assets
                where id = :asset_id
                  and workspace_id = :workspace_id
                  and dataset_id = :dataset_id
                  and archived_at is null
                """
            ),
            {"asset_id": asset_id, "workspace_id": user.current_workspace_id, "dataset_id": dataset_id},
        ).scalar_one_or_none()
        if value is None:
            raise AsrLabNotFoundError(str(asset_id))
        return cast(int, value)

    @staticmethod
    def _require_frozen_version(connection: Connection, user: CurrentUser, version_id: UUID) -> None:
        exists = connection.execute(
            text(
                """
                select 1
                from evaluation_dataset_versions versions
                join evaluation_datasets datasets on datasets.id = versions.dataset_id
                where versions.id = :version_id
                  and versions.status = 'frozen'
                  and datasets.workspace_id = :workspace_id
                """
            ),
            {"version_id": version_id, "workspace_id": user.current_workspace_id},
        ).scalar_one_or_none()
        if exists is None:
            raise AsrLabNotFoundError(str(version_id))

    @staticmethod
    def _model_row(connection: Connection, user: CurrentUser, model_id: UUID) -> DatabaseRow:
        row = connection.execute(
            text("select * from model_versions where id = :model_id and workspace_id = :workspace_id"),
            {"model_id": model_id, "workspace_id": user.current_workspace_id},
        ).mappings().one_or_none()
        if row is None:
            raise AsrLabNotFoundError(str(model_id))
        return dict(row)

    @staticmethod
    def _validate_interval(start_ms: int, end_ms: int, duration_ms: int) -> None:
        if start_ms < 0 or end_ms <= start_ms:
            raise ValueError("Audio interval must satisfy 0 <= start_ms < end_ms")
        if end_ms > duration_ms:
            raise ValueError(f"Audio interval ends after source duration ({duration_ms} ms)")

    @staticmethod
    def _require_manager(user: CurrentUser) -> None:
        membership = next((item for item in user.memberships if item.workspace_id == user.current_workspace_id), None)
        if membership is not None and membership.role not in {"owner", "admin"}:
            raise AsrLabPermissionError("Workspace owner or admin role is required")

    @staticmethod
    def _safe_file_name(value: str | None) -> str:
        file_name = Path(value or "audio.bin").name.strip()
        if not file_name or file_name in {".", ".."}:
            raise ValueError("Audio file name is invalid")
        return file_name

    @staticmethod
    async def _write_upload(upload: UploadFile, destination: Path) -> tuple[int, str]:
        digest = hashlib.sha256()
        size = 0
        with destination.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size == 0:
            raise ValueError("Audio file is empty")
        return size, digest.hexdigest()

    @staticmethod
    def _file_checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    async def _probe_duration_ms(cls, path: Path) -> int:
        return await asyncio.to_thread(cls._probe_duration_ms_sync, path)

    @staticmethod
    def _probe_duration_ms_sync(path: Path) -> int:
        import subprocess

        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise ValueError(f"Unable to read audio duration: {result.stderr.strip()[-500:]}")
        try:
            duration_ms = round(float(result.stdout.strip()) * 1000)
        except ValueError as error:
            raise ValueError("Unable to parse audio duration") from error
        if duration_ms <= 0:
            raise ValueError("Audio duration must be greater than zero")
        return duration_ms
