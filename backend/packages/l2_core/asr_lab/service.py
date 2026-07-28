from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

from fastapi import UploadFile
from sqlalchemy import Connection, Engine, text
from sqlalchemy.exc import IntegrityError

from l1_foundation.infrastructure.storage.local import LocalStorage
from l2_core.access.recordings import RecordingAccessService
from l2_core.auth.contracts import CurrentUser
from l2_core.evaluation.contracts import ApprovedAnnotation, DatasetVersionPreview
from l2_core.evaluation.datasets import build_dataset_preview

type DatabaseRow = dict[str, Any]


class AsrLabNotFoundError(LookupError):
    """Raised when a workspace-scoped ASR Lab resource does not exist."""


class AsrLabConflictError(ValueError):
    """Raised when a requested state transition is invalid or stale."""


class AsrLabPermissionError(PermissionError):
    """Raised when a workspace member lacks the required management role."""


class AsrLabService:
    """Workspace-scoped use cases for annotation, datasets, runs, and models."""

    def __init__(self, engine: Engine, storage: LocalStorage) -> None:
        self._engine = engine
        self._storage = storage
        self._recording_access = RecordingAccessService(engine)

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
                    left join evaluation_annotations annotations on annotations.dataset_id = datasets.id
                    left join evaluation_source_assets assets on assets.id = annotations.source_asset_id
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
                        where annotations.dataset_id = :dataset_id
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
        return {"dataset": dataset, "assets": assets, "annotations": annotations, "versions": versions}

    async def upload_asset(self, user: CurrentUser, dataset_id: UUID, upload: UploadFile) -> DatabaseRow:
        self._require_dataset(user, dataset_id)
        file_name = self._safe_file_name(upload.filename)
        asset_id = uuid4()
        storage_key = f"asr-lab/assets/{asset_id}/{file_name}"
        destination = self._storage.resolve(storage_key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            file_size_bytes, checksum = await self._write_upload(upload, destination)
            duration_ms = await self._probe_duration_ms(destination)
            with self._engine.begin() as connection:
                asset = connection.execute(
                    text(
                        """
                        insert into evaluation_source_assets (
                            id, workspace_id, dataset_id, artifact_uri, checksum, file_name, mime_type,
                            file_size_bytes, duration_ms, created_by_user_id
                        )
                        values (
                            :id, :workspace_id, :dataset_id, :artifact_uri, :checksum, :file_name, :mime_type,
                            :file_size_bytes, :duration_ms, :user_id
                        )
                        returning *
                        """
                    ),
                    {
                        "id": asset_id,
                        "workspace_id": user.current_workspace_id,
                        "dataset_id": dataset_id,
                        "artifact_uri": storage_key,
                        "checksum": checksum,
                        "file_name": file_name,
                        "mime_type": upload.content_type or "application/octet-stream",
                        "file_size_bytes": file_size_bytes,
                        "duration_ms": duration_ms,
                        "user_id": user.id,
                    },
                ).mappings().one()
                result = dict(asset)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()
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
    ) -> DatabaseRow:
        cleaned_text = reference_text.strip()
        if not cleaned_text:
            raise ValueError("Reference text is required")
        with self._engine.begin() as connection:
            self._dataset_row(connection, user, dataset_id)
            duration_ms = self._asset_duration(connection, user, dataset_id, source_asset_id)
            self._validate_interval(start_ms, end_ms, duration_ms)
            row = connection.execute(
                text(
                    """
                    insert into evaluation_annotations (
                        dataset_id, source_asset_id, start_ms, end_ms, reference_text,
                        language, train_allowed, evaluation_allowed, contains_sensitive_data,
                        group_key, created_by_user_id
                    )
                    values (
                        :dataset_id, :source_asset_id, :start_ms, :end_ms, :reference_text,
                        :language, :train_allowed, :evaluation_allowed, :contains_sensitive_data,
                        :group_key, :user_id
                    )
                    returning *
                    """
                ),
                {
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
        return dict(row)

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
    ) -> DatabaseRow:
        cleaned_text = reference_text.strip()
        if not cleaned_text:
            raise ValueError("Reference text is required")
        with self._engine.begin() as connection:
            existing = self._annotation_row(connection, user, annotation_id)
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
        return dict(row)

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
                      and (annotations.train_allowed or annotations.evaluation_allowed)
                    order by annotations.group_key, annotations.start_ms, annotations.id
                    """
                ),
                {"dataset_id": dataset_id},
            ).mappings().all()
            total_count = cast(
                int,
                connection.execute(
                    text("select count(*) from evaluation_annotations where dataset_id = :dataset_id"),
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
        return build_dataset_preview(
            annotations,
            normalization_name=normalization_name,
            normalization_version=normalization_version,
            seed=seed,
            excluded_count=total_count - len(annotations),
        )

    def freeze_dataset_version(
        self,
        user: CurrentUser,
        dataset_id: UUID,
        *,
        normalization_name: str,
        normalization_version: str,
        seed: str,
    ) -> DatabaseRow:
        self._require_manager(user)
        preview = self.preview_dataset_version(
            user,
            dataset_id,
            normalization_name=normalization_name,
            normalization_version=normalization_version,
            seed=seed,
        )
        strategy = json.dumps(
            {"name": "deterministic_group_hash_v1", "seed": seed, "ratios": {"train": 80, "validation": 10, "test": 10}},
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
        dataset_version_id: UUID,
        base_model_version_id: UUID,
        preset_name: str,
        candidate_model_name: str,
        idempotency_key: str,
    ) -> DatabaseRow:
        self._require_manager(user)
        with self._engine.begin() as connection:
            self._require_frozen_version(connection, user, dataset_version_id)
            train_count = cast(
                int,
                connection.execute(
                    text("select count(*) from evaluation_cases where dataset_version_id = :version_id and split = 'train'"),
                    {"version_id": dataset_version_id},
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
                    "dataset_version_id": dataset_version_id,
                    "base_model_version_id": base_model_version_id,
                    "preset_name": preset_name.strip(),
                    "candidate_model_name": candidate_model_name.strip(),
                    "idempotency_key": idempotency_key.strip(),
                    "config_snapshot": json.dumps({"preset_name": preset_name.strip(), "training_method": "lora"}),
                    "user_id": user.id,
                },
            ).mappings().one()
        return dict(row)

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
                        {"normalization_name": normalization_name, "normalization_version": normalization_version},
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
