from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast
from uuid import UUID

from sqlalchemy import Connection, Engine, text

from l1_foundation.settings import Settings
from l2_core.auth.contracts import CurrentUser
from l2_core.rag_evaluation.service import (
    RagEvaluationConflictError,
    RagEvaluationNotFoundError,
    RagEvaluationService,
)


class RagAdjudicationEvaluationService:
    """Workspace-scoped datasets and runs for adjudication component evaluation."""

    def __init__(self, engine: Engine, settings: Settings) -> None:
        self._engine = engine
        self._settings = settings
        self._search = RagEvaluationService(engine, settings)

    def list_datasets(self, user: CurrentUser) -> list[dict[str, Any]]:
        with self._engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select datasets.*, count(distinct drafts.id) as case_count,
                               count(distinct versions.id) as version_count,
                               max(versions.version_number) as latest_version_number
                        from evaluation_datasets datasets
                        left join rag_adjudication_evaluation_case_drafts drafts
                          on drafts.dataset_id = datasets.id
                        left join evaluation_dataset_versions versions
                          on versions.dataset_id = datasets.id
                        where datasets.workspace_id = :workspace_id
                          and datasets.task_type = 'rag_adjudication'
                          and datasets.status = 'active'
                        group by datasets.id
                        order by datasets.updated_at desc
                        """
                    ),
                    {"workspace_id": user.current_workspace_id},
                ).mappings()
            ]

    def create_dataset(self, user: CurrentUser, name: str, description: str | None) -> dict[str, Any]:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Dataset name is required")
        with self._engine.begin() as connection:
            return dict(
                connection.execute(
                    text(
                        """
                        insert into evaluation_datasets (
                            workspace_id, name, description, task_type, created_by_user_id
                        ) values (
                            :workspace_id, :name, :description, 'rag_adjudication', :user_id
                        ) returning *
                        """
                    ),
                    {
                        "workspace_id": user.current_workspace_id,
                        "name": clean_name,
                        "description": description.strip() if description and description.strip() else None,
                        "user_id": user.id,
                    },
                )
                .mappings()
                .one()
            )

    def get_dataset(self, user: CurrentUser, dataset_id: UUID) -> dict[str, Any]:
        with self._engine.connect() as connection:
            dataset = self._require_dataset(connection, user, dataset_id)
            cases = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select * from rag_adjudication_evaluation_case_drafts
                        where dataset_id = :dataset_id order by updated_at desc, id
                        """
                    ),
                    {"dataset_id": dataset_id},
                ).mappings()
            ]
            for case in cases:
                evidence = [
                    dict(row)
                    for row in connection.execute(
                        text(
                            """
                            select * from rag_adjudication_evaluation_evidence_drafts
                            where case_draft_id = :case_id
                            order by case when role = 'target' then 0 else 1 end, position, id
                            """
                        ),
                        {"case_id": case["id"]},
                    ).mappings()
                ]
                for item in evidence:
                    item["corrections"] = [
                        dict(row)
                        for row in connection.execute(
                            text(
                                """
                                select * from rag_adjudication_evaluation_correction_drafts
                                where target_evidence_draft_id = :evidence_id
                                order by start_char, end_char
                                """
                            ),
                            {"evidence_id": item["id"]},
                        ).mappings()
                    ]
                case["evidence"] = evidence
            versions = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select * from evaluation_dataset_versions
                        where dataset_id = :dataset_id order by version_number desc
                        """
                    ),
                    {"dataset_id": dataset_id},
                ).mappings()
            ]
            return {"dataset": dataset, "cases": cases, "versions": versions}

    def create_case(self, user: CurrentUser, dataset_id: UUID, query: str, tags: list[str] | None = None) -> dict[str, Any]:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query is required")
        clean_tags = list(dict.fromkeys(item.strip() for item in tags or [] if item.strip()))
        with self._engine.begin() as connection:
            self._require_dataset(connection, user, dataset_id)
            row = (
                connection.execute(
                    text(
                        """
                    insert into rag_adjudication_evaluation_case_drafts (
                        dataset_id, query, tags, created_by_user_id
                    ) values (:dataset_id, :query, cast(:tags as text[]), :user_id)
                    returning *
                    """
                    ),
                    {"dataset_id": dataset_id, "query": clean_query, "tags": clean_tags, "user_id": user.id},
                )
                .mappings()
                .one()
            )
            self._touch_dataset(connection, dataset_id)
            return {**dict(row), "evidence": []}

    def delete_case(self, user: CurrentUser, case_id: UUID) -> None:
        with self._engine.begin() as connection:
            case = self._require_case(connection, user, case_id, for_update=True)
            connection.execute(
                text("delete from rag_adjudication_evaluation_case_drafts where id=:id"),
                {"id": case_id},
            )
            self._touch_dataset(connection, cast(UUID, case["dataset_id"]))

    def update_case(
        self,
        user: CurrentUser,
        case_id: UUID,
        *,
        query: str,
        tags: list[str],
        revision: int,
    ) -> dict[str, Any]:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query is required")
        clean_tags = list(dict.fromkeys(item.strip() for item in tags if item.strip()))
        with self._engine.begin() as connection:
            case = self._require_case(connection, user, case_id, for_update=True)
            if int(case["revision"]) != revision:
                raise RagEvaluationConflictError("Case changed; refresh and retry")
            return dict(
                connection.execute(
                    text(
                        """
                        update rag_adjudication_evaluation_case_drafts
                        set query=:query, tags=cast(:tags as text[]), status='draft',
                            reviewed_by_user_id=null, reviewed_at=null,
                            approved_by_user_id=null, approved_at=null,
                            revision=revision+1, updated_at=now()
                        where id=:id returning *
                        """
                    ),
                    {"id": case_id, "query": clean_query, "tags": clean_tags},
                )
                .mappings()
                .one()
            )

    def add_evidence(self, user: CurrentUser, case_id: UUID, chunk_id: UUID, role: str, position: int) -> dict[str, Any]:
        if role not in {"target", "reference"}:
            raise ValueError("Evidence role must be target or reference")
        if position < 0:
            raise ValueError("Evidence position must be non-negative")
        with self._engine.begin() as connection:
            case = self._require_case(connection, user, case_id, for_update=True)
            if role == "target":
                target_count = cast(
                    int,
                    connection.execute(
                        text(
                            """
                            select count(*) from rag_adjudication_evaluation_evidence_drafts
                            where case_draft_id = :case_id and role = 'target'
                            """
                        ),
                        {"case_id": case_id},
                    ).scalar_one(),
                )
                if target_count >= 2:
                    raise RagEvaluationConflictError("A case can contain at most two target evidence items")
            chunk = (
                connection.execute(
                    text(
                        """
                    select chunks.*, recordings.title as recording_title,
                           recordings.file_name as recording_file_name
                    from recording_search_chunks chunks
                    join recordings on recordings.id = chunks.recording_id
                    where chunks.id = :chunk_id
                      and recordings.workspace_id = :workspace_id
                      and recordings.status = 'completed'
                    """
                    ),
                    {"chunk_id": chunk_id, "workspace_id": user.current_workspace_id},
                )
                .mappings()
                .one_or_none()
            )
            if chunk is None:
                raise RagEvaluationNotFoundError("Search chunk not found")
            row = (
                connection.execute(
                    text(
                        """
                    insert into rag_adjudication_evaluation_evidence_drafts (
                        case_draft_id, role, position, source_recording_id, source_chunk_id,
                        recording_title, recording_file_name, chunk_index, text,
                        start_ms, end_ms, metadata, content_checksum
                    ) values (
                        :case_id, :role, :position, :recording_id, :chunk_id,
                        :recording_title, :recording_file_name, :chunk_index, :text,
                        :start_ms, :end_ms, cast(:metadata as jsonb), :checksum
                    ) returning *
                    """
                    ),
                    {
                        "case_id": case_id,
                        "role": role,
                        "position": position,
                        "recording_id": chunk["recording_id"],
                        "chunk_id": chunk_id,
                        "recording_title": chunk["recording_title"],
                        "recording_file_name": chunk["recording_file_name"],
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                        "start_ms": chunk["start_ms"],
                        "end_ms": chunk["end_ms"],
                        "metadata": _json(chunk["metadata"] if isinstance(chunk["metadata"], Mapping) else {}),
                        "checksum": _checksum([chunk["text"], chunk["start_ms"], chunk["end_ms"]]),
                    },
                )
                .mappings()
                .one()
            )
            self._return_to_draft(connection, case_id)
            self._touch_dataset(connection, cast(UUID, case["dataset_id"]))
            return {**dict(row), "corrections": []}

    def update_evidence(self, user: CurrentUser, evidence_id: UUID, *, role: str, position: int) -> dict[str, Any]:
        if role not in {"target", "reference"} or position < 0:
            raise ValueError("Invalid evidence role or position")
        with self._engine.begin() as connection:
            evidence = self._require_evidence(connection, user, evidence_id, for_update=True)
            if role == "target" and evidence["role"] != "target":
                count = cast(
                    int,
                    connection.execute(
                        text(
                            """
                            select count(*) from rag_adjudication_evaluation_evidence_drafts
                            where case_draft_id = :case_id and role = 'target'
                            """
                        ),
                        {"case_id": evidence["case_draft_id"]},
                    ).scalar_one(),
                )
                if count >= 2:
                    raise RagEvaluationConflictError("A case can contain at most two target evidence items")
            if role == "reference" and evidence["role"] == "target":
                correction_count = cast(
                    int,
                    connection.execute(
                        text(
                            """
                            select count(*) from rag_adjudication_evaluation_correction_drafts
                            where target_evidence_draft_id = :id
                            """
                        ),
                        {"id": evidence_id},
                    ).scalar_one(),
                )
                if correction_count:
                    raise RagEvaluationConflictError("Delete target corrections before changing its role")
            row = (
                connection.execute(
                    text(
                        """
                    update rag_adjudication_evaluation_evidence_drafts
                    set role = :role, position = :position
                    where id = :id returning *
                    """
                    ),
                    {"id": evidence_id, "role": role, "position": position},
                )
                .mappings()
                .one()
            )
            self._return_to_draft(connection, cast(UUID, evidence["case_draft_id"]))
            return dict(row)

    def delete_evidence(self, user: CurrentUser, evidence_id: UUID) -> None:
        with self._engine.begin() as connection:
            evidence = self._require_evidence(connection, user, evidence_id, for_update=True)
            connection.execute(
                text("delete from rag_adjudication_evaluation_evidence_drafts where id = :id"),
                {"id": evidence_id},
            )
            self._return_to_draft(connection, cast(UUID, evidence["case_draft_id"]))

    def add_correction(
        self,
        user: CurrentUser,
        target_evidence_id: UUID,
        *,
        start_char: int,
        end_char: int,
        original_expression: str,
        accepted_expressions: list[str],
    ) -> dict[str, Any]:
        clean_accepted = list(dict.fromkeys(value.strip() for value in accepted_expressions if value.strip()))
        if not clean_accepted:
            raise ValueError("At least one accepted expression is required")
        with self._engine.begin() as connection:
            evidence = self._require_evidence(connection, user, target_evidence_id, for_update=True)
            if evidence["role"] != "target":
                raise RagEvaluationConflictError("Corrections can only be added to target evidence")
            source = str(evidence["text"])
            if start_char < 0 or end_char <= start_char or end_char > len(source):
                raise ValueError("Correction span is out of bounds")
            if source[start_char:end_char] != original_expression:
                raise ValueError("Correction span does not match the target evidence text")
            overlap = cast(
                int,
                connection.execute(
                    text(
                        """
                        select count(*) from rag_adjudication_evaluation_correction_drafts
                        where target_evidence_draft_id = :id
                          and start_char < :end_char and :start_char < end_char
                        """
                    ),
                    {"id": target_evidence_id, "start_char": start_char, "end_char": end_char},
                ).scalar_one(),
            )
            if overlap:
                raise RagEvaluationConflictError("Correction spans cannot overlap")
            row = (
                connection.execute(
                    text(
                        """
                    insert into rag_adjudication_evaluation_correction_drafts (
                        target_evidence_draft_id, start_char, end_char,
                        original_expression, accepted_expressions
                    ) values (
                        :id, :start_char, :end_char, :original, cast(:accepted as text[])
                    ) returning *
                    """
                    ),
                    {
                        "id": target_evidence_id,
                        "start_char": start_char,
                        "end_char": end_char,
                        "original": original_expression,
                        "accepted": clean_accepted,
                    },
                )
                .mappings()
                .one()
            )
            self._return_to_draft(connection, cast(UUID, evidence["case_draft_id"]))
            return dict(row)

    def delete_correction(self, user: CurrentUser, correction_id: UUID) -> None:
        with self._engine.begin() as connection:
            row = (
                connection.execute(
                    text(
                        """
                    select corrections.id, evidence.case_draft_id
                    from rag_adjudication_evaluation_correction_drafts corrections
                    join rag_adjudication_evaluation_evidence_drafts evidence
                      on evidence.id = corrections.target_evidence_draft_id
                    join rag_adjudication_evaluation_case_drafts cases
                      on cases.id = evidence.case_draft_id
                    join evaluation_datasets datasets on datasets.id = cases.dataset_id
                    where corrections.id = :id and datasets.workspace_id = :workspace_id
                    for update
                    """
                    ),
                    {"id": correction_id, "workspace_id": user.current_workspace_id},
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                raise RagEvaluationNotFoundError("Correction not found")
            connection.execute(
                text("delete from rag_adjudication_evaluation_correction_drafts where id = :id"),
                {"id": correction_id},
            )
            self._return_to_draft(connection, cast(UUID, row["case_draft_id"]))

    def update_correction(
        self,
        user: CurrentUser,
        correction_id: UUID,
        *,
        start_char: int,
        end_char: int,
        original_expression: str,
        accepted_expressions: list[str],
    ) -> dict[str, Any]:
        clean_accepted = list(dict.fromkeys(value.strip() for value in accepted_expressions if value.strip()))
        if not clean_accepted:
            raise ValueError("At least one accepted expression is required")
        with self._engine.begin() as connection:
            row = (
                connection.execute(
                    text(
                        """
                    select corrections.*, evidence.text, evidence.case_draft_id
                    from rag_adjudication_evaluation_correction_drafts corrections
                    join rag_adjudication_evaluation_evidence_drafts evidence
                      on evidence.id=corrections.target_evidence_draft_id
                    join rag_adjudication_evaluation_case_drafts cases on cases.id=evidence.case_draft_id
                    join evaluation_datasets datasets on datasets.id=cases.dataset_id
                    where corrections.id=:id and datasets.workspace_id=:workspace_id
                    for update
                    """
                    ),
                    {"id": correction_id, "workspace_id": user.current_workspace_id},
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                raise RagEvaluationNotFoundError("Correction not found")
            source = str(row["text"])
            if start_char < 0 or end_char <= start_char or end_char > len(source):
                raise ValueError("Correction span is out of bounds")
            if source[start_char:end_char] != original_expression:
                raise ValueError("Correction span does not match the target evidence text")
            overlap = cast(
                int,
                connection.execute(
                    text(
                        """
                        select count(*) from rag_adjudication_evaluation_correction_drafts
                        where target_evidence_draft_id=:evidence_id and id<>:id
                          and start_char<:end_char and :start_char<end_char
                        """
                    ),
                    {
                        "evidence_id": row["target_evidence_draft_id"],
                        "id": correction_id,
                        "start_char": start_char,
                        "end_char": end_char,
                    },
                ).scalar_one(),
            )
            if overlap:
                raise RagEvaluationConflictError("Correction spans cannot overlap")
            updated = (
                connection.execute(
                    text(
                        """
                    update rag_adjudication_evaluation_correction_drafts
                    set start_char=:start_char, end_char=:end_char,
                        original_expression=:original,
                        accepted_expressions=cast(:accepted as text[]), updated_at=now()
                    where id=:id returning *
                    """
                    ),
                    {
                        "id": correction_id,
                        "start_char": start_char,
                        "end_char": end_char,
                        "original": original_expression,
                        "accepted": clean_accepted,
                    },
                )
                .mappings()
                .one()
            )
            self._return_to_draft(connection, cast(UUID, row["case_draft_id"]))
            return dict(updated)

    def transition_case(self, user: CurrentUser, case_id: UUID, revision: int, action: str) -> dict[str, Any]:
        with self._engine.begin() as connection:
            case = self._require_case(connection, user, case_id, for_update=True)
            if int(case["revision"]) != revision:
                raise RagEvaluationConflictError("Case changed; refresh and retry")
            self._validate_case(connection, case_id)
            if action == "review" and case["status"] == "draft":
                sql = """
                    update rag_adjudication_evaluation_case_drafts
                    set status='reviewed', reviewed_by_user_id=:user_id, reviewed_at=now(),
                        revision=revision+1, updated_at=now()
                    where id=:id returning *
                """
            elif action == "approve" and case["status"] == "reviewed":
                sql = """
                    update rag_adjudication_evaluation_case_drafts
                    set status='approved', approved_by_user_id=:user_id, approved_at=now(),
                        revision=revision+1, updated_at=now()
                    where id=:id returning *
                """
            else:
                raise RagEvaluationConflictError(f"Cannot {action} case in {case['status']} status")
            return dict(connection.execute(text(sql), {"id": case_id, "user_id": user.id}).mappings().one())

    def preview_version(self, user: CurrentUser, dataset_id: UUID) -> dict[str, Any]:
        with self._engine.connect() as connection:
            self._require_dataset(connection, user, dataset_id)
            payload = self._approved_payload(connection, dataset_id)
            return {
                "case_count": len(payload),
                "target_count": sum(sum(e["role"] == "target" for e in c["evidence"]) for c in payload),
                "correction_count": sum(sum(len(e["corrections"]) for e in c["evidence"]) for c in payload),
                "checksum": _checksum(payload),
            }

    def freeze_version(self, user: CurrentUser, dataset_id: UUID, expected_checksum: str) -> dict[str, Any]:
        with self._engine.begin() as connection:
            self._require_dataset(connection, user, dataset_id, for_update=True)
            payload = self._approved_payload(connection, dataset_id)
            checksum = _checksum(payload)
            if checksum != expected_checksum:
                raise RagEvaluationConflictError("Annotations changed; preview again")
            number = cast(
                int,
                connection.execute(
                    text(
                        """
                        select coalesce(max(version_number), 0) + 1
                        from evaluation_dataset_versions where dataset_id = :id
                        """
                    ),
                    {"id": dataset_id},
                ).scalar_one(),
            )
            version = (
                connection.execute(
                    text(
                        """
                    insert into evaluation_dataset_versions (
                        dataset_id, version_number, status, normalization_name,
                        normalization_version, definition_snapshot, split_strategy,
                        case_count, created_by_user_id
                    ) values (
                        :dataset_id, :number, 'building', 'adjudication_expression',
                        'v1', cast(:definition as jsonb), '{"name":"all_test_v1"}'::jsonb,
                        :count, :user_id
                    ) returning *
                    """
                    ),
                    {
                        "dataset_id": dataset_id,
                        "number": number,
                        "definition": _json({"task_type": "rag_adjudication", "metric": "gold_correction_accuracy_v1"}),
                        "count": len(payload),
                        "user_id": user.id,
                    },
                )
                .mappings()
                .one()
            )
            for case in payload:
                case_id = cast(
                    UUID,
                    connection.execute(
                        text(
                            """
                            insert into rag_adjudication_evaluation_cases (
                                dataset_version_id, query, tags
                            ) values (:version, :query, cast(:tags as text[]))
                            returning id
                            """
                        ),
                        {
                            "version": version["id"],
                            "query": case["query"],
                            "tags": case["tags"],
                        },
                    ).scalar_one(),
                )
                evidence_map: dict[UUID, UUID] = {}
                for evidence in case["evidence"]:
                    frozen_id = cast(
                        UUID,
                        connection.execute(
                            text(
                                """
                                insert into rag_adjudication_evaluation_evidence (
                                    evaluation_case_id, role, position,
                                    source_recording_id, source_chunk_id, recording_title,
                                    recording_file_name, chunk_index, text, start_ms, end_ms,
                                    metadata, content_checksum
                                ) values (
                                    :case_id, :role, :position, :recording_id,
                                    :chunk_id, :title, :file_name, :chunk_index, :text,
                                    :start_ms, :end_ms, cast(:metadata as jsonb), :checksum
                                ) returning id
                                """
                            ),
                            {
                                "case_id": case_id,
                                "role": evidence["role"],
                                "position": evidence["position"],
                                "recording_id": evidence["source_recording_id"],
                                "chunk_id": evidence["source_chunk_id"],
                                "title": evidence["recording_title"],
                                "file_name": evidence["recording_file_name"],
                                "chunk_index": evidence["chunk_index"],
                                "text": evidence["text"],
                                "start_ms": evidence["start_ms"],
                                "end_ms": evidence["end_ms"],
                                "metadata": _json(evidence["metadata"]),
                                "checksum": evidence["content_checksum"],
                            },
                        ).scalar_one(),
                    )
                    evidence_map[cast(UUID, evidence["id"])] = frozen_id
                    for correction in evidence["corrections"]:
                        connection.execute(
                            text(
                                """
                                insert into rag_adjudication_evaluation_corrections (
                                    target_evidence_id, start_char, end_char,
                                    original_expression, accepted_expressions
                                ) values (
                                    :evidence_id, :start_char, :end_char,
                                    :original, cast(:accepted as text[])
                                )
                                """
                            ),
                            {
                                "evidence_id": frozen_id,
                                "start_char": correction["start_char"],
                                "end_char": correction["end_char"],
                                "original": correction["original_expression"],
                                "accepted": correction["accepted_expressions"],
                            },
                        )
            return dict(
                connection.execute(
                    text(
                        """
                        update evaluation_dataset_versions
                        set status='frozen', checksum=:checksum, frozen_at=now()
                        where id=:id returning *
                        """
                    ),
                    {"id": version["id"], "checksum": checksum},
                )
                .mappings()
                .one()
            )

    def create_run(self, user: CurrentUser, dataset_version_id: UUID, idempotency_key: str) -> dict[str, Any]:
        clean_key = idempotency_key.strip()
        if not clean_key:
            raise ValueError("Idempotency key is required")
        config = self._config_snapshot()
        with self._engine.begin() as connection:
            version = (
                connection.execute(
                    text(
                        """
                    select versions.*, datasets.workspace_id, datasets.task_type
                    from evaluation_dataset_versions versions
                    join evaluation_datasets datasets on datasets.id = versions.dataset_id
                    where versions.id=:id and datasets.workspace_id=:workspace_id
                    """
                    ),
                    {"id": dataset_version_id, "workspace_id": user.current_workspace_id},
                )
                .mappings()
                .one_or_none()
            )
            if version is None:
                raise RagEvaluationNotFoundError("Dataset version not found")
            if version["status"] != "frozen" or version["task_type"] != "rag_adjudication":
                raise RagEvaluationConflictError("Only frozen adjudication versions can be evaluated")
            existing = (
                connection.execute(
                    text("select * from evaluation_runs where workspace_id=:workspace_id and idempotency_key=:key"),
                    {"workspace_id": user.current_workspace_id, "key": clean_key},
                )
                .mappings()
                .one_or_none()
            )
            if existing is not None:
                return dict(existing)
            count = cast(
                int,
                connection.execute(
                    text(
                        """
                        select count(*) from rag_adjudication_evaluation_cases
                        where dataset_version_id=:id
                        """
                    ),
                    {"id": dataset_version_id},
                ).scalar_one(),
            )
            run = (
                connection.execute(
                    text(
                        """
                    insert into evaluation_runs (
                        workspace_id, dataset_version_id, evaluator_type, split, status,
                        idempotency_key, config_snapshot, total_case_count, created_by_user_id
                    ) values (
                        :workspace_id, :version, 'rag_adjudication', 'test', 'queued',
                        :key, cast(:config as jsonb), :count, :user_id
                    ) returning *
                    """
                    ),
                    {
                        "workspace_id": user.current_workspace_id,
                        "version": dataset_version_id,
                        "key": clean_key,
                        "config": _json(config),
                        "count": count,
                        "user_id": user.id,
                    },
                )
                .mappings()
                .one()
            )
            connection.execute(
                text(
                    """
                    insert into rag_adjudication_evaluation_run_specs (
                        evaluation_run_id, config_snapshot
                    ) values (:id, cast(:config as jsonb))
                    """
                ),
                {"id": run["id"], "config": _json(config)},
            )
            return dict(run)

    def list_runs(self, user: CurrentUser) -> list[dict[str, Any]]:
        with self._engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select runs.*, datasets.name as dataset_name, versions.version_number
                        from evaluation_runs runs
                        join evaluation_dataset_versions versions on versions.id=runs.dataset_version_id
                        join evaluation_datasets datasets on datasets.id=versions.dataset_id
                        where runs.workspace_id=:workspace_id
                          and runs.evaluator_type='rag_adjudication'
                        order by runs.created_at desc
                        """
                    ),
                    {"workspace_id": user.current_workspace_id},
                ).mappings()
            ]

    def get_run(self, user: CurrentUser, run_id: UUID) -> dict[str, Any]:
        with self._engine.connect() as connection:
            run = (
                connection.execute(
                    text(
                        """
                    select runs.*, datasets.name as dataset_name, versions.version_number
                    from evaluation_runs runs
                    join evaluation_dataset_versions versions on versions.id=runs.dataset_version_id
                    join evaluation_datasets datasets on datasets.id=versions.dataset_id
                    where runs.id=:id and runs.workspace_id=:workspace_id
                      and runs.evaluator_type='rag_adjudication'
                    """
                    ),
                    {"id": run_id, "workspace_id": user.current_workspace_id},
                )
                .mappings()
                .one_or_none()
            )
            if run is None:
                raise RagEvaluationNotFoundError("Evaluation run not found")
            results = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select results.*, cases.query
                        from rag_adjudication_evaluation_case_results results
                        join rag_adjudication_evaluation_cases cases
                          on cases.id=results.evaluation_case_id
                        where results.evaluation_run_id=:id order by cases.id
                        """
                    ),
                    {"id": run_id},
                ).mappings()
            ]
            for result in results:
                result["corrections"] = [
                    dict(row)
                    for row in connection.execute(
                        text(
                            """
                            select scored.*, gold.start_char, gold.end_char,
                                   gold.original_expression, gold.accepted_expressions
                            from rag_adjudication_evaluation_correction_results scored
                            join rag_adjudication_evaluation_corrections gold
                              on gold.id=scored.gold_correction_id
                            where scored.case_result_id=:id order by gold.id
                            """
                        ),
                        {"id": result["id"]},
                    ).mappings()
                ]
            metric = (
                connection.execute(
                    text(
                        """
                    select * from rag_adjudication_evaluation_metric_values
                    where evaluation_run_id=:id and metric_name='gold_correction_accuracy'
                    """
                    ),
                    {"id": run_id},
                )
                .mappings()
                .one_or_none()
            )
            return {"run": dict(run), "metric": dict(metric) if metric else None, "cases": results}

    def cancel_run(self, user: CurrentUser, run_id: UUID) -> dict[str, Any]:
        with self._engine.begin() as connection:
            row = (
                connection.execute(
                    text(
                        """
                    update evaluation_runs set cancel_requested=true, updated_at=now()
                    where id=:id and workspace_id=:workspace_id
                      and evaluator_type='rag_adjudication'
                      and status in ('queued','running') returning *
                    """
                    ),
                    {"id": run_id, "workspace_id": user.current_workspace_id},
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                raise RagEvaluationConflictError("Run cannot be cancelled")
            return dict(row)

    def delete_run(self, user: CurrentUser, run_id: UUID) -> None:
        with self._engine.begin() as connection:
            result = connection.execute(
                text(
                    """
                    delete from evaluation_runs
                    where id=:id and workspace_id=:workspace_id
                      and evaluator_type='rag_adjudication'
                      and status in ('succeeded','failed','cancelled')
                    """
                ),
                {"id": run_id, "workspace_id": user.current_workspace_id},
            )
            if result.rowcount != 1:
                raise RagEvaluationConflictError("Only terminal runs can be deleted")

    def search_chunks(self, user: CurrentUser, **kwargs: Any) -> list[dict[str, Any]]:
        return self._search.search_chunks(user, **kwargs)

    def list_recordings(self, user: CurrentUser) -> list[dict[str, Any]]:
        return self._search.list_recordings(user)

    def _approved_payload(self, connection: Connection, dataset_id: UUID) -> list[dict[str, Any]]:
        cases = [
            dict(row)
            for row in connection.execute(
                text(
                    """
                    select * from rag_adjudication_evaluation_case_drafts
                    where dataset_id=:id and status='approved' order by id
                    """
                ),
                {"id": dataset_id},
            ).mappings()
        ]
        if not cases:
            raise RagEvaluationConflictError("At least one approved case is required")
        for case in cases:
            self._validate_case(connection, cast(UUID, case["id"]))
            case["evidence"] = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select * from rag_adjudication_evaluation_evidence_drafts
                        where case_draft_id=:id
                        order by case when role='target' then 0 else 1 end, position, id
                        """
                    ),
                    {"id": case["id"]},
                ).mappings()
            ]
            for evidence in case["evidence"]:
                evidence["corrections"] = [
                    dict(row)
                    for row in connection.execute(
                        text(
                            """
                            select * from rag_adjudication_evaluation_correction_drafts
                            where target_evidence_draft_id=:id order by start_char
                            """
                        ),
                        {"id": evidence["id"]},
                    ).mappings()
                ]
        return cases

    @staticmethod
    def _validate_case(connection: Connection, case_id: UUID) -> None:
        evidence = [
            dict(row)
            for row in connection.execute(
                text(
                    """
                    select * from rag_adjudication_evaluation_evidence_drafts
                    where case_draft_id=:id
                    """
                ),
                {"id": case_id},
            ).mappings()
        ]
        targets = [item for item in evidence if item["role"] == "target"]
        if not 1 <= len(targets) <= 2:
            raise RagEvaluationConflictError("A case requires one or two target evidence items")
        for target in targets:
            count = cast(
                int,
                connection.execute(
                    text(
                        """
                        select count(*) from rag_adjudication_evaluation_correction_drafts
                        where target_evidence_draft_id=:id
                        """
                    ),
                    {"id": target["id"]},
                ).scalar_one(),
            )
            if count == 0:
                raise RagEvaluationConflictError("Every target requires at least one gold correction")

    def _config_snapshot(self) -> dict[str, Any]:
        return {
            "evaluator_version": "1",
            "metric_version": "1",
            "provider": self._settings.rag_answer_provider,
            "context_size": self._settings.rag_context_size,
            "max_iterations": 4,
            "max_searches": 3,
            "web_search_enabled": self._settings.rag_asr_adjudication_web_search_enabled,
            "auto_resolve_confidence": self._settings.rag_asr_adjudication_auto_resolve_confidence,
            "audit_prompt_variant": self._settings.rag_asr_adjudication_audit_prompt_variant,
            "audit_model": self._settings.rag_asr_adjudication_audit_model,
        }

    @staticmethod
    def _return_to_draft(connection: Connection, case_id: UUID) -> None:
        connection.execute(
            text(
                """
                update rag_adjudication_evaluation_case_drafts
                set status='draft', reviewed_by_user_id=null, reviewed_at=null,
                    approved_by_user_id=null, approved_at=null,
                    revision=revision+1, updated_at=now()
                where id=:id
                """
            ),
            {"id": case_id},
        )

    @staticmethod
    def _touch_dataset(connection: Connection, dataset_id: UUID) -> None:
        connection.execute(
            text("update evaluation_datasets set updated_at=now() where id=:id"),
            {"id": dataset_id},
        )

    @staticmethod
    def _require_dataset(connection: Connection, user: CurrentUser, dataset_id: UUID, *, for_update: bool = False) -> dict[str, Any]:
        suffix = " for update" if for_update else ""
        row = (
            connection.execute(
                text(
                    """
                select * from evaluation_datasets
                where id=:id and workspace_id=:workspace_id
                  and task_type='rag_adjudication' and status='active'
                """
                    + suffix
                ),
                {"id": dataset_id, "workspace_id": user.current_workspace_id},
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise RagEvaluationNotFoundError("Adjudication dataset not found")
        return dict(row)

    @staticmethod
    def _require_case(connection: Connection, user: CurrentUser, case_id: UUID, *, for_update: bool = False) -> dict[str, Any]:
        suffix = " for update" if for_update else ""
        row = (
            connection.execute(
                text(
                    """
                select cases.* from rag_adjudication_evaluation_case_drafts cases
                join evaluation_datasets datasets on datasets.id=cases.dataset_id
                where cases.id=:id and datasets.workspace_id=:workspace_id
                  and datasets.task_type='rag_adjudication'
                """
                    + suffix
                ),
                {"id": case_id, "workspace_id": user.current_workspace_id},
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise RagEvaluationNotFoundError("Adjudication case not found")
        return dict(row)

    @staticmethod
    def _require_evidence(connection: Connection, user: CurrentUser, evidence_id: UUID, *, for_update: bool = False) -> dict[str, Any]:
        suffix = " for update" if for_update else ""
        row = (
            connection.execute(
                text(
                    """
                select evidence.* from rag_adjudication_evaluation_evidence_drafts evidence
                join rag_adjudication_evaluation_case_drafts cases on cases.id=evidence.case_draft_id
                join evaluation_datasets datasets on datasets.id=cases.dataset_id
                where evidence.id=:id and datasets.workspace_id=:workspace_id
                  and datasets.task_type='rag_adjudication'
                """
                    + suffix
                ),
                {"id": evidence_id, "workspace_id": user.current_workspace_id},
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise RagEvaluationNotFoundError("Adjudication evidence not found")
        return dict(row)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _checksum(value: object) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()
