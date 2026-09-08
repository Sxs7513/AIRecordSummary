from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, cast
from uuid import UUID

from sqlalchemy import Connection, Engine, text

from l1_foundation.llm import LlmProvider, create_language_model_from_settings
from l1_foundation.model_ref import OnlineModelRef
from l1_foundation.settings import Settings
from l2_core.auth.contracts import CurrentUser
from l2_core.rag.normalization import normalize_search_text
from l2_core.rag_evaluation.answer_annotations import AnswerAnnotation
from l2_core.rag_evaluation.answer_evaluator import AnswerAnnotationGenerator
from l2_core.rag_evaluation.diagnosis import summarize_journeys

logger = logging.getLogger("evaluation")


class RagEvaluationNotFoundError(LookupError):
    pass


class RagEvaluationConflictError(RuntimeError):
    pass


class RagEvaluationPermissionError(PermissionError):
    pass


class RagEvaluationService:
    """Workspace-scoped annotation, versioning and run control for retrieval evaluation."""

    def __init__(self, engine: Engine, settings: Settings) -> None:
        self._engine = engine
        self._settings = settings

    def list_datasets(self, user: CurrentUser) -> list[dict[str, Any]]:
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    select datasets.*,
                           count(distinct drafts.id) as case_count,
                           count(distinct versions.id) as version_count,
                           max(versions.version_number) as latest_version_number
                    from evaluation_datasets datasets
                    left join rag_evaluation_case_drafts drafts on drafts.dataset_id = datasets.id
                    left join evaluation_dataset_versions versions on versions.dataset_id = datasets.id
                    where datasets.workspace_id = :workspace_id
                      and datasets.task_type = 'rag_retrieval'
                      and datasets.status = 'active'
                    group by datasets.id
                    order by datasets.updated_at desc
                    """
                ),
                {"workspace_id": user.current_workspace_id},
            ).mappings()
            return [dict(row) for row in rows]

    def create_dataset(self, user: CurrentUser, name: str, description: str | None) -> dict[str, Any]:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Dataset name is required")
        with self._engine.begin() as connection:
            row = (
                connection.execute(
                    text(
                        """
                    insert into evaluation_datasets (
                        workspace_id, name, description, task_type, created_by_user_id
                    ) values (
                        :workspace_id, :name, :description, 'rag_retrieval', :user_id
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
            return dict(row)

    def get_dataset(self, user: CurrentUser, dataset_id: UUID) -> dict[str, Any]:
        with self._engine.connect() as connection:
            dataset = self._require_dataset(connection, user, dataset_id)
            drafts = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select drafts.*
                        from rag_evaluation_case_drafts drafts
                        where drafts.dataset_id = :dataset_id
                        order by drafts.updated_at desc, drafts.id
                        """
                    ),
                    {"dataset_id": dataset_id},
                ).mappings()
            ]
            evidence_rows = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select evidence.*, recordings.title as recording_title,
                               recordings.file_name as recording_file_name
                        from rag_evaluation_evidence_drafts evidence
                        join rag_evaluation_case_drafts drafts on drafts.id = evidence.case_draft_id
                        join recordings on recordings.id = evidence.source_recording_id
                        where drafts.dataset_id = :dataset_id
                        order by evidence.case_draft_id, evidence.created_at
                        """
                    ),
                    {"dataset_id": dataset_id},
                ).mappings()
            ]
            evidence_by_case: dict[UUID, list[dict[str, Any]]] = {}
            for evidence in evidence_rows:
                evidence_by_case.setdefault(cast(UUID, evidence["case_draft_id"]), []).append(evidence)
            for draft in drafts:
                draft["evidence"] = evidence_by_case.get(cast(UUID, draft["id"]), [])
            versions = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select * from evaluation_dataset_versions
                        where dataset_id = :dataset_id
                        order by version_number desc
                        """
                    ),
                    {"dataset_id": dataset_id},
                ).mappings()
            ]
            return {"dataset": dataset, "cases": drafts, "versions": versions}

    def create_case(
        self,
        user: CurrentUser,
        dataset_id: UUID,
        *,
        query: str,
        recording_ids: list[UUID] | None = None,
        tags: list[str] | None = None,
        group_key: str | None = None,
    ) -> dict[str, Any]:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query is required")
        clean_tags = list(dict.fromkeys(item.strip() for item in tags or [] if item.strip()))
        scope = {"recording_ids": [str(item) for item in recording_ids or []]}
        stable_group = group_key.strip() if group_key and group_key.strip() else hashlib.sha256(clean_query.encode()).hexdigest()[:24]
        with self._engine.begin() as connection:
            self._require_dataset(connection, user, dataset_id)
            self._require_recordings(connection, user, recording_ids or [])
            row = (
                connection.execute(
                    text(
                        """
                    insert into rag_evaluation_case_drafts (
                        dataset_id, query, scope, tags, group_key, created_by_user_id
                    ) values (
                        :dataset_id, :query, cast(:scope as jsonb), cast(:tags as text[]),
                        :group_key, :user_id
                    ) returning *
                    """
                    ),
                    {
                        "dataset_id": dataset_id,
                        "query": clean_query,
                        "scope": _json(scope),
                        "tags": clean_tags,
                        "group_key": stable_group,
                        "user_id": user.id,
                    },
                )
                .mappings()
                .one()
            )
            connection.execute(
                text("update evaluation_datasets set updated_at = now() where id = :dataset_id"),
                {"dataset_id": dataset_id},
            )
            return {**dict(row), "evidence": []}

    def delete_case(self, user: CurrentUser, case_id: UUID) -> None:
        with self._engine.begin() as connection:
            draft = self._require_case_draft(connection, user, case_id, for_update=True)
            frozen_count = cast(
                int,
                connection.execute(
                    text("select count(*) from rag_evaluation_cases where source_draft_id = :case_id"),
                    {"case_id": case_id},
                ).scalar_one(),
            )
            if frozen_count:
                raise RagEvaluationConflictError("已冻结版本引用该问题，不能删除；请创建新的数据集版本")
            connection.execute(text("delete from rag_evaluation_case_drafts where id = :case_id"), {"case_id": case_id})
            connection.execute(
                text("update evaluation_datasets set updated_at = now() where id = :dataset_id"),
                {"dataset_id": draft["dataset_id"]},
            )

    def archive_case(self, user: CurrentUser, case_id: UUID) -> dict[str, Any]:
        """Exclude a draft from future versions without breaking frozen history."""
        with self._engine.begin() as connection:
            draft = self._require_case_draft(connection, user, case_id, for_update=True)
            row = (
                connection.execute(
                    text(
                        """
                    update rag_evaluation_case_drafts
                    set archived_by_user_id = :user_id, archived_at = now(), updated_at = now()
                    where id = :case_id and archived_at is null
                    returning *
                    """
                    ),
                    {"case_id": case_id, "user_id": user.id},
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                return dict(draft)
            connection.execute(
                text("update evaluation_datasets set updated_at = now() where id = :dataset_id"),
                {"dataset_id": draft["dataset_id"]},
            )
            return dict(row)

    def add_evidence(self, user: CurrentUser, case_id: UUID, chunk_id: UUID, relevance: int = 3) -> dict[str, Any]:
        if relevance not in {1, 2, 3}:
            raise ValueError("Evidence relevance must be between 1 and 3")
        with self._engine.begin() as connection:
            draft = self._require_case_draft(connection, user, case_id, for_update=True)
            chunk = (
                connection.execute(
                    text(
                        """
                    select chunks.*, recordings.title as recording_title,
                           recordings.file_name as recording_file_name
                    from recording_search_chunks chunks
                    join recordings on recordings.id = chunks.recording_id
                    where chunks.id = :chunk_id and recordings.workspace_id = :workspace_id
                    """
                    ),
                    {"chunk_id": chunk_id, "workspace_id": user.current_workspace_id},
                )
                .mappings()
                .one_or_none()
            )
            if chunk is None:
                raise RagEvaluationNotFoundError("Search chunk not found")
            checksum = _content_checksum(str(chunk["text"]), int(chunk["start_ms"]), int(chunk["end_ms"]))
            row = (
                connection.execute(
                    text(
                        """
                    insert into rag_evaluation_evidence_drafts (
                        case_draft_id, source_recording_id, source_chunk_id, quote,
                        start_ms, end_ms, relevance, content_checksum, metadata
                    ) values (
                        :case_id, :recording_id, :chunk_id, :quote,
                        :start_ms, :end_ms, :relevance, :checksum, cast(:metadata as jsonb)
                    )
                    on conflict (case_draft_id, source_recording_id, start_ms, end_ms, content_checksum)
                    do update set relevance = excluded.relevance
                    returning *
                    """
                    ),
                    {
                        "case_id": case_id,
                        "recording_id": chunk["recording_id"],
                        "chunk_id": chunk_id,
                        "quote": chunk["text"],
                        "start_ms": chunk["start_ms"],
                        "end_ms": chunk["end_ms"],
                        "relevance": relevance,
                        "checksum": checksum,
                        "metadata": _json(chunk["metadata"] if isinstance(chunk["metadata"], Mapping) else {}),
                    },
                )
                .mappings()
                .one()
            )
            self._return_case_to_draft(connection, case_id)
            connection.execute(
                text("update evaluation_datasets set updated_at = now() where id = :dataset_id"),
                {"dataset_id": draft["dataset_id"]},
            )
            return {
                **dict(row),
                "recording_title": chunk["recording_title"],
                "recording_file_name": chunk["recording_file_name"],
            }

    def delete_evidence(self, user: CurrentUser, evidence_id: UUID) -> None:
        with self._engine.begin() as connection:
            row = (
                connection.execute(
                    text(
                        """
                    select evidence.id, evidence.case_draft_id, drafts.dataset_id
                    from rag_evaluation_evidence_drafts evidence
                    join rag_evaluation_case_drafts drafts on drafts.id = evidence.case_draft_id
                    join evaluation_datasets datasets on datasets.id = drafts.dataset_id
                    where evidence.id = :evidence_id and datasets.workspace_id = :workspace_id
                    for update
                    """
                    ),
                    {"evidence_id": evidence_id, "workspace_id": user.current_workspace_id},
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                raise RagEvaluationNotFoundError("Evidence not found")
            connection.execute(text("delete from rag_evaluation_evidence_drafts where id = :id"), {"id": evidence_id})

    def suggest_answer_annotation(self, user: CurrentUser, case_id: UUID) -> dict[str, Any]:
        with self._engine.connect() as connection:
            draft = self._require_case_draft(connection, user, case_id)
            evidence = self._draft_evidence(connection, case_id)
        evidence_log = [
            {
                "evidence_id": str(item["id"]),
                "recording_id": str(item["source_recording_id"]),
                "chunk_id": str(item["source_chunk_id"]) if item["source_chunk_id"] is not None else None,
                "start_ms": int(item["start_ms"]),
                "end_ms": int(item["end_ms"]),
                "relevance": int(item["relevance"]),
            }
            for item in evidence
        ]
        logger.info(
            "RAG 答案标注：开始自动生成 case_id=%s query=%s evidence_count=%s evidences=%s",
            case_id,
            draft["query"],
            len(evidence_log),
            _json(evidence_log),
        )
        if not evidence:
            annotation = AnswerAnnotation(
                expected_verdict="abstain",
                reference_answer=None,
                key_points=[],
            )
            logger.info(
                "RAG 答案标注：无 Gold Evidence，生成拒答标注 case_id=%s annotation=%s",
                case_id,
                annotation.model_dump_json(),
            )
            return annotation.model_dump(mode="json")
        model_ref = OnlineModelRef.parse(self._settings.rag_answer_judge_model)
        model = create_language_model_from_settings(
            self._settings,
            LlmProvider(model_ref.provider),
            local_context_size=self._settings.rag_context_size,
        )
        try:
            annotation = AnswerAnnotationGenerator(
                model,
                model_ref,
                context_size=self._settings.rag_context_size,
                max_output_tokens=self._settings.rag_answer_judge_max_output_tokens,
            ).generate(
                query=str(draft["query"]),
                evidence=[{"id": str(item["id"]), "quote": str(item["quote"]), "relevance": int(item["relevance"])} for item in evidence],
            )
            self._validate_answer_annotation(annotation, evidence)
        except Exception:
            logger.exception(
                "RAG 答案标注：自动生成失败 case_id=%s evidence_count=%s",
                case_id,
                len(evidence_log),
            )
            raise
        finally:
            model.release()
        logger.info(
            "RAG 答案标注：自动生成完成 case_id=%s annotation=%s",
            case_id,
            annotation.model_dump_json(),
        )
        return annotation.model_dump(mode="json")

    def update_answer_annotation(
        self,
        user: CurrentUser,
        case_id: UUID,
        annotation: AnswerAnnotation,
    ) -> dict[str, Any]:
        with self._engine.begin() as connection:
            draft = self._require_case_draft(connection, user, case_id, for_update=True)
            evidence = self._draft_evidence(connection, case_id)
            self._validate_answer_annotation(annotation, evidence)
            row = (
                connection.execute(
                    text(
                        """
                        update rag_evaluation_case_drafts
                        set answer_annotation = cast(:annotation as jsonb), status = 'draft',
                            reviewed_by_user_id = null, reviewed_at = null,
                            approved_by_user_id = null, approved_at = null,
                            revision = revision + 1, updated_at = now()
                        where id = :case_id
                        returning *
                        """
                    ),
                    {"case_id": case_id, "annotation": _json(annotation.model_dump(mode="json"))},
                )
                .mappings()
                .one()
            )
            connection.execute(
                text("update evaluation_datasets set updated_at = now() where id = :dataset_id"),
                {"dataset_id": draft["dataset_id"]},
            )
            return {**dict(row), "evidence": evidence}

    def transition_case(self, user: CurrentUser, case_id: UUID, revision: int, action: str) -> dict[str, Any]:
        with self._engine.begin() as connection:
            draft = self._require_case_draft(connection, user, case_id, for_update=True)
            if int(draft["revision"]) != revision:
                raise RagEvaluationConflictError("问题已被其他操作修改，请刷新后重试")
            evidence = self._draft_evidence(connection, case_id)
            annotation = self._annotation_from_row(draft)
            self._validate_answer_annotation(annotation, evidence)
            if action == "review" and draft["status"] == "draft":
                statement = """
                    update rag_evaluation_case_drafts
                    set status = 'reviewed', reviewed_by_user_id = :user_id,
                        reviewed_at = now(), revision = revision + 1, updated_at = now()
                    where id = :case_id returning *
                """
            elif action == "approve" and draft["status"] == "reviewed":
                statement = """
                    update rag_evaluation_case_drafts
                    set status = 'approved', approved_by_user_id = :user_id,
                        approved_at = now(), revision = revision + 1, updated_at = now()
                    where id = :case_id returning *
                """
            else:
                raise RagEvaluationConflictError(f"Cannot {action} case in {draft['status']} status")
            return dict(connection.execute(text(statement), {"case_id": case_id, "user_id": user.id}).mappings().one())

    def search_chunks(
        self,
        user: CurrentUser,
        *,
        query: str = "",
        recording_id: UUID | None = None,
        limit: int = 30,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(100, limit))
        bounded_offset = max(0, offset)
        normalized = normalize_search_text(query)
        values: dict[str, object] = {
            "workspace_id": user.current_workspace_id,
            "limit": bounded_limit,
            "offset": bounded_offset,
        }
        clauses = ["recordings.workspace_id = :workspace_id", "recordings.status = 'completed'"]
        if recording_id is not None:
            clauses.append("recordings.id = :recording_id")
            values["recording_id"] = recording_id
        if normalized:
            # pg_trgm's word_similarity can return zero for short queries and
            # CJK text without word boundaries.  A normalized substring match
            # guarantees that an exact keyword remains searchable, while the
            # trigram predicate still recalls fuzzy matches.
            clauses.append(
                "(position(:query in coalesce(nullif(chunks.normalized_original_text, ''), chunks.normalized_text)) > 0 "
                "or word_similarity(:query, coalesce(nullif(chunks.normalized_original_text, ''), chunks.normalized_text)) > 0)"
            )
            values["query"] = normalized
            score = "word_similarity(:query, coalesce(nullif(chunks.normalized_original_text, ''), chunks.normalized_text))"
            order = (
                "(position(:query in coalesce(nullif(chunks.normalized_original_text, ''), chunks.normalized_text)) > 0) desc, "
                ":query <<-> coalesce(nullif(chunks.normalized_original_text, ''), chunks.normalized_text)"
            )
        else:
            score = "0.0"
            order = "recordings.created_at desc, chunks.chunk_index"
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(
                    f"""
                    select chunks.id, chunks.recording_id, chunks.chunk_index, chunks.text,
                           chunks.start_ms, chunks.end_ms, chunks.metadata,
                           recordings.title as recording_title, recordings.file_name,
                           {score} as score
                    from recording_search_chunks chunks
                    join recordings on recordings.id = chunks.recording_id
                    where {" and ".join(clauses)}
                    order by {order}
                    limit :limit offset :offset
                    """
                ),
                values,
            ).mappings()
            return [dict(row) for row in rows]

    def list_recordings(self, user: CurrentUser) -> list[dict[str, Any]]:
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    select recordings.id, recordings.title, recordings.file_name,
                           recordings.created_at, count(chunks.id) as chunk_count
                    from recordings
                    join recording_search_chunks chunks on chunks.recording_id = recordings.id
                    where recordings.workspace_id = :workspace_id
                      and recordings.status = 'completed'
                    group by recordings.id
                    order by recordings.created_at desc, recordings.id
                    """
                ),
                {"workspace_id": user.current_workspace_id},
            ).mappings()
            return [dict(row) for row in rows]

    def preview_version(self, user: CurrentUser, dataset_id: UUID) -> dict[str, Any]:
        with self._engine.connect() as connection:
            self._require_dataset(connection, user, dataset_id)
            cases = self._approved_case_payload(connection, dataset_id)
            checksum = _stable_checksum(cases)
            return {"case_count": len(cases), "evidence_count": sum(len(item["evidence"]) for item in cases), "checksum": checksum}

    def freeze_version(self, user: CurrentUser, dataset_id: UUID, expected_checksum: str) -> dict[str, Any]:
        with self._engine.begin() as connection:
            self._require_dataset(connection, user, dataset_id, for_update=True)
            cases = self._approved_case_payload(connection, dataset_id)
            checksum = _stable_checksum(cases)
            if checksum != expected_checksum:
                raise RagEvaluationConflictError("标注已变化，请重新预览后再冻结")
            version_number = cast(
                int,
                connection.execute(
                    text("select coalesce(max(version_number), 0) + 1 from evaluation_dataset_versions where dataset_id = :dataset_id"),
                    {"dataset_id": dataset_id},
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
                        :dataset_id, :version_number, 'building', 'rag_query', 'v1',
                        cast(:definition as jsonb), cast(:split_strategy as jsonb),
                        :case_count, :user_id
                    ) returning *
                    """
                    ),
                    {
                        "dataset_id": dataset_id,
                        "version_number": version_number,
                        "definition": _json(
                            {
                                "task_type": "rag_retrieval",
                                "query_normalization": "search_text_v1",
                                "evidence_matcher": "recording_time_overlap_or_quote_v1",
                                "answer_ground_truth": "reference_answer_and_key_points_v1",
                                "relevance_scale": [0, 1, 2, 3],
                            }
                        ),
                        "split_strategy": _json({"name": "all_test_v1"}),
                        "case_count": len(cases),
                        "user_id": user.id,
                    },
                )
                .mappings()
                .one()
            )
            for item in cases:
                frozen_case_id = cast(
                    UUID,
                    connection.execute(
                        text(
                            """
                            insert into rag_evaluation_cases (
                                dataset_version_id, source_draft_id, query, query_normalized,
                                scope, tags, split, group_key, answer_annotation
                            ) values (
                                :version_id, :draft_id, :query, :query_normalized,
                                cast(:scope as jsonb), cast(:tags as text[]), 'test', :group_key,
                                cast(:answer_annotation as jsonb)
                            ) returning id
                            """
                        ),
                        {
                            "version_id": version["id"],
                            "draft_id": item["id"],
                            "query": item["query"],
                            "query_normalized": normalize_search_text(str(item["query"])),
                            "scope": _json(item["scope"]),
                            "tags": item["tags"],
                            "group_key": item["group_key"],
                            "answer_annotation": _json(item["answer_annotation"]),
                        },
                    ).scalar_one(),
                )
                evidence_id_mapping: dict[UUID, UUID] = {}
                for evidence in cast(list[dict[str, Any]], item["evidence"]):
                    frozen_evidence_id = cast(
                        UUID,
                        connection.execute(
                            text(
                                """
                            insert into rag_evaluation_evidence (
                                evaluation_case_id, source_recording_id, source_chunk_id,
                                quote, start_ms, end_ms, relevance, content_checksum, metadata
                            ) values (
                                :case_id, :recording_id, :chunk_id, :quote, :start_ms,
                                :end_ms, :relevance, :checksum, cast(:metadata as jsonb)
                            ) returning id
                            """
                            ),
                            {
                                "case_id": frozen_case_id,
                                "recording_id": evidence["source_recording_id"],
                                "chunk_id": evidence["source_chunk_id"],
                                "quote": evidence["quote"],
                                "start_ms": evidence["start_ms"],
                                "end_ms": evidence["end_ms"],
                                "relevance": evidence["relevance"],
                                "checksum": evidence["content_checksum"],
                                "metadata": _json(evidence["metadata"]),
                            },
                        ).scalar_one(),
                    )
                    evidence_id_mapping[UUID(str(evidence["id"]))] = frozen_evidence_id
                frozen_annotation = AnswerAnnotation.model_validate(item["answer_annotation"]).remap_evidence_ids(evidence_id_mapping)
                connection.execute(
                    text("update rag_evaluation_cases set answer_annotation = cast(:annotation as jsonb) where id = :case_id"),
                    {"case_id": frozen_case_id, "annotation": _json(frozen_annotation.model_dump(mode="json"))},
                )
            frozen = (
                connection.execute(
                    text(
                        """
                    update evaluation_dataset_versions
                    set status = 'frozen', checksum = :checksum, frozen_at = now()
                    where id = :version_id returning *
                    """
                    ),
                    {"version_id": version["id"], "checksum": checksum},
                )
                .mappings()
                .one()
            )
            return dict(frozen)

    def create_run(
        self,
        user: CurrentUser,
        *,
        dataset_version_id: UUID,
        idempotency_key: str,
        baseline_run_id: UUID | None = None,
    ) -> dict[str, Any]:
        clean_key = idempotency_key.strip()
        if not clean_key:
            raise ValueError("Idempotency key is required")
        with self._engine.begin() as connection:
            version = (
                connection.execute(
                    text(
                        """
                    select versions.*, datasets.workspace_id, datasets.task_type
                    from evaluation_dataset_versions versions
                    join evaluation_datasets datasets on datasets.id = versions.dataset_id
                    where versions.id = :version_id and datasets.workspace_id = :workspace_id
                    """
                    ),
                    {"version_id": dataset_version_id, "workspace_id": user.current_workspace_id},
                )
                .mappings()
                .one_or_none()
            )
            if version is None:
                raise RagEvaluationNotFoundError("Dataset version not found")
            if version["status"] != "frozen" or version["task_type"] != "rag_retrieval":
                raise RagEvaluationConflictError("Only frozen RAG retrieval versions can be evaluated")
            existing = (
                connection.execute(
                    text("select * from evaluation_runs where workspace_id = :workspace_id and idempotency_key = :key"),
                    {"workspace_id": user.current_workspace_id, "key": clean_key},
                )
                .mappings()
                .one_or_none()
            )
            if existing is not None:
                return dict(existing)
            snapshot_id = self._ensure_corpus_snapshot(connection, user)
            pipeline_id, config = self._ensure_pipeline_version(connection, user)
            case_count = cast(
                int,
                connection.execute(
                    text("select count(*) from rag_evaluation_cases where dataset_version_id = :version_id and split = 'test'"),
                    {"version_id": dataset_version_id},
                ).scalar_one(),
            )
            incomplete_answer_count = cast(
                int,
                connection.execute(
                    text(
                        """
                        select count(*) from rag_evaluation_cases
                        where dataset_version_id = :version_id and split = 'test'
                          and answer_annotation is null
                        """
                    ),
                    {"version_id": dataset_version_id},
                ).scalar_one(),
            )
            if incomplete_answer_count:
                raise RagEvaluationConflictError("该冻结版本缺少答案标注，请从草稿重新冻结一个版本")
            run = (
                connection.execute(
                    text(
                        """
                    insert into evaluation_runs (
                        workspace_id, dataset_version_id, evaluator_type, split, status,
                        idempotency_key, config_snapshot, total_case_count, created_by_user_id
                    ) values (
                        :workspace_id, :version_id, 'rag_retrieval', 'test', 'queued',
                        :key, cast(:config as jsonb), :case_count, :user_id
                    ) returning *
                    """
                    ),
                    {
                        "workspace_id": user.current_workspace_id,
                        "version_id": dataset_version_id,
                        "key": clean_key,
                        "config": _json({**config, "evaluator_version": "2", "metric_version": "2"}),
                        "case_count": case_count,
                        "user_id": user.id,
                    },
                )
                .mappings()
                .one()
            )
            connection.execute(
                text(
                    """
                    insert into rag_evaluation_run_specs (
                        evaluation_run_id, corpus_snapshot_id, pipeline_version_id, baseline_run_id
                    ) values (:run_id, :snapshot_id, :pipeline_id, :baseline_run_id)
                    """
                ),
                {
                    "run_id": run["id"],
                    "snapshot_id": snapshot_id,
                    "pipeline_id": pipeline_id,
                    "baseline_run_id": baseline_run_id,
                },
            )
            return dict(run)

    def list_runs(self, user: CurrentUser) -> list[dict[str, Any]]:
        with self._engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    select runs.*, datasets.name as dataset_name, versions.version_number,
                           pipelines.name as pipeline_name, pipelines.config_hash
                    from evaluation_runs runs
                    join evaluation_dataset_versions versions on versions.id = runs.dataset_version_id
                    join evaluation_datasets datasets on datasets.id = versions.dataset_id
                    join rag_evaluation_run_specs specs on specs.evaluation_run_id = runs.id
                    join rag_pipeline_versions pipelines on pipelines.id = specs.pipeline_version_id
                    where runs.workspace_id = :workspace_id and runs.evaluator_type = 'rag_retrieval'
                    order by runs.created_at desc
                    """
                ),
                {"workspace_id": user.current_workspace_id},
            ).mappings()
            return [dict(row) for row in rows]

    def get_run(self, user: CurrentUser, run_id: UUID) -> dict[str, Any]:
        with self._engine.connect() as connection:
            run = (
                connection.execute(
                    text(
                        """
                    select runs.*, datasets.name as dataset_name, versions.version_number,
                           pipelines.name as pipeline_name, pipelines.config_hash,
                           pipelines.config_snapshot as pipeline_config
                    from evaluation_runs runs
                    join evaluation_dataset_versions versions on versions.id = runs.dataset_version_id
                    join evaluation_datasets datasets on datasets.id = versions.dataset_id
                    join rag_evaluation_run_specs specs on specs.evaluation_run_id = runs.id
                    join rag_pipeline_versions pipelines on pipelines.id = specs.pipeline_version_id
                    where runs.id = :run_id and runs.workspace_id = :workspace_id
                      and runs.evaluator_type = 'rag_retrieval'
                    """
                    ),
                    {"run_id": run_id, "workspace_id": user.current_workspace_id},
                )
                .mappings()
                .one_or_none()
            )
            if run is None:
                raise RagEvaluationNotFoundError("Evaluation run not found")
            metrics = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select * from rag_evaluation_metric_values
                        where evaluation_run_id = :run_id
                        order by scope, operation nulls first, metric_name
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
                        select results.*, cases.query, cases.tags, cases.answer_annotation
                        from rag_evaluation_case_results results
                        join rag_evaluation_cases cases on cases.id = results.evaluation_case_id
                        where results.evaluation_run_id = :run_id
                        order by cases.id
                        """
                    ),
                    {"run_id": run_id},
                ).mappings()
            ]
            journey_rows = [
                dict(row)
                for row in connection.execute(
                    text(
                        """
                        select diagnostics.*, evidence.quote, evidence.start_ms, evidence.end_ms,
                               evidence.relevance, evidence.source_recording_id,
                               evidence.source_chunk_id, recordings.title as recording_title
                        from rag_evaluation_evidence_diagnostics diagnostics
                        join rag_evaluation_evidence evidence on evidence.id = diagnostics.evidence_id
                        left join recordings on recordings.id = evidence.source_recording_id
                        where diagnostics.evaluation_run_id = :run_id
                        order by diagnostics.case_result_id, evidence.relevance desc, evidence.id
                        """
                    ),
                    {"run_id": run_id},
                ).mappings()
            ]
            journeys_by_case: dict[object, list[dict[str, Any]]] = defaultdict(list)
            all_journeys: list[dict[str, Any]] = []
            for row in journey_rows:
                journey = {
                    "diagnostic_version": row["diagnostic_version"],
                    "evidence_id": str(row["evidence_id"]),
                    "recording_id": str(row["source_recording_id"]),
                    "recording_title": row["recording_title"],
                    "source_chunk_id": str(row["source_chunk_id"]) if row["source_chunk_id"] else None,
                    "quote": row["quote"],
                    "start_ms": row["start_ms"],
                    "end_ms": row["end_ms"],
                    "relevance": row["relevance"],
                    "final_covered": row["final_covered"],
                    "last_visible_node": row["last_visible_node"],
                    "first_loss_node": row["first_loss_node"],
                    "stages": row["stage_journey"],
                }
                journeys_by_case[row["case_result_id"]].append(journey)
                all_journeys.append(journey)
            for case in cases:
                case["evidence_journeys"] = journeys_by_case.get(case["id"], [])
                steps = [
                    dict(row)
                    for row in connection.execute(
                        text(
                            """
                            select * from rag_evaluation_step_results
                            where case_result_id = :case_result_id order by sequence
                            """
                        ),
                        {"case_result_id": case["id"]},
                    ).mappings()
                ]
                for step in steps:
                    step["ranked_results"] = [
                        dict(row)
                        for row in connection.execute(
                            text(
                                """
                                select ranked.*, recordings.title as recording_title,
                                       chunks.text, chunks.start_ms, chunks.end_ms
                                from rag_evaluation_ranked_results ranked
                                left join recordings on recordings.id = ranked.recording_id
                                left join rag_corpus_snapshot_chunks chunks
                                  on chunks.id = ranked.corpus_snapshot_chunk_id
                                where ranked.step_result_id = :step_id
                                order by ranked.rank
                                """
                            ),
                            {"step_id": step["id"]},
                        ).mappings()
                    ]
                case["steps"] = steps
            return {
                "run": dict(run),
                "metrics": metrics,
                "cases": cases,
                "evidence_diagnosis": summarize_journeys(all_journeys),
            }

    def cancel_run(self, user: CurrentUser, run_id: UUID) -> dict[str, Any]:
        with self._engine.begin() as connection:
            row = (
                connection.execute(
                    text(
                        """
                    update evaluation_runs
                    set cancel_requested = true, updated_at = now()
                    where id = :run_id and workspace_id = :workspace_id
                      and evaluator_type = 'rag_retrieval'
                      and status in ('queued', 'running')
                    returning *
                    """
                    ),
                    {"run_id": run_id, "workspace_id": user.current_workspace_id},
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                raise RagEvaluationConflictError("Run cannot be cancelled")
            return dict(row)

    def delete_run(self, user: CurrentUser, run_id: UUID) -> None:
        """Delete a completed RAG evaluation run and its derived results."""
        with self._engine.begin() as connection:
            result = connection.execute(
                text(
                    """
                    delete from evaluation_runs
                    where id = :run_id and workspace_id = :workspace_id
                      and evaluator_type = 'rag_retrieval'
                      and status in ('succeeded', 'failed', 'cancelled')
                    """
                ),
                {"run_id": run_id, "workspace_id": user.current_workspace_id},
            )
            if result.rowcount != 1:
                raise RagEvaluationConflictError("Only completed, failed, or cancelled runs can be deleted")

    def _ensure_corpus_snapshot(self, connection: Connection, user: CurrentUser) -> UUID:
        rows = [
            dict(row)
            for row in connection.execute(
                text(
                    """
                    select chunks.*, models.id as current_embedding_model_id
                    from recording_search_chunks chunks
                    join recordings on recordings.id = chunks.recording_id
                    join recording_search_chunk_embeddings vectors on vectors.chunk_id = chunks.id
                    join embedding_models models on models.id = vectors.embedding_model_id
                    where recordings.workspace_id = :workspace_id
                      and recordings.status = 'completed'
                      and vectors.model_key = :model_key
                      and models.provider = 'sentence_transformers'
                      and models.model_name = :model_name
                      and models.dimensions = :dimensions
                    order by chunks.recording_id, chunks.chunk_index
                    """
                ),
                {
                    "workspace_id": user.current_workspace_id,
                    "model_key": self._settings.embedding_profile,
                    "model_name": self._settings.embedding_model,
                    "dimensions": self._settings.embedding_dimensions,
                },
            ).mappings()
        ]
        if not rows:
            raise RagEvaluationConflictError("当前 Workspace 没有可评测的 SearchChunk 索引")
        checksum_payload = [
            [str(row["id"]), str(row["recording_id"]), row["chunk_index"], _content_checksum(str(row["text"]), row["start_ms"], row["end_ms"])] for row in rows
        ]
        checksum = _stable_checksum(checksum_payload)
        existing = connection.execute(
            text(
                """
                select id from rag_corpus_snapshots
                where workspace_id = :workspace_id and checksum = :checksum and status = 'frozen'
                """
            ),
            {"workspace_id": user.current_workspace_id, "checksum": checksum},
        ).scalar_one_or_none()
        if existing is not None:
            return cast(UUID, existing)
        snapshot_id = cast(
            UUID,
            connection.execute(
                text(
                    """
                    insert into rag_corpus_snapshots (
                        workspace_id, name, status, recording_pipeline_version,
                        search_chunk_version, embedding_model_id, config_snapshot,
                        recording_count, chunk_count, created_by_user_id
                    ) values (
                        :workspace_id, :name, 'building', '21', '3', :embedding_model_id,
                        cast(:config as jsonb), :recording_count, :chunk_count, :user_id
                    ) returning id
                    """
                ),
                {
                    "workspace_id": user.current_workspace_id,
                    "name": f"自动快照 {checksum[:8]}",
                    "embedding_model_id": rows[0]["current_embedding_model_id"],
                    "config": _json(
                        {
                            "embedding_profile": self._settings.embedding_profile,
                            "embedding_model": self._settings.embedding_model,
                            "dimensions": self._settings.embedding_dimensions,
                        }
                    ),
                    "recording_count": len({row["recording_id"] for row in rows}),
                    "chunk_count": len(rows),
                    "user_id": user.id,
                },
            ).scalar_one(),
        )
        for row in rows:
            connection.execute(
                text(
                    """
                    insert into rag_corpus_snapshot_chunks (
                        corpus_snapshot_id, source_chunk_id, recording_id, chunk_index,
                        text, normalized_text, start_ms, end_ms, metadata, content_checksum
                    ) values (
                        :snapshot_id, :chunk_id, :recording_id, :chunk_index,
                        :text, :normalized_text, :start_ms, :end_ms,
                        cast(:metadata as jsonb), :checksum
                    )
                    """
                ),
                {
                    "snapshot_id": snapshot_id,
                    "chunk_id": row["id"],
                    "recording_id": row["recording_id"],
                    "chunk_index": row["chunk_index"],
                    "text": row["text"],
                    "normalized_text": row["normalized_text"],
                    "start_ms": row["start_ms"],
                    "end_ms": row["end_ms"],
                    "metadata": _json(_string_mapping(row["metadata"])),
                    "checksum": _content_checksum(str(row["text"]), row["start_ms"], row["end_ms"]),
                },
            )
        connection.execute(
            text("update rag_corpus_snapshots set status = 'frozen', checksum = :checksum, frozen_at = now() where id = :id"),
            {"id": snapshot_id, "checksum": checksum},
        )
        return snapshot_id

    def _ensure_pipeline_version(self, connection: Connection, user: CurrentUser) -> tuple[UUID, dict[str, Any]]:
        config = {
            "retrieval_contract_version": "1",
            "embedding": {
                "profile": self._settings.embedding_profile,
                "provider": "sentence_transformers",
                "model": self._settings.embedding_model,
                "dimensions": self._settings.embedding_dimensions,
                "distance_metric": "cosine",
            },
            "hybrid_enabled": self._settings.rag_hybrid_search_enabled,
            "online_default_model": self._settings.rag_online_default_model,
            "answer_judge": {
                "model": self._settings.rag_answer_judge_model,
                "prompt_version": self._settings.rag_answer_judge_prompt_version,
                "max_output_tokens": self._settings.rag_answer_judge_max_output_tokens,
            },
            "query_term_expansion_enabled": self._settings.rag_query_term_expansion_enabled,
            "vector_top_k": self._settings.rag_vector_candidate_limit,
            "lexical_top_k": self._settings.rag_lexical_candidate_limit,
            "fused_top_k": self._settings.rag_fused_candidate_limit,
            "rrf_k": self._settings.rag_rrf_k,
            "original_vector_weight": self._settings.rag_original_vector_weight,
            "expanded_vector_weight": self._settings.rag_expanded_vector_weight,
            "lexical_weight": self._settings.rag_lexical_weight,
            "context_window_utterances": self._settings.rag_chunk_context_window_utterances,
            "rerank": {
                "enabled": self._settings.rag_rerank_enabled,
                "model": self._settings.rag_rerank_model,
                "candidate_limit": self._settings.rag_rerank_candidate_limit,
                "output_limit": self._settings.rag_rerank_output_limit,
                "max_total_tokens": self._settings.rag_rerank_max_total_tokens,
            },
        }
        config_hash = _stable_checksum(config)
        pipeline_id = cast(
            UUID,
            connection.execute(
                text(
                    """
                    insert into rag_pipeline_versions (
                        workspace_id, name, config_hash, config_snapshot, created_by_user_id
                    ) values (
                        :workspace_id, :name, :hash, cast(:config as jsonb), :user_id
                    )
                    on conflict (workspace_id, config_hash)
                    do update set config_hash = excluded.config_hash
                    returning id
                    """
                ),
                {
                    "workspace_id": user.current_workspace_id,
                    "name": f"当前生产配置 {config_hash[:8]}",
                    "hash": config_hash,
                    "config": _json(config),
                    "user_id": user.id,
                },
            ).scalar_one(),
        )
        return pipeline_id, config

    def _approved_case_payload(self, connection: Connection, dataset_id: UUID) -> list[dict[str, Any]]:
        cases = [
            dict(row)
            for row in connection.execute(
                text(
                    """
                    select * from rag_evaluation_case_drafts
                    where dataset_id = :dataset_id and status = 'approved' and archived_at is null
                    order by id
                    """
                ),
                {"dataset_id": dataset_id},
            ).mappings()
        ]
        if not cases:
            raise RagEvaluationConflictError("至少需要一个已批准的问题")
        for case in cases:
            case["evidence"] = [
                dict(row)
                for row in connection.execute(
                    text("select * from rag_evaluation_evidence_drafts where case_draft_id = :case_id order by id"),
                    {"case_id": case["id"]},
                ).mappings()
            ]
            annotation = self._annotation_from_row(case)
            self._validate_answer_annotation(annotation, cast(list[dict[str, Any]], case["evidence"]))
        return cases

    @staticmethod
    def _draft_evidence(connection: Connection, case_id: UUID) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in connection.execute(
                text("select * from rag_evaluation_evidence_drafts where case_draft_id = :case_id order by id"),
                {"case_id": case_id},
            ).mappings()
        ]

    @staticmethod
    def _annotation_from_row(row: Mapping[str, object]) -> AnswerAnnotation:
        raw = row.get("answer_annotation")
        if raw is None:
            raise RagEvaluationConflictError("请先保存答案标注")
        try:
            return AnswerAnnotation.model_validate(raw)
        except ValueError as error:
            raise RagEvaluationConflictError(f"答案标注无效：{error}") from error

    @staticmethod
    def _validate_answer_annotation(annotation: AnswerAnnotation, evidence: list[dict[str, Any]]) -> None:
        try:
            annotation.validate_case_evidence({UUID(str(item["id"])) for item in evidence})
        except ValueError as error:
            raise RagEvaluationConflictError(str(error)) from error

    @staticmethod
    def _return_case_to_draft(connection: Connection, case_id: UUID) -> None:
        connection.execute(
            text(
                """
                update rag_evaluation_case_drafts
                set status = 'draft', reviewed_by_user_id = null, reviewed_at = null,
                    approved_by_user_id = null, approved_at = null,
                    revision = revision + 1, updated_at = now()
                where id = :case_id
                """
            ),
            {"case_id": case_id},
        )

    @staticmethod
    def _require_recordings(connection: Connection, user: CurrentUser, recording_ids: list[UUID]) -> None:
        if not recording_ids:
            return
        count = cast(
            int,
            connection.execute(
                text("select count(*) from recordings where workspace_id = :workspace_id and id = any(cast(:ids as uuid[]))"),
                {"workspace_id": user.current_workspace_id, "ids": [str(item) for item in recording_ids]},
            ).scalar_one(),
        )
        if count != len(set(recording_ids)):
            raise RagEvaluationPermissionError("Recording scope contains inaccessible recordings")

    @staticmethod
    def _require_dataset(
        connection: Connection,
        user: CurrentUser,
        dataset_id: UUID,
        *,
        for_update: bool = False,
    ) -> dict[str, Any]:
        suffix = " for update" if for_update else ""
        row = (
            connection.execute(
                text("select * from evaluation_datasets where id = :dataset_id and workspace_id = :workspace_id and task_type = 'rag_retrieval'" + suffix),
                {"dataset_id": dataset_id, "workspace_id": user.current_workspace_id},
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise RagEvaluationNotFoundError("RAG evaluation dataset not found")
        return dict(row)

    @staticmethod
    def _require_case_draft(
        connection: Connection,
        user: CurrentUser,
        case_id: UUID,
        *,
        for_update: bool = False,
    ) -> dict[str, Any]:
        suffix = " for update" if for_update else ""
        row = (
            connection.execute(
                text(
                    """
                select drafts.* from rag_evaluation_case_drafts drafts
                join evaluation_datasets datasets on datasets.id = drafts.dataset_id
                where drafts.id = :case_id and datasets.workspace_id = :workspace_id
                  and datasets.task_type = 'rag_retrieval'
                """
                    + suffix
                ),
                {"case_id": case_id, "workspace_id": user.current_workspace_id},
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise RagEvaluationNotFoundError("RAG evaluation case not found")
        return dict(row)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _stable_checksum(value: object) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _content_checksum(content: str, start_ms: object, end_ms: object) -> str:
    return _stable_checksum([content, int(cast(int, start_ms)), int(cast(int, end_ms))])


def _string_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in cast(Mapping[object, object], value).items()}
