from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast
from uuid import UUID

from sqlalchemy import Connection, Engine, text

logger = logging.getLogger("evaluation")


@dataclass(frozen=True, slots=True)
class RagEvaluationCleanupResult:
    cleaned_draft_cases: int
    deleted_dataset_versions: int
    deleted_corpus_snapshots: int
    deleted_runs: int

    @property
    def changed(self) -> bool:
        return any(
            (
                self.cleaned_draft_cases,
                self.deleted_dataset_versions,
                self.deleted_corpus_snapshots,
                self.deleted_runs,
            )
        )


class RagEvaluationOrphanCleanup:
    """Remove RAG evaluation aggregates whose weak recording references are orphaned."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def run(self) -> RagEvaluationCleanupResult:
        with self._engine.begin() as connection:
            connection.execute(text("select set_config('app.evaluation_maintenance', 'on', true)"))
            draft_case_ids = self._uuid_list(
                connection,
                """
                select distinct drafts.id
                from rag_evaluation_case_drafts drafts
                where exists (
                    select 1 from rag_evaluation_evidence_drafts evidence
                    where evidence.case_draft_id = drafts.id
                      and not exists (select 1 from recordings where id = evidence.source_recording_id)
                ) or exists (
                    select 1
                    from jsonb_array_elements_text(
                        case when jsonb_typeof(drafts.scope -> 'recording_ids') = 'array'
                             then drafts.scope -> 'recording_ids' else '[]'::jsonb end
                    ) scoped(recording_id)
                    where not exists (
                        select 1 from recordings where id = scoped.recording_id::uuid
                    )
                )
                """,
            )
            version_ids = self._uuid_list(
                connection,
                """
                select distinct versions.id
                from evaluation_dataset_versions versions
                join evaluation_datasets datasets on datasets.id = versions.dataset_id
                join rag_evaluation_cases cases on cases.dataset_version_id = versions.id
                where datasets.task_type = 'rag_retrieval' and (
                    exists (
                        select 1 from rag_evaluation_evidence evidence
                        where evidence.evaluation_case_id = cases.id
                          and not exists (select 1 from recordings where id = evidence.source_recording_id)
                    ) or exists (
                        select 1
                        from jsonb_array_elements_text(
                            case when jsonb_typeof(cases.scope -> 'recording_ids') = 'array'
                                 then cases.scope -> 'recording_ids' else '[]'::jsonb end
                        ) scoped(recording_id)
                        where not exists (
                            select 1 from recordings where id = scoped.recording_id::uuid
                        )
                    )
                )
                """,
            )
            snapshot_ids = self._uuid_list(
                connection,
                """
                select distinct snapshots.id
                from rag_corpus_snapshots snapshots
                join rag_corpus_snapshot_chunks chunks on chunks.corpus_snapshot_id = snapshots.id
                where not exists (select 1 from recordings where id = chunks.recording_id)
                """,
            )
            run_ids = self._uuid_list(
                connection,
                """
                select distinct runs.id
                from evaluation_runs runs
                left join rag_evaluation_run_specs specs on specs.evaluation_run_id = runs.id
                where runs.evaluator_type = 'rag_retrieval' and (
                    runs.dataset_version_id = any(cast(:version_ids as uuid[]))
                    or specs.corpus_snapshot_id = any(cast(:snapshot_ids as uuid[]))
                    or exists (
                        select 1
                        from rag_evaluation_case_results case_results
                        join rag_evaluation_step_results steps on steps.case_result_id = case_results.id
                        join rag_evaluation_ranked_results ranked on ranked.step_result_id = steps.id
                        where case_results.evaluation_run_id = runs.id
                          and not exists (select 1 from recordings where id = ranked.recording_id)
                    )
                )
                """,
                {"version_ids": version_ids, "snapshot_ids": snapshot_ids},
            )

            deleted_runs = self._delete_by_ids(connection, "evaluation_runs", run_ids)
            if version_ids:
                connection.execute(
                    text(
                        """
                        delete from rag_evaluation_evidence
                        where evaluation_case_id in (
                            select id from rag_evaluation_cases
                            where dataset_version_id = any(cast(:ids as uuid[]))
                        )
                        """
                    ),
                    {"ids": version_ids},
                )
                connection.execute(
                    text("delete from rag_evaluation_cases where dataset_version_id = any(cast(:ids as uuid[]))"),
                    {"ids": version_ids},
                )
            deleted_versions = self._delete_by_ids(connection, "evaluation_dataset_versions", version_ids)
            if snapshot_ids:
                connection.execute(
                    text("delete from rag_corpus_snapshot_chunks where corpus_snapshot_id = any(cast(:ids as uuid[]))"),
                    {"ids": snapshot_ids},
                )
            deleted_snapshots = self._delete_by_ids(connection, "rag_corpus_snapshots", snapshot_ids)
            if draft_case_ids:
                connection.execute(
                    text(
                        """
                        delete from rag_evaluation_evidence_drafts
                        where case_draft_id = any(cast(:ids as uuid[]))
                          and not exists (select 1 from recordings where id = source_recording_id)
                        """
                    ),
                    {"ids": draft_case_ids},
                )
                connection.execute(
                    text(
                        """
                        update rag_evaluation_case_drafts drafts
                        set scope = jsonb_set(
                                drafts.scope,
                                '{recording_ids}',
                                coalesce((
                                    select jsonb_agg(scoped.recording_id)
                                    from jsonb_array_elements_text(
                                        case when jsonb_typeof(drafts.scope -> 'recording_ids') = 'array'
                                             then drafts.scope -> 'recording_ids' else '[]'::jsonb end
                                    ) scoped(recording_id)
                                    where exists (
                                        select 1 from recordings where id = scoped.recording_id::uuid
                                    )
                                ), '[]'::jsonb),
                                true
                            ),
                            status = 'draft', reviewed_by_user_id = null, reviewed_at = null,
                            approved_by_user_id = null, approved_at = null,
                            revision = revision + 1, updated_at = now()
                        where drafts.id = any(cast(:ids as uuid[]))
                        """
                    ),
                    {"ids": draft_case_ids},
                )

        result = RagEvaluationCleanupResult(
            cleaned_draft_cases=len(draft_case_ids),
            deleted_dataset_versions=deleted_versions,
            deleted_corpus_snapshots=deleted_snapshots,
            deleted_runs=deleted_runs,
        )
        if result.changed:
            logger.info("RAG 评测孤儿数据清理完成: %s", result)
        return result

    @staticmethod
    def _uuid_list(
        connection: Connection,
        statement: str,
        parameters: dict[str, object] | None = None,
    ) -> list[UUID]:
        rows = connection.execute(text(statement), parameters or {}).scalars()
        return [cast(UUID, value) for value in rows]

    @staticmethod
    def _delete_by_ids(connection: Connection, table: str, ids: list[UUID]) -> int:
        if not ids:
            return 0
        result = connection.execute(
            text(f"delete from {table} where id = any(cast(:ids as uuid[]))"),
            {"ids": ids},
        )
        return int(result.rowcount)
