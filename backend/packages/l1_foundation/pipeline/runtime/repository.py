from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, cast
from uuid import UUID

from sqlalchemy import Engine, text

from l1_foundation.pipeline.contracts import ArtifactRef, PipelineRunId, StageRunId
from l1_foundation.pipeline.definitions.graph import PipelineDefinition


class PipelineRepository:
    """PostgreSQL persistence for the pipeline runtime state machine."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def create_run(self, subject_type: str, subject_id: UUID, definition: PipelineDefinition, initial_artifacts: tuple[ArtifactRef, ...] = ()) -> PipelineRunId:
        """Persist one pipeline graph and atomically enqueue it through the outbox."""
        with self._engine.begin() as connection:
            run_id = cast(
                UUID,
                connection.execute(
                    text(
                        """
                        insert into pipeline_runs (subject_type, subject_id, pipeline_name, pipeline_version, status)
                        values (:subject_type, :subject_id, :pipeline_name, :pipeline_version, 'queued')
                        returning id
                        """
                    ),
                    {"subject_type": subject_type, "subject_id": subject_id, "pipeline_name": definition.name, "pipeline_version": definition.version},
                ).scalar_one(),
            )
            for artifact in initial_artifacts:
                connection.execute(
                    text(
                        """
                        insert into artifacts (subject_type, subject_id, pipeline_run_id, artifact_type, artifact_version, uri, checksum, metadata)
                        values (:subject_type, :subject_id, :pipeline_run_id, :artifact_type, :artifact_version, :uri, :checksum, cast(:metadata as jsonb))
                        """
                    ),
                    {
                        "subject_type": subject_type,
                        "subject_id": subject_id,
                        "pipeline_run_id": run_id,
                        "artifact_type": artifact.artifact_type,
                        "artifact_version": artifact.artifact_version,
                        "uri": artifact.uri,
                        "checksum": artifact.checksum,
                        "metadata": json.dumps(artifact.metadata),
                    },
                )
            run_by_node: dict[str, UUID] = {}
            for node in definition.topologically_sorted_nodes():
                payload_json = json.dumps(node.input_payload or {}, sort_keys=True)
                fingerprint = hashlib.sha256(f"{node.stage_name}@{node.stage_version}:{payload_json}".encode()).hexdigest()
                stage_run_id = cast(
                    UUID,
                    connection.execute(
                        text(
                            """
                            insert into stage_runs (
                                pipeline_run_id, subject_type, subject_id, node_name, stage_name, stage_version, required,
                                resource_queue, status, max_attempts, input_fingerprint, input_payload, input_artifacts
                            ) values (
                                :pipeline_run_id, :subject_type, :subject_id, :node_name, :stage_name, :stage_version, :required,
                                :resource_queue, 'pending', :max_attempts, :input_fingerprint, cast(:input_payload as jsonb), cast(:input_artifacts as jsonb)
                            ) returning id
                            """
                        ),
                        {
                            "pipeline_run_id": run_id,
                            "subject_type": subject_type,
                            "subject_id": subject_id,
                            "node_name": node.name,
                            "stage_name": node.stage_name,
                            "stage_version": node.stage_version,
                            "required": node.required,
                            "resource_queue": node.resource_queue.value,
                            "max_attempts": node.retry_policy.max_attempts,
                            "input_fingerprint": fingerprint,
                            "input_payload": payload_json,
                            "input_artifacts": json.dumps([asdict(binding) for binding in node.input_artifacts]),
                        },
                    ).scalar_one(),
                )
                run_by_node[node.name] = stage_run_id
            for node in definition.nodes:
                for dependency in node.depends_on:
                    connection.execute(
                        text(
                            """
                            insert into stage_run_dependencies (stage_run_id, depends_on_stage_run_id)
                            values (:stage_run_id, :dependency_stage_run_id)
                            """
                        ),
                        {"stage_run_id": run_by_node[node.name], "dependency_stage_run_id": run_by_node[dependency]},
                    )
            self._append_event(connection, run_id, None, "pipeline.queued", {"subject_type": subject_type, "subject_id": str(subject_id)})
            connection.execute(
                text(
                    """
                    insert into outbox_events (topic, aggregate_type, aggregate_id, payload)
                    values ('pipeline.run.created', 'pipeline_run', :run_id, cast(:payload as jsonb))
                    """
                ),
                {"run_id": run_id, "payload": json.dumps({"pipeline_run_id": str(run_id)})},
            )
            return PipelineRunId(run_id)

    def resolve_stage_input(self, stage_run_id: StageRunId) -> dict[str, Any]:
        """Resolve the declared ArtifactBinding values for a claimed stage."""
        with self._engine.connect() as connection:
            stage_run = (
                connection.execute(
                    text("select pipeline_run_id, input_payload, input_artifacts from stage_runs where id = :stage_run_id"),
                    {"stage_run_id": stage_run_id},
                )
                .mappings()
                .one()
            )
            input_payload = cast(dict[str, Any], stage_run["input_payload"]).copy()
            bindings = cast(list[dict[str, Any]], stage_run["input_artifacts"])
            for binding in bindings:
                if binding["from_node"] is None:
                    artifact_row = (
                        connection.execute(
                            text(
                                """
                            select artifact_type, artifact_version, uri, checksum, metadata
                            from artifacts
                            where pipeline_run_id = :pipeline_run_id and stage_run_id is null and artifact_type = :artifact_type
                            order by created_at desc
                            limit 1
                            """
                            ),
                            {"pipeline_run_id": stage_run["pipeline_run_id"], "artifact_type": binding["artifact_type"]},
                        )
                        .mappings()
                        .one_or_none()
                    )
                else:
                    artifact_row = (
                        connection.execute(
                            text(
                                """
                            select artifacts.artifact_type, artifacts.artifact_version, artifacts.uri, artifacts.checksum, artifacts.metadata
                            from artifacts
                            join stage_runs producer on producer.id = artifacts.stage_run_id
                            where artifacts.pipeline_run_id = :pipeline_run_id
                              and producer.node_name = :producer_node
                              and artifacts.artifact_type = :artifact_type
                            order by artifacts.created_at desc
                            limit 1
                            """
                            ),
                            {
                                "pipeline_run_id": stage_run["pipeline_run_id"],
                                "producer_node": binding["from_node"],
                                "artifact_type": binding["artifact_type"],
                            },
                        )
                        .mappings()
                        .one_or_none()
                    )
                if artifact_row is None:
                    raise LookupError(f"Required artifact is missing: {binding['artifact_type']}")
                input_payload[cast(str, binding["name"])] = ArtifactRef(
                    artifact_type=cast(str, artifact_row["artifact_type"]),
                    artifact_version=cast(str, artifact_row["artifact_version"]),
                    uri=cast(str, artifact_row["uri"]),
                    checksum=cast(str | None, artifact_row["checksum"]),
                    metadata=cast(dict[str, Any], artifact_row["metadata"]),
                )
            return input_payload

    def get_stage(self, stage_run_id: StageRunId) -> dict[str, Any]:
        """Load the persisted workflow metadata required by a pipeline-owned operation."""
        with self._engine.connect() as connection:
            row = connection.execute(text("select * from stage_runs where id = :stage_run_id"), {"stage_run_id": stage_run_id}).mappings().one()
        return dict(row)

    def get_run(self, pipeline_run_id: UUID) -> dict[str, Any]:
        with self._engine.connect() as connection:
            row = connection.execute(text("select * from pipeline_runs where id = :pipeline_run_id"), {"pipeline_run_id": pipeline_run_id}).mappings().one()
        return dict(row)

    def requeue_stage(self, subject_type: str, subject_id: UUID, node_name: str) -> StageRunId:
        """Requeue one terminal node from the subject's latest run while preserving its attempt history."""
        with self._engine.begin() as connection:
            row = (
                connection.execute(
                    text(
                        """
                        select stage_runs.id, stage_runs.pipeline_run_id, stage_runs.status
                        from stage_runs
                        join pipeline_runs on pipeline_runs.id = stage_runs.pipeline_run_id
                        where pipeline_runs.subject_type = :subject_type
                          and pipeline_runs.subject_id = :subject_id
                          and stage_runs.node_name = :node_name
                        order by pipeline_runs.created_at desc, stage_runs.created_at desc
                        limit 1
                        for update of stage_runs, pipeline_runs
                        """
                    ),
                    {"subject_type": subject_type, "subject_id": subject_id, "node_name": node_name},
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                raise LookupError(f"Pipeline stage {node_name!r} does not exist for {subject_type} {subject_id}")
            if row["status"] in {"pending", "running", "retry_waiting"}:
                raise ValueError(f"Pipeline stage {node_name!r} is already active")
            blocked_dependency = connection.execute(
                text(
                    """
                    select 1
                    from stage_run_dependencies dependencies
                    join stage_runs upstream on upstream.id = dependencies.depends_on_stage_run_id
                    where dependencies.stage_run_id = :stage_run_id
                      and upstream.status <> 'succeeded'
                    limit 1
                    """
                ),
                {"stage_run_id": row["id"]},
            ).scalar_one_or_none()
            if blocked_dependency is not None:
                raise ValueError(f"Pipeline stage {node_name!r} has an incomplete dependency")
            connection.execute(
                text(
                    """
                    update stage_runs
                    set status = 'pending', available_at = now(), output_payload = null,
                        progress_percent = 0, progress_message = '等待重新执行', progress_updated_at = now(),
                        error_message = null, started_at = null, finished_at = null, updated_at = now()
                    where id = :stage_run_id
                    """
                ),
                {"stage_run_id": row["id"]},
            )
            connection.execute(
                text(
                    """
                    update pipeline_runs
                    set status = 'queued', finished_at = null, error_message = null, updated_at = now()
                    where id = :pipeline_run_id
                    """
                ),
                {"pipeline_run_id": row["pipeline_run_id"]},
            )
            self._append_event(
                connection,
                cast(UUID, row["pipeline_run_id"]),
                cast(UUID, row["id"]),
                "stage.requeued",
                {"node_name": node_name},
            )
            return StageRunId(cast(UUID, row["id"]))

    def resume_retry_stage(self, stage_run_id: StageRunId) -> None:
        """Make an already retry-waiting stage immediately available without creating another attempt record."""
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    update stage_runs
                    set status = 'pending', available_at = now(), error_message = null,
                        progress_percent = 0, progress_message = '等待重新执行',
                        progress_updated_at = now(), updated_at = now()
                    where id = :stage_run_id and status = 'retry_waiting'
                    """
                ),
                {"stage_run_id": stage_run_id},
            )

    def ready_stages(self) -> list[dict[str, Any]]:
        """Return workflow nodes ready for their owning coordinator to submit in memory."""
        with self._engine.connect() as connection:
            rows = (
                connection.execute(
                    text(
                        """
                    select stage_runs.*, pipeline_runs.pipeline_name, pipeline_runs.subject_type
                    from stage_runs
                    join pipeline_runs on pipeline_runs.id = stage_runs.pipeline_run_id
                    where stage_runs.status in ('pending', 'retry_waiting')
                      and stage_runs.available_at <= now()
                      and pipeline_runs.status in ('queued', 'running')
                      and not exists (
                          select 1
                          from stage_run_dependencies dependencies
                          join stage_runs upstream on upstream.id = dependencies.depends_on_stage_run_id
                          where dependencies.stage_run_id = stage_runs.id
                            and upstream.status <> 'succeeded'
                      )
                    order by stage_runs.created_at, stage_runs.id
                    """
                    )
                )
                .mappings()
                .all()
            )
        return [dict(row) for row in rows]

    def mark_stage_running(self, stage_run_id: StageRunId) -> int | None:
        """Claim a stage for this in-process coordinator before scheduling its callable."""
        with self._engine.begin() as connection:
            row = (
                connection.execute(
                    text(
                        """
                    update stage_runs
                    set status = 'running', attempt_count = attempt_count + 1, started_at = coalesce(started_at, now()),
                        progress_percent = 0, progress_message = '正在执行', progress_updated_at = now(), updated_at = now()
                    where id = :stage_run_id and status in ('pending', 'retry_waiting') and available_at <= now()
                    returning pipeline_run_id, attempt_count
                    """
                    ),
                    {"stage_run_id": stage_run_id},
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                return None
            connection.execute(
                text("update pipeline_runs set status = 'running', started_at = coalesce(started_at, now()), updated_at = now() where id = :run_id"),
                {"run_id": row["pipeline_run_id"]},
            )
            self._append_event(connection, cast(UUID, row["pipeline_run_id"]), stage_run_id, "stage.running", {"attempt": row["attempt_count"]})
            return cast(int, row["attempt_count"])

    def update_stage_progress(self, stage_run_id: StageRunId, percent: int, message: str) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    update stage_runs
                    set progress_percent = :percent, progress_message = :message, progress_updated_at = now(), updated_at = now()
                    where id = :stage_run_id and status = 'running'
                    """
                ),
                {"stage_run_id": stage_run_id, "percent": max(0, min(100, percent)), "message": message[:500]},
            )

    def mark_stage_succeeded(self, stage_run_id: StageRunId, output_payload: dict[str, Any], artifacts: tuple[ArtifactRef, ...]) -> None:
        with self._engine.begin() as connection:
            row = (
                connection.execute(
                    text(
                        """
                    update stage_runs
                    set status = 'succeeded', output_payload = cast(:output_payload as jsonb), finished_at = now(),
                        progress_percent = 100, progress_message = '完成', progress_updated_at = now(), error_message = null, updated_at = now()
                    where id = :stage_run_id and status = 'running'
                    returning pipeline_run_id, subject_type, subject_id
                    """
                    ),
                    {"stage_run_id": stage_run_id, "output_payload": json.dumps(output_payload)},
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                return
            for artifact in artifacts:
                connection.execute(
                    text(
                        """
                        insert into artifacts (
                            subject_type, subject_id, pipeline_run_id, stage_run_id,
                            artifact_type, artifact_version, uri, checksum, metadata
                        ) values (
                            :subject_type, :subject_id, :pipeline_run_id, :stage_run_id,
                            :artifact_type, :artifact_version, :uri, :checksum, cast(:metadata as jsonb)
                        )
                        on conflict (stage_run_id, artifact_type, artifact_version)
                        do update set uri = excluded.uri, checksum = excluded.checksum, metadata = excluded.metadata
                        """
                    ),
                    {
                        "subject_type": row["subject_type"],
                        "subject_id": row["subject_id"],
                        "pipeline_run_id": row["pipeline_run_id"],
                        "stage_run_id": stage_run_id,
                        "artifact_type": artifact.artifact_type,
                        "artifact_version": artifact.artifact_version,
                        "uri": artifact.uri,
                        "checksum": artifact.checksum,
                        "metadata": json.dumps(artifact.metadata),
                    },
                )
            run_id = cast(UUID, row["pipeline_run_id"])
            self._append_event(connection, run_id, stage_run_id, "stage.succeeded", {})
            self._refresh_run_status(connection, run_id)

    def mark_stage_retry(self, stage_run_id: StageRunId, error_message: str, retry_delay_seconds: int) -> None:
        with self._engine.begin() as connection:
            row = (
                connection.execute(
                    text(
                        """
                    update stage_runs
                    set status = 'retry_waiting', available_at = now() + make_interval(secs => :retry_delay_seconds),
                        error_message = :error_message, updated_at = now()
                    where id = :stage_run_id and status = 'running'
                    returning pipeline_run_id, attempt_count
                    """
                    ),
                    {"stage_run_id": stage_run_id, "error_message": error_message[:4000], "retry_delay_seconds": retry_delay_seconds},
                )
                .mappings()
                .one_or_none()
            )
            if row is not None:
                self._append_event(connection, cast(UUID, row["pipeline_run_id"]), stage_run_id, "stage.retry_waiting", {"attempt": row["attempt_count"]})

    @staticmethod
    def _append_event(connection: Any, pipeline_run_id: UUID, stage_run_id: UUID | None, event_type: str, payload: dict[str, Any]) -> None:
        connection.execute(
            text(
                """
                insert into pipeline_events (pipeline_run_id, stage_run_id, event_type, payload)
                values (:pipeline_run_id, :stage_run_id, :event_type, cast(:payload as jsonb))
                """
            ),
            {"pipeline_run_id": pipeline_run_id, "stage_run_id": stage_run_id, "event_type": event_type, "payload": json.dumps(payload)},
        )

    @staticmethod
    def _refresh_run_status(connection: Any, pipeline_run_id: UUID) -> None:
        summary = (
            connection.execute(
                text(
                    """
                select
                    count(*) filter (where required and status = 'failed') as required_failed,
                    count(*) filter (where not required and status = 'failed') as optional_failed,
                    count(*) filter (where status in ('pending', 'running', 'retry_waiting')) as active_count,
                    max(error_message) filter (where required and status = 'failed') as error_message
                from stage_runs
                where pipeline_run_id = :pipeline_run_id
                """
                ),
                {"pipeline_run_id": pipeline_run_id},
            )
            .mappings()
            .one()
        )
        active_count = cast(int, summary["active_count"])
        required_failed = cast(int, summary["required_failed"])
        optional_failed = cast(int, summary["optional_failed"])
        if active_count > 0:
            return
        if required_failed > 0:
            status = "failed"
        elif optional_failed > 0:
            status = "partial_failed"
        else:
            status = "succeeded"
        connection.execute(
            text(
                "update pipeline_runs set status = :status, error_message = :error_message, finished_at = now(), updated_at = now() where id = :pipeline_run_id"
            ),
            {"pipeline_run_id": pipeline_run_id, "status": status, "error_message": summary["error_message"]},
        )
