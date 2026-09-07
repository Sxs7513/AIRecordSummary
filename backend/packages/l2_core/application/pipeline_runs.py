from __future__ import annotations

# ruff: noqa: E501
import json
from typing import Any, cast
from uuid import UUID, uuid5

from sqlalchemy import Engine, text

from l1_foundation.pipeline.definitions.graph import PipelineDefinition, PipelineNode


class PipelineRunRepository:
    """Durable processing facts; deliberately excludes high-frequency live progress."""

    def __init__(self, engine: Engine, definition: PipelineDefinition) -> None:
        self._engine = engine
        self._definition = definition

    def start(self, state: dict[str, Any]) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text("""insert into pipeline_runs (id, recording_id, pipeline_name, pipeline_version, status, retry_count, started_at)
                values (:id, :recording_id, :pipeline_name, :pipeline_version, :status, :retry_count, :started_at)
                on conflict (id) do update set status=excluded.status, retry_count=excluded.retry_count,
                    error_message=null, finished_at=null, updated_at=now()"""),
                {
                    "id": state["processing_id"],
                    "recording_id": state["subject_id"],
                    "pipeline_name": state["pipeline_name"],
                    "pipeline_version": state["pipeline_version"],
                    "status": state["status"],
                    "retry_count": state.get("retry_count", 0),
                    "started_at": state.get("started_at"),
                },
            )

    def stage(self, run_id: UUID, recording_id: UUID, node: PipelineNode, stage: dict[str, Any], *, fingerprint: str | None = None) -> None:
        terminal = stage.get("status") in {"succeeded", "failed", "cancelled"}
        with self._engine.begin() as connection:
            connection.execute(
                text("""insert into pipeline_stage_runs (id,pipeline_run_id,recording_id,node_name,stage_name,stage_version,required,status,attempt_count,max_attempts,reused,input_fingerprint,artifacts,error_message,started_at,finished_at)
                values (:id,:run_id,:recording_id,:node_name,:stage_name,:stage_version,:required,:status,:attempt,:max_attempts,:reused,:fingerprint,cast(:artifacts as jsonb),:error,now(),case when :terminal then now() else null end)
                on conflict (pipeline_run_id,node_name) do update set status=excluded.status,attempt_count=excluded.attempt_count,reused=excluded.reused,input_fingerprint=coalesce(excluded.input_fingerprint,pipeline_stage_runs.input_fingerprint),artifacts=excluded.artifacts,error_message=excluded.error_message,finished_at=case when :terminal then now() else null end,updated_at=now()"""),
                {
                    "id": uuid5(run_id, node.name),
                    "run_id": run_id,
                    "recording_id": recording_id,
                    "node_name": node.name,
                    "stage_name": node.stage_name,
                    "stage_version": node.stage_version,
                    "required": node.required,
                    "status": stage["status"],
                    "attempt": stage.get("attempt", 0),
                    "max_attempts": node.retry_policy.max_attempts or 3,
                    "reused": stage.get("reused", False),
                    "fingerprint": fingerprint,
                    "artifacts": json.dumps(stage.get("artifacts", [])),
                    "error": stage.get("error"),
                    "terminal": terminal,
                },
            )

    def finish(self, state: dict[str, Any]) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text("update pipeline_runs set status=:status,error_message=:error,finished_at=:finished_at,updated_at=now() where id=:id"),
                {"id": state["processing_id"], "status": state["status"], "error": state.get("error_message"), "finished_at": state.get("finished_at")},
            )

    def state(self, run_id: UUID) -> dict[str, Any] | None:
        with self._engine.connect() as connection:
            run = connection.execute(text("select * from pipeline_runs where id=:id"), {"id": run_id}).mappings().one_or_none()
            if run is None:
                return None
            stages = connection.execute(text("select * from pipeline_stage_runs where pipeline_run_id=:id"), {"id": run_id}).mappings().all()
        return {
            "processing_id": str(run["id"]),
            "subject_type": "recording",
            "subject_id": str(run["recording_id"]),
            "pipeline_name": run["pipeline_name"],
            "pipeline_version": run["pipeline_version"],
            "status": run["status"],
            "error_message": run["error_message"],
            "retry_count": run["retry_count"],
            "created_at": run["created_at"].isoformat(),
            "started_at": run["started_at"].isoformat() if run["started_at"] else None,
            "finished_at": run["finished_at"].isoformat() if run["finished_at"] else None,
            "updated_at": run["updated_at"].isoformat(),
            "stages": {
                cast(str, item["node_name"]): {
                    "status": item["status"],
                    "attempt": item["attempt_count"],
                    "reused": item["reused"],
                    "artifacts": item["artifacts"],
                    "error": item["error_message"],
                    "started_at": item["started_at"],
                    "finished_at": item["finished_at"],
                    "created_at": item["created_at"],
                    "updated_at": item["updated_at"],
                }
                for item in stages
            },
        }

    def detail(self, run_id: UUID) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        state = self.state(run_id)
        if state is None:
            return None
        rows: list[dict[str, Any]] = []
        for node in self._definition.topologically_sorted_nodes():
            stage = cast(dict[str, Any], state["stages"].get(node.name, {}))
            rows.append(
                {
                    "id": uuid5(run_id, node.name),
                    "pipeline_run_id": run_id,
                    "recording_id": UUID(state["subject_id"]),
                    "node_name": node.name,
                    "stage_name": node.stage_name,
                    "stage_version": node.stage_version,
                    "required": node.required,
                    "status": stage.get("status", "pending"),
                    "attempt_count": stage.get("attempt", 0),
                    "max_attempts": node.retry_policy.max_attempts or 3,
                    "progress_percent": None,
                    "progress_message": None,
                    "progress_updated_at": None,
                    "generation_run_id": None,
                    "error_message": stage.get("error"),
                    "available_at": stage.get("created_at") or state["created_at"],
                    "started_at": stage.get("started_at"),
                    "finished_at": stage.get("finished_at"),
                    "created_at": stage.get("created_at") or state["created_at"],
                    "updated_at": stage.get("updated_at") or state["updated_at"],
                }
            )
        return state, rows

    def latest_state(self, recording_id: UUID) -> dict[str, Any] | None:
        with self._engine.connect() as connection:
            run_id = connection.execute(
                text("select id from pipeline_runs where recording_id=:recording_id order by created_at desc limit 1"),
                {"recording_id": recording_id},
            ).scalar_one_or_none()
        return None if run_id is None else self.state(cast(UUID, run_id))
