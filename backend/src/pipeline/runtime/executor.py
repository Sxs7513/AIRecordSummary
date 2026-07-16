from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, cast

from pydantic import BaseModel

from pipeline.contracts import ArtifactRef, StageContext, StageResult, StageRunId
from pipeline.registry import StageRegistry
from pipeline.runtime.artifact_store import ArtifactStore
from pipeline.runtime.repository import PipelineRepository
from task_runtime.resources import RetryPolicy


class PipelineProgressReporter:
    def __init__(self, repository: PipelineRepository, stage_run_id: StageRunId) -> None:
        self._repository = repository
        self._stage_run_id = stage_run_id

    def report(self, percent: int, message: str) -> None:
        self._repository.update_stage_progress(self._stage_run_id, percent, message)


@dataclass(frozen=True, slots=True)
class StageExecutionResult:
    output: object
    stage_result: dict[str, Any]
    artifacts: tuple[ArtifactRef, ...]
    retry_policy: RetryPolicy


class PipelineExecutor:
    """Business-neutral execution of one declared stage."""

    def __init__(self, repository: PipelineRepository, registry: StageRegistry, artifact_store: ArtifactStore) -> None:
        self._repository = repository
        self._registry = registry
        self._artifact_store = artifact_store

    async def execute(self, stage_run_id: StageRunId, attempt_count: int) -> StageExecutionResult:
        stage_row = self._repository.get_stage(stage_run_id)
        stage = self._registry.get(cast(str, stage_row["stage_name"]), cast(str, stage_row["stage_version"]))
        stage_input: object = self._repository.resolve_stage_input(stage_run_id)
        input_model = getattr(stage, "input_model", None)
        if isinstance(input_model, type) and issubclass(input_model, BaseModel):
            stage_input = input_model.model_validate(stage_input)
        result = await stage.run(
            StageContext(
                subject_id=stage_row["subject_id"],
                pipeline_run_id=stage_row["pipeline_run_id"],
                stage_run_id=stage_run_id,
                attempt_count=attempt_count,
                progress_reporter=PipelineProgressReporter(self._repository, stage_run_id),
            ),
            stage_input,
        )
        artifacts = tuple(
            self._artifact_store.write_json(stage_row["subject_id"], stage_row["pipeline_run_id"], stage_run_id, cast(str, stage_row["node_name"]), artifact)
            for artifact in result.artifacts
        )
        return StageExecutionResult(result.output, self._serialize_result(result), artifacts, stage.retry_policy)

    def retry_policy(self, stage_run_id: StageRunId) -> RetryPolicy:
        row = self._repository.get_stage(stage_run_id)
        return self._registry.get(cast(str, row["stage_name"]), cast(str, row["stage_version"])).retry_policy

    @staticmethod
    def _serialize_result(result: StageResult[Any]) -> dict[str, Any]:
        output = result.output
        if isinstance(output, BaseModel):
            output = output.model_dump(mode="json")
        elif is_dataclass(output):
            output = asdict(cast(Any, output))
        if not isinstance(output, dict):
            output = {"value": output}
        return {"output": output, "artifact_types": [artifact.artifact_type for artifact in result.artifacts]}
