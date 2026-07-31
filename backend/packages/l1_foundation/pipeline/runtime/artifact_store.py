from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

from pydantic import BaseModel

from l1_foundation.pipeline.contracts import ArtifactPayload, ArtifactRef, PipelineRunId, StageResult, StageRunId


class ArtifactStore:
    """Stores JSON pipeline artifacts under the existing local storage root."""

    def __init__(self, storage_root: Path) -> None:
        self._storage_root = storage_root.resolve()

    def write_json(
        self,
        subject_id: UUID,
        pipeline_run_id: PipelineRunId,
        stage_run_id: StageRunId,
        stage_name: str,
        payload: ArtifactPayload,
        *,
        stage_version: str = "1",
    ) -> ArtifactRef:
        del subject_id
        relative_path = self._json_relative_path(
            pipeline_run_id,
            stage_run_id,
            stage_name,
            stage_version,
            payload.artifact_type,
            payload.artifact_version,
        )
        path = self._storage_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload.data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        temporary = path.with_name(f".{path.name}.{uuid4()}.tmp")
        try:
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return ArtifactRef(
            artifact_type=payload.artifact_type,
            artifact_version=payload.artifact_version,
            uri=relative_path.as_posix(),
            producer_stage=stage_name,
            checksum=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            metadata=payload.metadata,
        )

    def try_restore_json[OutputT: BaseModel](
        self,
        pipeline_run_id: PipelineRunId,
        stage_run_id: StageRunId,
        stage_name: str,
        stage_version: str,
        artifact_type: str,
        output_model: type[OutputT],
        *,
        artifact_version: str = "1",
    ) -> StageResult[OutputT] | None:
        """Restore one complete JSON stage result from its deterministic object key."""
        relative_path = self._json_relative_path(pipeline_run_id, stage_run_id, stage_name, stage_version, artifact_type, artifact_version)
        path = self._storage_root / relative_path
        if not path.is_file():
            return None
        try:
            serialized = path.read_text(encoding="utf-8")
            parsed = json.loads(serialized)
            if not isinstance(parsed, dict):
                return None
            output = output_model.model_validate(parsed)
        except (OSError, ValueError):
            return None
        ref = ArtifactRef(
            artifact_type=artifact_type,
            artifact_version=artifact_version,
            uri=relative_path.as_posix(),
            producer_stage=stage_name,
            checksum=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        )
        return StageResult(output=output, artifacts=(ref,))

    def read_json(self, artifact: ArtifactRef) -> dict[str, object]:
        path = (self._storage_root / artifact.uri).resolve()
        if self._storage_root not in path.parents:
            raise ValueError("Artifact URI escapes storage root")
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"Artifact must contain a JSON object: {artifact.uri}")
        return cast(dict[str, object], parsed)

    @staticmethod
    def _json_relative_path(
        pipeline_run_id: PipelineRunId,
        stage_run_id: StageRunId,
        stage_name: str,
        stage_version: str,
        artifact_type: str,
        artifact_version: str,
    ) -> Path:
        identity = "|".join(
            (
                "audio-processing-artifact-v1",
                str(pipeline_run_id),
                str(stage_run_id),
                stage_name,
                stage_version,
                artifact_type,
                artifact_version,
            )
        )
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return Path("artifacts") / f"{key}.json"
