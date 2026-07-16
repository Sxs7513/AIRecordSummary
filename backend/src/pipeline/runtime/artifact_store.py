from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast
from uuid import UUID

from pipeline.contracts import ArtifactPayload, ArtifactRef, PipelineRunId, StageRunId


class ArtifactStore:
    """Stores JSON pipeline artifacts under the existing local storage root."""

    def __init__(self, storage_root: Path) -> None:
        self._storage_root = storage_root.resolve()

    def write_json(self, subject_id: UUID, pipeline_run_id: PipelineRunId, stage_run_id: StageRunId, stage_name: str, payload: ArtifactPayload) -> ArtifactRef:
        relative_path = Path("artifacts") / str(subject_id) / str(pipeline_run_id) / stage_name / f"{payload.artifact_type.replace('.', '_')}.json"
        path = self._storage_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload.data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        path.write_text(serialized, encoding="utf-8")
        return ArtifactRef(
            artifact_type=payload.artifact_type,
            artifact_version=payload.artifact_version,
            uri=relative_path.as_posix(),
            producer_stage=stage_name,
            checksum=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            metadata=payload.metadata,
        )

    def read_json(self, artifact: ArtifactRef) -> dict[str, object]:
        path = (self._storage_root / artifact.uri).resolve()
        if self._storage_root not in path.parents:
            raise ValueError("Artifact URI escapes storage root")
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"Artifact must contain a JSON object: {artifact.uri}")
        return cast(dict[str, object], parsed)
