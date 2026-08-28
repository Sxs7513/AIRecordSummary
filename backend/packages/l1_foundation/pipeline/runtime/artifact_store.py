from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from uuid import UUID

from pydantic import BaseModel

from l1_foundation.files import FileStore
from l1_foundation.infrastructure.storage.local import LocalStorage
from l1_foundation.pipeline.contracts import ArtifactPayload, ArtifactRef, PipelineRunId, StageResult, StageRunId


class ArtifactStore:
    """Stores JSON pipeline artifacts under the existing local storage root."""

    def __init__(self, file_store: FileStore | Path) -> None:
        # Path compatibility keeps isolated stage tests concise while production
        # composition always injects the provider-neutral FileStore.
        self._file_store = LocalStorage(file_store) if isinstance(file_store, Path) else file_store

    @property
    def file_store(self) -> FileStore:
        return self._file_store

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
        serialized = json.dumps(payload.data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with TemporaryDirectory(prefix="artifact-store-") as temporary_directory:
            temporary = Path(temporary_directory) / "artifact.json"
            temporary.write_text(serialized, encoding="utf-8")
            self._file_store.put_file(temporary, key=relative_path.as_posix())
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
        try:
            path = self._file_store.get_file_by_key(relative_path.as_posix())
        except FileNotFoundError:
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
        path = self._file_store.get_file_by_key(artifact.uri)
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
