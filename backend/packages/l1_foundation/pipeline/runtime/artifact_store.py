from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
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
        input_fingerprint: str = "",
    ) -> ArtifactRef:
        del subject_id
        relative_path = self._json_relative_path(
            pipeline_run_id,
            stage_run_id,
            stage_name,
            stage_version,
            payload.artifact_type,
            payload.artifact_version,
            input_fingerprint,
        )
        payload_serialized = _canonical_json(payload.data)
        checksum = hashlib.sha256(payload_serialized.encode("utf-8")).hexdigest()
        envelope = {
            "_artifact": {
                "schema_version": 2,
                "artifact_type": payload.artifact_type,
                "artifact_version": payload.artifact_version,
                "producer_stage": stage_name,
                "input_fingerprint": input_fingerprint,
                "checksum": {"algorithm": "sha256", "value": checksum},
                "metadata": payload.metadata,
            },
            "data": payload.data,
        }
        serialized = _canonical_json(envelope)
        with TemporaryDirectory(prefix="artifact-store-") as temporary_directory:
            temporary = Path(temporary_directory) / "artifact.json"
            temporary.write_text(serialized, encoding="utf-8")
            self._file_store.put_file(temporary, key=relative_path.as_posix())
        return ArtifactRef(
            artifact_type=payload.artifact_type,
            artifact_version=payload.artifact_version,
            uri=relative_path.as_posix(),
            producer_stage=stage_name,
            checksum=checksum,
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
        input_fingerprint: str = "",
        allow_legacy_restore: bool = False,
    ) -> StageResult[OutputT] | None:
        """Restore one complete JSON stage result from its deterministic object key."""
        current_path = self._json_relative_path(
            pipeline_run_id,
            stage_run_id,
            stage_name,
            stage_version,
            artifact_type,
            artifact_version,
            input_fingerprint,
        )
        relative_path = current_path
        try:
            path = self._file_store.get_file_by_key(relative_path.as_posix())
        except FileNotFoundError:
            if not allow_legacy_restore:
                return None
            relative_path = self._legacy_json_relative_path(
                pipeline_run_id,
                stage_run_id,
                stage_name,
                stage_version,
                artifact_type,
                artifact_version,
            )
            try:
                path = self._file_store.get_file_by_key(relative_path.as_posix())
            except FileNotFoundError:
                return None
        try:
            serialized = path.read_text(encoding="utf-8")
            parsed: object = json.loads(serialized)
            is_legacy = isinstance(parsed, dict) and "_artifact" not in parsed
            data, checksum, metadata = self._unpack_json(
                cast(object, parsed),
                expected_artifact_type=artifact_type,
                expected_artifact_version=artifact_version,
                expected_stage_name=stage_name,
                expected_input_fingerprint=input_fingerprint,
            )
            output = output_model.model_validate(data)
        except (OSError, ValueError):
            return None
        if is_legacy:
            self._write_envelope(
                current_path,
                data,
                artifact_type=artifact_type,
                artifact_version=artifact_version,
                producer_stage=stage_name,
                input_fingerprint=input_fingerprint,
                checksum=checksum,
                metadata=metadata,
            )
            relative_path = current_path
        ref = ArtifactRef(
            artifact_type=artifact_type,
            artifact_version=artifact_version,
            uri=relative_path.as_posix(),
            producer_stage=stage_name,
            checksum=checksum,
            metadata=metadata,
        )
        return StageResult(output=output, artifacts=(ref,))

    def _write_envelope(
        self,
        relative_path: Path,
        data: dict[str, object],
        *,
        artifact_type: str,
        artifact_version: str,
        producer_stage: str,
        input_fingerprint: str,
        checksum: str,
        metadata: dict[str, Any],
    ) -> None:
        envelope = {
            "_artifact": {
                "schema_version": 2,
                "artifact_type": artifact_type,
                "artifact_version": artifact_version,
                "producer_stage": producer_stage,
                "input_fingerprint": input_fingerprint,
                "checksum": {"algorithm": "sha256", "value": checksum},
                "metadata": metadata,
            },
            "data": data,
        }
        with TemporaryDirectory(prefix="artifact-store-") as temporary_directory:
            temporary = Path(temporary_directory) / "artifact.json"
            temporary.write_text(_canonical_json(envelope), encoding="utf-8")
            self._file_store.put_file(temporary, key=relative_path.as_posix())

    def read_json(self, artifact: ArtifactRef) -> dict[str, object]:
        path = self._file_store.get_file_by_key(artifact.uri)
        parsed = json.loads(path.read_text(encoding="utf-8"))
        data, checksum, _metadata = self._unpack_json(parsed)
        if artifact.checksum is not None and artifact.checksum != checksum:
            raise ValueError(f"Artifact checksum does not match its reference: {artifact.uri}")
        return data

    @staticmethod
    def _unpack_json(
        parsed: object,
        *,
        expected_artifact_type: str | None = None,
        expected_artifact_version: str | None = None,
        expected_stage_name: str | None = None,
        expected_input_fingerprint: str | None = None,
    ) -> tuple[dict[str, object], str, dict[str, Any]]:
        if not isinstance(parsed, dict):
            raise ValueError("Artifact must contain a JSON object")
        container = cast(dict[str, object], parsed)
        header_value = container.get("_artifact")
        if header_value is None:
            data = container
            return data, hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest(), {}
        if not isinstance(header_value, dict):
            raise ValueError("Unsupported artifact envelope")
        header = cast(dict[str, object], header_value)
        if header.get("schema_version") != 2:
            raise ValueError("Unsupported artifact envelope")
        data_value = container.get("data")
        if not isinstance(data_value, dict):
            raise ValueError("Artifact envelope data must contain a JSON object")
        data = cast(dict[str, object], data_value)
        checks = (
            ("artifact_type", expected_artifact_type),
            ("artifact_version", expected_artifact_version),
            ("producer_stage", expected_stage_name),
            ("input_fingerprint", expected_input_fingerprint),
        )
        if any(expected is not None and header.get(field) != expected for field, expected in checks):
            raise ValueError("Artifact envelope identity does not match its lookup key")
        checksum_value = header.get("checksum")
        if not isinstance(checksum_value, dict):
            raise ValueError("Artifact envelope has no supported checksum")
        checksum_descriptor = cast(dict[str, object], checksum_value)
        if checksum_descriptor.get("algorithm") != "sha256":
            raise ValueError("Artifact envelope has no supported checksum")
        checksum = hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()
        if checksum_descriptor.get("value") != checksum:
            raise ValueError("Artifact checksum mismatch")
        metadata_value = header.get("metadata", {})
        if not isinstance(metadata_value, dict):
            raise ValueError("Artifact metadata must contain a JSON object")
        return data, checksum, cast(dict[str, Any], metadata_value)

    @staticmethod
    def _json_relative_path(
        pipeline_run_id: PipelineRunId,
        stage_run_id: StageRunId,
        stage_name: str,
        stage_version: str,
        artifact_type: str,
        artifact_version: str,
        input_fingerprint: str,
    ) -> Path:
        identity = "|".join(
            (
                "audio-processing-artifact-v2",
                str(pipeline_run_id),
                str(stage_run_id),
                stage_name,
                stage_version,
                artifact_type,
                artifact_version,
                input_fingerprint,
            )
        )
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return Path("artifacts") / f"{key}.json"

    @staticmethod
    def _legacy_json_relative_path(
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
        return Path("artifacts") / f"{hashlib.sha256(identity.encode('utf-8')).hexdigest()}.json"


def build_input_fingerprint(
    stage_name: str,
    stage_version: str,
    payload: dict[str, object],
    cache_config: object = None,
) -> str:
    """Hash literal inputs, upstream content identities, and stage cache configuration."""
    identity = {
        "schema_version": 1,
        "stage_name": stage_name,
        "stage_version": stage_version,
        "inputs": _fingerprint_value(payload),
        "cache_config": _fingerprint_value(cache_config),
    }
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def _fingerprint_value(value: object) -> object:
    if isinstance(value, ArtifactRef):
        content_identity = value.checksum if value.checksum is not None else f"uri:{value.uri}"
        return {
            "artifact_type": value.artifact_type,
            "artifact_version": value.artifact_version,
            "content_identity": content_identity,
        }
    if isinstance(value, BaseModel):
        return _fingerprint_value(value.model_dump(mode="python"))
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        return {str(key): _fingerprint_value(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], value)
        return [_fingerprint_value(item) for item in sequence]
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
