from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from l1_foundation.pipeline.contracts import ArtifactPayload, ArtifactRef, PipelineRunId, StageRunId
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore


class _Output(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str


def test_json_artifact_key_is_deterministic_and_restorable(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    processing_id = PipelineRunId(uuid4())
    stage_run_id = StageRunId(uuid4())
    payload = ArtifactPayload(artifact_type="test.output", data={"value": "ready"})

    first = store.write_json(uuid4(), processing_id, stage_run_id, "test_stage", payload, stage_version="3")
    second = store.write_json(uuid4(), processing_id, stage_run_id, "test_stage", payload, stage_version="3")
    restored = store.try_restore_json(processing_id, stage_run_id, "test_stage", "3", "test.output", _Output)

    assert first.uri == second.uri
    assert first.uri.startswith("artifacts/")
    assert first.uri.count("/") == 1
    assert restored is not None
    assert restored.output == _Output(value="ready")
    restored_artifact = restored.artifacts[0]
    assert isinstance(restored_artifact, ArtifactRef)
    assert restored_artifact.uri == first.uri


def test_corrupt_json_artifact_is_not_restored(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    processing_id = PipelineRunId(uuid4())
    stage_run_id = StageRunId(uuid4())
    artifact = store.write_json(
        uuid4(),
        processing_id,
        stage_run_id,
        "test_stage",
        ArtifactPayload(artifact_type="test.output", data={"value": "ready"}),
        stage_version="3",
    )
    (tmp_path / artifact.uri).write_text("not-json", encoding="utf-8")

    assert store.try_restore_json(processing_id, stage_run_id, "test_stage", "3", "test.output", _Output) is None
