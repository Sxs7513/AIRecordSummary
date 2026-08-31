from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from l1_foundation.pipeline.contracts import ArtifactPayload, ArtifactRef, PipelineRunId, StageRunId
from l1_foundation.pipeline.runtime.artifact_store import ArtifactStore, build_input_fingerprint


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


def test_json_artifact_persists_and_validates_its_checksum_and_metadata(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    processing_id = PipelineRunId(uuid4())
    stage_run_id = StageRunId(uuid4())
    fingerprint = "input-v1"
    artifact = store.write_json(
        uuid4(),
        processing_id,
        stage_run_id,
        "test_stage",
        ArtifactPayload(artifact_type="test.output", data={"value": "ready"}, metadata={"quality": "normal"}),
        stage_version="3",
        input_fingerprint=fingerprint,
    )

    raw = json.loads((tmp_path / artifact.uri).read_text(encoding="utf-8"))
    assert raw["_artifact"]["checksum"] == {"algorithm": "sha256", "value": artifact.checksum}
    assert raw["_artifact"]["input_fingerprint"] == fingerprint
    assert raw["data"] == {"value": "ready"}

    restored = store.try_restore_json(
        processing_id,
        stage_run_id,
        "test_stage",
        "3",
        "test.output",
        _Output,
        input_fingerprint=fingerprint,
    )
    assert restored is not None
    assert restored.artifacts[0].metadata == {"quality": "normal"}
    assert (
        store.try_restore_json(
            processing_id,
            stage_run_id,
            "test_stage",
            "3",
            "test.output",
            _Output,
            input_fingerprint="different-input",
        )
        is None
    )

    raw["data"]["value"] = "tampered"
    (tmp_path / artifact.uri).write_text(json.dumps(raw), encoding="utf-8")
    assert (
        store.try_restore_json(
            processing_id,
            stage_run_id,
            "test_stage",
            "3",
            "test.output",
            _Output,
            input_fingerprint=fingerprint,
        )
        is None
    )


def test_input_fingerprint_uses_upstream_content_not_artifact_location() -> None:
    first = ArtifactRef(artifact_type="upstream", artifact_version="1", uri="one.json", checksum="sha256-value")
    moved = first.model_copy(update={"uri": "two.json"})
    changed = first.model_copy(update={"checksum": "different-value"})

    fingerprint = build_input_fingerprint("stage", "1", {"literal": 3, "source": first})

    assert fingerprint == build_input_fingerprint("stage", "1", {"source": moved, "literal": 3})
    assert fingerprint != build_input_fingerprint("stage", "1", {"literal": 3, "source": changed})
    assert fingerprint != build_input_fingerprint("stage", "1", {"literal": 4, "source": first})


def test_legacy_artifact_is_adopted_into_a_fingerprinted_envelope(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    processing_id = PipelineRunId(uuid4())
    stage_run_id = StageRunId(uuid4())
    identity = "|".join(
        (
            "audio-processing-artifact-v1",
            str(processing_id),
            str(stage_run_id),
            "test_stage",
            "3",
            "test.output",
            "1",
        )
    )
    legacy_path = tmp_path / "artifacts" / f"{hashlib.sha256(identity.encode()).hexdigest()}.json"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text('{"value":"ready"}', encoding="utf-8")

    restored = store.try_restore_json(
        processing_id,
        stage_run_id,
        "test_stage",
        "3",
        "test.output",
        _Output,
        input_fingerprint="current-input",
        allow_legacy_restore=True,
    )

    assert restored is not None
    adopted = restored.artifacts[0]
    assert isinstance(adopted, ArtifactRef)
    assert adopted.uri != legacy_path.relative_to(tmp_path).as_posix()
    raw = json.loads((tmp_path / adopted.uri).read_text(encoding="utf-8"))
    assert raw["_artifact"]["input_fingerprint"] == "current-input"
