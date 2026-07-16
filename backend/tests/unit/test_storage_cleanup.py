from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, text

from application.storage_cleanup import StorageCleanupService
from infrastructure.storage.local import LocalStorage


def test_cleanup_removes_only_intermediates_for_inactive_pipeline_runs(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "uploads")
    storage.initialize()
    engine = create_engine("sqlite://")
    active_recording_id = "recording-active"
    active_run_id = "run-active"
    finished_recording_id = "recording-finished"
    finished_run_id = "run-finished"
    orphan_recording_id = "recording-orphan"
    orphan_run_id = "run-orphan"
    with engine.begin() as connection:
        connection.execute(text("pragma foreign_keys = on"))
        connection.execute(text("create table recordings (id text primary key)"))
        connection.execute(text("create table pipeline_runs (id text primary key, subject_type text, subject_id text, status text)"))
        connection.execute(
            text(
                """
                create table stage_runs (
                    id text primary key,
                    pipeline_run_id text not null references pipeline_runs(id) on delete cascade
                )
                """
            )
        )
        connection.execute(text("create table outbox_events (aggregate_type text, aggregate_id text)"))
        connection.execute(text("insert into recordings (id) values (:id)"), {"id": active_recording_id})
        connection.execute(text("insert into recordings (id) values (:id)"), {"id": finished_recording_id})
        connection.execute(
            text("insert into pipeline_runs (id, subject_type, subject_id, status) values (:id, 'recording', :subject_id, :status)"),
            {"id": active_run_id, "subject_id": active_recording_id, "status": "running"},
        )
        connection.execute(
            text("insert into pipeline_runs (id, subject_type, subject_id, status) values (:id, 'recording', :subject_id, :status)"),
            {"id": finished_run_id, "subject_id": finished_recording_id, "status": "succeeded"},
        )
        connection.execute(
            text("insert into pipeline_runs (id, subject_type, subject_id, status) values (:id, 'recording', :subject_id, :status)"),
            {"id": orphan_run_id, "subject_id": orphan_recording_id, "status": "running"},
        )
        connection.execute(
            text("insert into stage_runs (id, pipeline_run_id) values ('stage-orphan-1', :run_id), ('stage-orphan-2', :run_id)"),
            {"run_id": orphan_run_id},
        )
        connection.execute(
            text("insert into outbox_events (aggregate_type, aggregate_id) values ('pipeline_run', :run_id)"),
            {"run_id": orphan_run_id},
        )

    _write(storage.resolve(f"normalized/{active_recording_id}/{active_run_id}.wav"), b"active-normalized")
    _write(storage.resolve(f"normalized/{finished_recording_id}/{finished_run_id}.wav"), b"finished-normalized")
    _write(storage.resolve(f"artifacts/{active_recording_id}/{active_run_id}/stage/result.json"), b"active-artifact")
    _write(storage.resolve(f"artifacts/{finished_recording_id}/{finished_run_id}/stage/result.json"), b"finished-artifact")
    _write(storage.resolve(f"artifacts/{orphan_recording_id}/{orphan_run_id}/stage/result.json"), b"orphan-artifact")
    _write(storage.resolve(f"recordings/{active_recording_id}/active.mp3"), b"active-recording")
    _write(storage.resolve(f"recordings/{finished_recording_id}/finished.mp3"), b"finished-recording")
    _write(storage.resolve(f"recordings/{orphan_recording_id}/orphan.mp3"), b"orphan-recording")

    result = StorageCleanupService(engine, storage).remove_inactive_pipeline_intermediates()

    assert storage.resolve(f"normalized/{active_recording_id}/{active_run_id}.wav").is_file()
    assert storage.resolve(f"artifacts/{active_recording_id}/{active_run_id}").is_dir()
    assert not storage.resolve(f"normalized/{finished_recording_id}/{finished_run_id}.wav").exists()
    assert not storage.resolve(f"artifacts/{finished_recording_id}/{finished_run_id}").exists()
    assert not storage.resolve(f"artifacts/{orphan_recording_id}/{orphan_run_id}").exists()
    assert storage.resolve(f"recordings/{active_recording_id}").is_dir()
    assert storage.resolve(f"recordings/{finished_recording_id}").is_dir()
    assert not storage.resolve(f"recordings/{orphan_recording_id}").exists()
    with engine.connect() as connection:
        assert connection.execute(text("select count(*) from pipeline_runs where id = :id"), {"id": orphan_run_id}).scalar_one() == 0
        assert connection.execute(text("select count(*) from stage_runs where pipeline_run_id = :id"), {"id": orphan_run_id}).scalar_one() == 0
        assert connection.execute(text("select count(*) from outbox_events where aggregate_id = :id"), {"id": orphan_run_id}).scalar_one() == 0
    assert result.orphan_pipeline_runs_removed == 1
    assert result.orphan_stage_runs_removed == 2
    assert result.recording_directories_removed == 1
    assert result.normalized_files_removed == 1
    assert result.artifact_runs_removed == 2
    assert result.bytes_reclaimed == (len(b"orphan-recording") + len(b"finished-normalized") + len(b"finished-artifact") + len(b"orphan-artifact"))


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
