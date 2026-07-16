from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Engine, text

from infrastructure.storage.local import LocalStorage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StorageCleanupResult:
    """Counts emitted after one startup cleanup pass."""

    orphan_pipeline_runs_removed: int = 0
    orphan_stage_runs_removed: int = 0
    recording_directories_removed: int = 0
    normalized_files_removed: int = 0
    artifact_runs_removed: int = 0
    bytes_reclaimed: int = 0


class StorageCleanupService:
    """Remove intermediate files belonging to pipeline runs that are no longer active."""

    def __init__(self, engine: Engine, storage: LocalStorage) -> None:
        self._engine = engine
        self._storage = storage

    def remove_inactive_pipeline_intermediates(self) -> StorageCleanupResult:
        """Remove orphan workflow rows/uploads and keep files needed by active recording runs."""
        orphan_pipeline_runs_removed, orphan_stage_runs_removed = self._remove_orphan_recording_pipeline_runs()
        recording_directories_removed, recording_bytes = self._remove_orphan_recordings()
        active_runs = self._active_runs()
        normalized_files_removed, normalized_bytes = self._remove_normalized_files(active_runs)
        artifact_runs_removed, artifact_bytes = self._remove_artifact_runs(active_runs)
        return StorageCleanupResult(
            orphan_pipeline_runs_removed=orphan_pipeline_runs_removed,
            orphan_stage_runs_removed=orphan_stage_runs_removed,
            recording_directories_removed=recording_directories_removed,
            normalized_files_removed=normalized_files_removed,
            artifact_runs_removed=artifact_runs_removed,
            bytes_reclaimed=recording_bytes + normalized_bytes + artifact_bytes,
        )

    def _remove_orphan_recording_pipeline_runs(self) -> tuple[int, int]:
        """Delete recording-owned workflow rows whose recording aggregate no longer exists."""
        orphan_predicate = """
            pipeline_runs.subject_type = 'recording'
            and not exists (
                select 1 from recordings
                where recordings.id = pipeline_runs.subject_id
            )
        """
        with self._engine.begin() as connection:
            pipeline_count = int(connection.execute(text(f"select count(*) from pipeline_runs where {orphan_predicate}")).scalar_one())
            if pipeline_count == 0:
                return 0, 0
            stage_count = int(
                connection.execute(
                    text(
                        f"""
                        select count(*)
                        from stage_runs
                        join pipeline_runs on pipeline_runs.id = stage_runs.pipeline_run_id
                        where {orphan_predicate}
                        """
                    )
                ).scalar_one()
            )
            connection.execute(
                text(
                    f"""
                    delete from outbox_events
                    where aggregate_type = 'pipeline_run'
                      and aggregate_id in (
                          select pipeline_runs.id
                          from pipeline_runs
                          where {orphan_predicate}
                      )
                    """
                )
            )
            connection.execute(text(f"delete from pipeline_runs where {orphan_predicate}"))
        logger.info(
            "Removed orphan recording pipeline rows: pipeline_runs=%d stage_runs=%d",
            pipeline_count,
            stage_count,
        )
        return pipeline_count, stage_count

    def _remove_orphan_recordings(self) -> tuple[int, int]:
        """Delete upload directories that do not belong to a persisted recording."""
        with self._engine.connect() as connection:
            recording_ids = {str(recording_id) for recording_id in connection.execute(text("select id from recordings")).scalars()}
        root = self._storage.resolve("recordings")
        if not root.is_dir():
            return 0, 0
        removed = 0
        reclaimed = 0
        for recording_directory in self._child_directories(root):
            if recording_directory.name in recording_ids:
                continue
            reclaimed += self._tree_size(recording_directory)
            self._storage.remove_tree(f"recordings/{recording_directory.name}")
            removed += 1
        return removed, reclaimed

    def _active_runs(self) -> set[tuple[str, str]]:
        with self._engine.connect() as connection:
            rows = connection.execute(
                text("select subject_id, id from pipeline_runs where subject_type = 'recording' and status in ('queued', 'running')")
            ).mappings()
            return {(str(row["subject_id"]), str(row["id"])) for row in rows}

    def _remove_normalized_files(self, active_runs: set[tuple[str, str]]) -> tuple[int, int]:
        root = self._storage.resolve("normalized")
        if not root.is_dir():
            return 0, 0
        removed = 0
        reclaimed = 0
        for recording_directory in self._child_directories(root):
            recording_id = recording_directory.name
            for normalized_file in self._child_files(recording_directory):
                run_id = normalized_file.stem
                if (recording_id, run_id) in active_runs:
                    continue
                reclaimed += self._remove_file(normalized_file)
                removed += 1
            self._remove_if_empty(recording_directory)
        return removed, reclaimed

    def _remove_artifact_runs(self, active_runs: set[tuple[str, str]]) -> tuple[int, int]:
        root = self._storage.resolve("artifacts")
        if not root.is_dir():
            return 0, 0
        removed = 0
        reclaimed = 0
        for recording_directory in self._child_directories(root):
            recording_id = recording_directory.name
            for run_directory in self._child_directories(recording_directory):
                run_id = run_directory.name
                if (recording_id, run_id) in active_runs:
                    continue
                reclaimed += self._tree_size(run_directory)
                self._storage.remove_tree(f"artifacts/{recording_id}/{run_id}")
                removed += 1
            self._remove_if_empty(recording_directory)
        return removed, reclaimed

    @staticmethod
    def _child_directories(path: Path) -> list[Path]:
        return [child for child in path.iterdir() if child.is_dir() and not child.is_symlink()]

    @staticmethod
    def _child_files(path: Path) -> list[Path]:
        return [child for child in path.iterdir() if child.is_file() and not child.is_symlink()]

    @staticmethod
    def _remove_file(path: Path) -> int:
        size = path.stat().st_size
        path.unlink()
        return size

    @staticmethod
    def _tree_size(path: Path) -> int:
        return sum(child.stat().st_size for child in path.rglob("*") if child.is_file() and not child.is_symlink())

    @staticmethod
    def _remove_if_empty(path: Path) -> None:
        with suppress(OSError):
            path.rmdir()
