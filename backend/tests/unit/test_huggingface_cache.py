from __future__ import annotations

from pathlib import Path

import pytest

import l1_foundation.infrastructure.huggingface as huggingface_cache


class FakeHuggingFaceHub:
    def __init__(self, snapshot_path: Path) -> None:
        self.snapshot_path = snapshot_path
        self.calls: list[tuple[str, str, bool]] = []

    def snapshot_download(self, *, repo_id: str, cache_dir: str, local_files_only: bool) -> str:
        self.calls.append((repo_id, cache_dir, local_files_only))
        return str(self.snapshot_path)


def test_resolve_local_snapshot_disables_runtime_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot_path = tmp_path / "snapshots" / "revision"
    hub = FakeHuggingFaceHub(snapshot_path)

    monkeypatch.setattr(huggingface_cache, "import_module", lambda name: hub)

    resolved = huggingface_cache.resolve_local_snapshot("Qwen/example", tmp_path / "cache")

    assert resolved == snapshot_path.resolve()
    assert hub.calls == [("Qwen/example", str((tmp_path / "cache").resolve()), True)]


def test_resolve_local_snapshot_reports_missing_install_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class MissingHuggingFaceHub:
        @staticmethod
        def snapshot_download(*, repo_id: str, cache_dir: str, local_files_only: bool) -> str:
            raise FileNotFoundError(repo_id, cache_dir, local_files_only)

    monkeypatch.setattr(huggingface_cache, "import_module", lambda name: MissingHuggingFaceHub())

    with pytest.raises(RuntimeError, match="scripts/install_audio_dependencies.sh"):
        huggingface_cache.resolve_local_snapshot("Qwen/missing", tmp_path / "cache")
