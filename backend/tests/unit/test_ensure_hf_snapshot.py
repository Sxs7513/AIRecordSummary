from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "ensure_hf_snapshot.py"
    spec = importlib.util.spec_from_file_location("project_ensure_hf_snapshot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ensure_hf_snapshot = _load_script()


def test_ensure_snapshot_uses_complete_local_cache_without_network(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[bool] = []
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    def fake_download(*, repo_id: str, cache_dir: str, local_files_only: bool = False, max_workers: int = 8) -> str:
        del repo_id, cache_dir, max_workers
        calls.append(local_files_only)
        if not local_files_only:
            raise AssertionError("network download must not run for a cached model")
        return str(snapshot)

    monkeypatch.setattr(ensure_hf_snapshot, "snapshot_download", fake_download)

    assert ensure_hf_snapshot.ensure_snapshot("org/model", tmp_path / "cache") == snapshot
    assert calls == [True]


def test_ensure_snapshot_downloads_after_local_cache_miss(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[bool] = []
    snapshot = tmp_path / "downloaded"
    snapshot.mkdir()

    def fake_download(*, repo_id: str, cache_dir: str, local_files_only: bool = False, max_workers: int = 8) -> str:
        del repo_id, cache_dir, max_workers
        calls.append(local_files_only)
        if local_files_only:
            raise FileNotFoundError("cache miss")
        return str(snapshot)

    monkeypatch.setattr(ensure_hf_snapshot, "snapshot_download", fake_download)

    assert ensure_hf_snapshot.ensure_snapshot("org/model", tmp_path / "cache") == snapshot
    assert calls == [True, False]
