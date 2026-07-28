from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Protocol, cast


class SnapshotDownload(Protocol):
    def __call__(self, *, repo_id: str, cache_dir: str, local_files_only: bool) -> str: ...


class HuggingFaceHubModule(Protocol):
    snapshot_download: SnapshotDownload


def resolve_local_snapshot(model_name: str, cache_dir: Path) -> Path:
    """Resolve a complete Hugging Face snapshot without allowing runtime downloads."""
    resolved_cache_dir = cache_dir.resolve()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        huggingface_hub = cast(HuggingFaceHubModule, import_module("huggingface_hub"))
    except (ImportError, AttributeError) as error:
        raise RuntimeError("huggingface_hub is not installed; run scripts/install_audio_dependencies.sh") from error
    try:
        snapshot_path = huggingface_hub.snapshot_download(
            repo_id=model_name,
            cache_dir=str(resolved_cache_dir),
            local_files_only=True,
        )
    except Exception as error:
        raise RuntimeError(f"Model {model_name} is not available in {resolved_cache_dir}; run scripts/install_audio_dependencies.sh first") from error
    return Path(snapshot_path).resolve()
