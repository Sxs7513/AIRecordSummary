from __future__ import annotations

import os
import queue
import sys
import threading
from importlib import import_module
from pathlib import Path
from time import monotonic
from typing import Protocol, cast

# Configure the Hub before importing it. Plain HTTP is slower than Xet in some
# environments, but behaves predictably behind proxies and honors request timeouts.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")


class SnapshotDownload(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        cache_dir: str,
        local_files_only: bool = False,
        max_workers: int = 8,
    ) -> str: ...


class HuggingFaceHubModule(Protocol):
    snapshot_download: SnapshotDownload


snapshot_download = cast(HuggingFaceHubModule, import_module("huggingface_hub")).snapshot_download


def _log(message: str) -> None:
    print(f"[install-audio-deps] {message}", flush=True)


def _cached_snapshot(repo_id: str, cache_dir: Path) -> str | None:
    try:
        return snapshot_download(repo_id=repo_id, cache_dir=str(cache_dir), local_files_only=True)
    except Exception:
        return None


def _download_with_heartbeat(repo_id: str, cache_dir: Path) -> str:
    completed: queue.Queue[str | BaseException] = queue.Queue(maxsize=1)

    def download() -> None:
        try:
            path = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_dir),
                max_workers=max(1, int(os.environ.get("HF_HUB_DOWNLOAD_WORKERS", "4"))),
            )
        except BaseException as error:
            completed.put(error)
        else:
            completed.put(path)

    worker = threading.Thread(target=download, name="huggingface-snapshot-download", daemon=True)
    worker.start()
    started = monotonic()
    heartbeat_seconds = max(5, int(os.environ.get("HF_HUB_DOWNLOAD_HEARTBEAT_SECONDS", "15")))
    max_wait_seconds = max(heartbeat_seconds, int(os.environ.get("HF_HUB_DOWNLOAD_MAX_WAIT_SECONDS", "3600")))
    while worker.is_alive():
        worker.join(heartbeat_seconds)
        if worker.is_alive():
            elapsed = round(monotonic() - started)
            if elapsed >= max_wait_seconds:
                raise TimeoutError(
                    f"Downloading {repo_id} exceeded {max_wait_seconds}s; check Hugging Face connectivity or increase "
                    "HF_HUB_DOWNLOAD_MAX_WAIT_SECONDS before retrying"
                )
            _log(
                f"Still downloading {repo_id} ({elapsed}s elapsed); "
                "Ctrl+C can safely stop and the next run will resume from cache."
            )
    result = completed.get_nowait()
    if isinstance(result, BaseException):
        raise result
    return result


def ensure_snapshot(repo_id: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Checking local model cache: {repo_id}")
    cached = _cached_snapshot(repo_id, cache_dir)
    if cached is not None:
        path = Path(cached).resolve()
        _log(f"Model is already available locally: {path}")
        return path
    _log(
        f"Local cache miss; downloading {repo_id} "
        f"(etag timeout={os.environ['HF_HUB_ETAG_TIMEOUT']}s, request timeout={os.environ['HF_HUB_DOWNLOAD_TIMEOUT']}s)"
    )
    path = Path(_download_with_heartbeat(repo_id, cache_dir)).resolve()
    _log(f"Model download completed: {path}")
    return path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: ensure_hf_snapshot.py REPO_ID CACHE_DIR")
    ensure_snapshot(sys.argv[1], Path(sys.argv[2]).resolve())


if __name__ == "__main__":
    main()
