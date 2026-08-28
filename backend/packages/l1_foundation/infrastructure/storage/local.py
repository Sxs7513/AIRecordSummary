from __future__ import annotations

import os
from pathlib import Path
from shutil import copyfileobj
from uuid import uuid4

from l1_foundation.files import FileStore


class LocalStorage(FileStore):
    """Development storage adapter rooted at a single local directory."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def initialize(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        if not key or Path(key).is_absolute():
            raise ValueError("Storage key must be a non-empty relative path")
        candidate = (self._root / key).resolve()
        if candidate == self._root:
            raise ValueError("Storage key must identify a file")
        if self._root not in candidate.parents:
            raise ValueError("Storage key escapes configured root")
        return candidate

    def put_file(self, source: Path, *, key: str) -> str:
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = self._path_for_key(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source == destination:
            return key
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            with source.open("rb") as input_file, temporary.open("wb") as output_file:
                copyfileobj(input_file, output_file, length=1024 * 1024)
                output_file.flush()
                os.fsync(output_file.fileno())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return key

    def get_file_by_key(self, key: str) -> Path:
        path = self._path_for_key(key)
        if not path.is_file():
            raise FileNotFoundError(f"Stored file does not exist: {key}")
        return path

    def delete_file(self, key: str) -> None:
        self._path_for_key(key).unlink(missing_ok=True)
