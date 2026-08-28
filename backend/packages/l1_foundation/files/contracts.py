from __future__ import annotations

from pathlib import Path
from typing import Protocol


class FileStore(Protocol):
    """Provider-neutral storage for files addressed by opaque keys."""

    def put_file(self, source: Path, *, key: str) -> str:
        """Persist ``source`` at ``key`` and return the persisted key."""
        ...

    def get_file_by_key(self, key: str) -> Path:
        """Return a readable path materialized on the current node."""
        ...

    def delete_file(self, key: str) -> None:
        """Delete one persisted file; missing files are ignored."""
        ...
