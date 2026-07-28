from __future__ import annotations

from pathlib import Path
from shutil import rmtree


class LocalStorage:
    """Development storage adapter rooted at a single local directory."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def initialize(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)

    def resolve(self, key: str) -> Path:
        candidate = (self._root / key).resolve()
        if candidate != self._root and self._root not in candidate.parents:
            raise ValueError("Storage key escapes configured root")
        return candidate

    def remove_tree(self, key: str) -> None:
        """Remove a recording-owned directory without allowing path traversal."""
        target = self.resolve(key)
        if target == self._root:
            raise ValueError("Refusing to remove storage root")
        rmtree(target, ignore_errors=True)
