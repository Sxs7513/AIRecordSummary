from pathlib import Path

import pytest

from l1_foundation.files import FileStore
from l1_foundation.infrastructure.storage.local import LocalStorage


def test_local_storage_implements_file_store_contract(tmp_path: Path) -> None:
    storage: FileStore = LocalStorage(tmp_path / "uploads")
    source = tmp_path / "source.txt"
    source.write_text("hello", encoding="utf-8")

    key = storage.put_file(source, key="documents/hello.txt")

    assert key == "documents/hello.txt"
    assert storage.get_file_by_key(key).read_text(encoding="utf-8") == "hello"

    storage.delete_file(key)
    with pytest.raises(FileNotFoundError):
        storage.get_file_by_key(key)


@pytest.mark.parametrize("key", ["", ".", "nested/..", "/absolute.txt", "../escape.txt", "nested/../../escape.txt"])
def test_local_storage_rejects_invalid_keys(tmp_path: Path, key: str) -> None:
    storage = LocalStorage(tmp_path / "uploads")
    source = tmp_path / "source.txt"
    source.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError):
        storage.put_file(source, key=key)
