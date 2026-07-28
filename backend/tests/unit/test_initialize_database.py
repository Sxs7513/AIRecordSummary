from pathlib import Path
from typing import Any

from l1_foundation.settings import Settings
from scripts import initialize_database as database_script


def test_base_schema_only_creates_missing_tables_and_indexes() -> None:
    schema = database_script.BASE_SCHEMA_PATH.read_text(encoding="utf-8").lower()

    assert database_script.REPOSITORY_ROOT.name == "AIRecordSummary"
    assert "drop table" not in schema
    assert "delete from" not in schema
    assert "schema_migrations" not in schema
    assert "create table if not exists recording_speaker_mappings" in schema
    assert "create table " not in schema.replace("create table if not exists ", "")
    assert "create index " not in schema.replace("create index if not exists ", "")


def test_create_missing_tables_selects_public_schema_before_baseline(monkeypatch: Any, tmp_path: Path) -> None:
    statements: list[str] = []

    class FakeConnection:
        def __enter__(self) -> FakeConnection:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, statement: object) -> None:
            statements.append(str(statement))

    monkeypatch.setattr(database_script, "BASE_SCHEMA_PATH", tmp_path / "base.sql")
    database_script.BASE_SCHEMA_PATH.write_text("create table if not exists example (id integer);", encoding="utf-8")
    monkeypatch.setattr(database_script.psycopg, "connect", lambda _url: FakeConnection())
    settings = Settings.model_validate(
        {
            "DB_HOST": "localhost",
            "DB_PORT": 5432,
            "DB_USER": "postgres",
            "DB_PASSWORD": "postgres",
            "DB_NAME": "test",
            "DB_ADMIN_DATABASE": "postgres",
            "DB_SSL": False,
        }
    )

    database_script.create_missing_tables(settings)

    assert statements[:2] == ["create schema if not exists public", "set local search_path to public"]
    assert "create table if not exists example" in statements[-1]
