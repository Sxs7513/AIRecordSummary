from pathlib import Path
from typing import Any

from l1_foundation.settings import Settings
from scripts import initialize_database as database_script


def test_base_schema_declares_only_the_final_tables_and_indexes() -> None:
    schema = database_script.BASE_SCHEMA_PATH.read_text(encoding="utf-8").lower()

    assert database_script.REPOSITORY_ROOT.name == "AIRecordSummary"
    assert "drop table" not in schema
    assert "delete from" not in schema
    assert "schema_migrations" not in schema
    assert "create table if not exists recording_speaker_mappings" in schema
    assert "create table " not in schema.replace("create table if not exists ", "")
    assert "create index " not in schema.replace("create index if not exists ", "")
    assert "owner_user_id uuid references users(id) on delete set null" in schema
    assert "alter table conversations alter column owner_user_id drop not null" in schema
    assert "client_creation_id uuid" in schema
    assert "conversations_owner_client_creation_idx" in schema
    recordings = schema.split("create table if not exists recordings (", 1)[1].split(");", 1)[0]
    assert "owner_user_id uuid not null references users(id) on delete restrict" in recordings


def test_rebuild_schema_drops_old_runtime_tables_with_public_schema(monkeypatch: Any, tmp_path: Path) -> None:
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

    def fake_connect(_url: str) -> FakeConnection:
        return FakeConnection()

    monkeypatch.setattr(database_script.psycopg, "connect", fake_connect)
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

    database_script.rebuild_schema(settings)

    assert statements[:4] == [
        "select pg_advisory_xact_lock(hashtext('ai_record_summary_schema_init'))",
        "drop schema if exists public cascade",
        "create schema public",
        "set local search_path to public",
    ]
    assert "create table if not exists example" in statements[-1]
