from pathlib import Path
from typing import Any, cast

import pytest

from l1_foundation.settings import Settings
from scripts import initialize_evaluation_database as evaluation_database_script


def _settings() -> Settings:
    return Settings.model_validate(
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


def test_evaluation_schema_is_additive_and_separate_from_base_schema() -> None:
    schema = evaluation_database_script.EVALUATION_SCHEMA_PATH.read_text(encoding="utf-8").lower()

    assert evaluation_database_script.REPOSITORY_ROOT.name == "AIRecordSummary"
    assert evaluation_database_script.EVALUATION_SCHEMA_PATH.name == "evaluation.sql"
    assert "drop table" not in schema
    assert "delete from" not in schema
    assert "create table " not in schema.replace("create table if not exists ", "")
    assert "create index " not in schema.replace("create index if not exists ", "")
    assert "create unique index " not in schema.replace("create unique index if not exists ", "")
    assert "create table if not exists evaluation_datasets" in schema
    assert "create table if not exists evaluation_annotations" in schema
    assert "create table if not exists evaluation_dataset_versions" in schema
    assert "create table if not exists evaluation_cases" in schema
    assert "create table if not exists training_runs" in schema
    assert "create table if not exists model_versions" in schema
    assert "create table if not exists evaluation_runs" in schema
    assert "create table if not exists evaluation_run_models" in schema
    assert "create table if not exists evaluation_case_results" in schema
    assert "create table if not exists evaluation_metric_values" in schema
    assert "model_version_ids uuid[]" not in schema


def test_initializer_checks_base_tables_then_executes_evaluation_schema(monkeypatch: Any, tmp_path: Path) -> None:
    statements: list[tuple[str, object | None]] = []
    checked_tables: list[str] = []

    class FakeResult:
        def __init__(self, row: tuple[str] | None = None) -> None:
            self._row = row

        def fetchone(self) -> tuple[str] | None:
            return self._row

    class FakeConnection:
        def __enter__(self) -> FakeConnection:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, statement: object, parameters: object | None = None) -> FakeResult:
            statement_text = str(statement)
            statements.append((statement_text, parameters))
            if statement_text == "select to_regclass(%s)":
                assert isinstance(parameters, tuple)
                typed_parameters = cast(tuple[object, ...], parameters)
                assert len(typed_parameters) == 1
                table_name = typed_parameters[0]
                assert isinstance(table_name, str)
                checked_tables.append(table_name)
                return FakeResult((table_name,))
            return FakeResult()

    def fake_connect(_url: str) -> FakeConnection:
        return FakeConnection()

    monkeypatch.setattr(evaluation_database_script, "EVALUATION_SCHEMA_PATH", tmp_path / "evaluation.sql")
    evaluation_database_script.EVALUATION_SCHEMA_PATH.write_text(
        "create table if not exists evaluation_example (id integer);",
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluation_database_script.psycopg, "connect", fake_connect)

    evaluation_database_script.create_missing_evaluation_tables(_settings())

    assert [statement for statement, _parameters in statements[:3]] == [
        "create schema if not exists public",
        "set local search_path to public",
        "select pg_advisory_xact_lock(hashtext('ai_record_summary_evaluation_schema_init'))",
    ]
    assert checked_tables == ["public.users", "public.workspaces", "public.recordings"]
    assert "create table if not exists evaluation_example" in statements[-1][0]


def test_initializer_rejects_database_without_base_schema(monkeypatch: Any, tmp_path: Path) -> None:
    executed_schema = False

    class FakeResult:
        def fetchone(self) -> tuple[None]:
            return (None,)

    class FakeConnection:
        def __enter__(self) -> FakeConnection:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, statement: object, _parameters: object | None = None) -> FakeResult:
            nonlocal executed_schema
            if str(statement).startswith("create table"):
                executed_schema = True
            return FakeResult()

    def fake_connect(_url: str) -> FakeConnection:
        return FakeConnection()

    monkeypatch.setattr(evaluation_database_script, "EVALUATION_SCHEMA_PATH", tmp_path / "evaluation.sql")
    evaluation_database_script.EVALUATION_SCHEMA_PATH.write_text(
        "create table if not exists evaluation_example (id integer);",
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluation_database_script.psycopg, "connect", fake_connect)

    with pytest.raises(RuntimeError, match=r"npm run db:init.*npm run db:init:evaluation"):
        evaluation_database_script.create_missing_evaluation_tables(_settings())

    assert not executed_schema
