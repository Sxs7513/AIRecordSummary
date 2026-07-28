from __future__ import annotations

from pathlib import Path
from typing import cast

import psycopg

from l1_foundation.settings import Settings, get_settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_SCHEMA_PATH = REPOSITORY_ROOT / "sql" / "evaluation.sql"
REQUIRED_BASE_TABLES = ("users", "workspaces", "recordings")


def create_missing_evaluation_tables(settings: Settings) -> None:
    """Install the additive evaluation schema into the application database."""
    database_url = settings.sqlalchemy_database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    schema = EVALUATION_SCHEMA_PATH.read_text(encoding="utf-8")

    with psycopg.connect(database_url) as connection:
        connection.execute("create schema if not exists public")
        connection.execute("set local search_path to public")
        connection.execute("select pg_advisory_xact_lock(hashtext('ai_record_summary_evaluation_schema_init'))")

        missing_tables: list[str] = []
        for table_name in REQUIRED_BASE_TABLES:
            row = connection.execute("select to_regclass(%s)", (f"public.{table_name}",)).fetchone()
            if row is None or cast(str | None, row[0]) is None:
                missing_tables.append(table_name)

        if missing_tables:
            joined_names = ", ".join(missing_tables)
            raise RuntimeError(
                f"Evaluation schema requires the base schema tables: {joined_names}. "
                "Run `npm run db:init` before `npm run db:init:evaluation`."
            )

        connection.execute(schema)  # pyright: ignore[reportCallIssue, reportArgumentType]  # Static repository-owned SQL.


def main() -> None:
    create_missing_evaluation_tables(get_settings())


if __name__ == "__main__":
    main()
