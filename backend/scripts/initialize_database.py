from __future__ import annotations

from pathlib import Path

import psycopg
from psycopg import sql

from l1_foundation.infrastructure.db.session import create_database_engine
from l1_foundation.settings import Settings, get_settings
from l2_core.auth.service import AuthService

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASE_SCHEMA_PATH = REPOSITORY_ROOT / "sql" / "base.sql"


def create_database_if_missing(settings: Settings) -> None:
    """Create the configured application database only when it does not exist."""
    admin_url = settings.sqlalchemy_admin_database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    with psycopg.connect(admin_url, autocommit=True) as connection, connection.cursor() as cursor:
        cursor.execute("select 1 from pg_database where datname = %s", (settings.db_name,))
        if cursor.fetchone() is None:
            cursor.execute(sql.SQL("create database {}").format(sql.Identifier(settings.db_name)))


def rebuild_schema(settings: Settings) -> None:
    """Discard the development schema and rebuild the repository's final schema."""
    database_url = settings.sqlalchemy_database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    schema = BASE_SCHEMA_PATH.read_text(encoding="utf-8")
    with psycopg.connect(database_url) as connection:
        connection.execute("select pg_advisory_xact_lock(hashtext('ai_record_summary_schema_init'))")
        connection.execute("drop schema if exists public cascade")
        connection.execute("create schema public")
        connection.execute("set local search_path to public")
        connection.execute(schema)  # pyright: ignore[reportCallIssue, reportArgumentType]  # Static repository-owned SQL.


def initialize_database(settings: Settings) -> None:
    """Create the database when needed, then rebuild its application schema."""
    create_database_if_missing(settings)
    rebuild_schema(settings)
    engine = create_database_engine(settings)
    try:
        AuthService(engine, settings.session_ttl_days).bootstrap_local_admin(
            settings.bootstrap_admin_email, settings.bootstrap_admin_password, settings.bootstrap_workspace_name
        )
    finally:
        engine.dispose()


def main() -> None:
    initialize_database(get_settings())


if __name__ == "__main__":
    main()
