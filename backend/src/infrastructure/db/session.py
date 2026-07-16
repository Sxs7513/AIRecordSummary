from __future__ import annotations

from sqlalchemy import Engine, create_engine

from settings import Settings


def create_database_engine(settings: Settings) -> Engine:
    """Create the synchronous engine used by repositories and health checks."""
    return create_engine(settings.sqlalchemy_database_url, pool_pre_ping=True)
