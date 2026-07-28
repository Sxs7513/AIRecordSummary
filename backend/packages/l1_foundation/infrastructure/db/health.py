from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from l1_foundation.infrastructure.db.session import create_database_engine
from l1_foundation.settings import Settings


def database_is_healthy(settings: Settings) -> bool:
    """Return whether PostgreSQL accepts a basic query."""
    try:
        with create_database_engine(settings).connect() as connection:
            connection.execute(text("select 1"))
    except SQLAlchemyError:
        return False
    return True
