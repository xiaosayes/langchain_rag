"""SQLite engine and session construction."""

from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from app.db import models as _models  # noqa: F401
from app.db.base import Base


def create_session_factory(path: Path) -> sessionmaker[Session]:
    """Create a session factory with SQLite integrity and WAL enabled."""
    engine = create_engine(f"sqlite:///{path.as_posix()}")

    def configure_sqlite(dbapi_connection: Any, _connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
        finally:
            cursor.close()

    event.listen(engine, "connect", configure_sqlite)
    Base.metadata.create_all(engine)
    # create_all does not alter existing tables. Add only the nullable compatibility
    # column so pre-fingerprint generations remain readable but explicitly unknown.
    with engine.begin() as connection:
        columns = inspect(connection).get_columns("index_generations")
        if "vector_space_fingerprint" not in {column["name"] for column in columns}:
            connection.execute(
                text(
                    "ALTER TABLE index_generations "
                    "ADD COLUMN vector_space_fingerprint VARCHAR(64)"
                )
            )
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
