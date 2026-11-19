"""SQLite engine and session construction."""

from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from app.db import models as _models  # noqa: F401
from app.db.base import Base

_DOCUMENT_COLUMNS = (
    ("filename", "VARCHAR NOT NULL DEFAULT ''"),
    ("media_type", "VARCHAR NOT NULL DEFAULT ''"),
    ("status", "VARCHAR(16) NOT NULL DEFAULT 'ready'"),
    ("current_version_id", "VARCHAR DEFAULT NULL"),
    ("error", "TEXT DEFAULT NULL"),
    ("created_at", "DATETIME NOT NULL DEFAULT '1970-01-01 00:00:00'"),
    ("updated_at", "DATETIME NOT NULL DEFAULT '1970-01-01 00:00:00'"),
    ("deleted", "BOOLEAN NOT NULL DEFAULT 0"),
)


def ensure_web_schema(session_factory: sessionmaker[Session]) -> None:
    """Create web tables and add missing document columns without rebuilding data."""
    from app.web import records as _records  # noqa: F401

    engine = session_factory.kw["bind"]
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        rows = connection.exec_driver_sql("PRAGMA table_info(documents)").all()
        existing_columns = {row[1] for row in rows}
        for name, definition in _DOCUMENT_COLUMNS:
            if name not in existing_columns:
                connection.execute(
                    text(f"ALTER TABLE documents ADD COLUMN {name} {definition}")
                )


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
    session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    ensure_web_schema(session_factory)
    return session_factory
