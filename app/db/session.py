"""SQLite engine and session construction."""

from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event
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
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
