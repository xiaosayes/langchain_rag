import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from app.db.session import create_session_factory, ensure_web_schema
from app.web.records import ChatRunEventRecord, ChatRunRecord


def test_monitoring_tables_are_added_incrementally_and_idempotently(tmp_path: Path):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE documents ("
            "id VARCHAR PRIMARY KEY, "
            "content_hash VARCHAR NOT NULL, "
            "metadata_json JSON NOT NULL)"
        )
        connection.execute(
            "INSERT INTO documents (id, content_hash, metadata_json) "
            "VALUES ('legacy-doc', 'legacy-hash', '{\"legacy\": true}')"
        )
        original_root_page = connection.execute(
            "SELECT rootpage FROM sqlite_master WHERE name = 'documents'"
        ).fetchone()[0]

    factory = create_session_factory(path)
    ensure_web_schema(factory)
    ensure_web_schema(factory)

    inspector = inspect(factory.kw["bind"])
    assert {"chat_runs", "chat_run_events"} <= set(inspector.get_table_names())
    assert {index["name"] for index in inspector.get_indexes("chat_runs")} >= {
        "ix_chat_runs_created_at",
        "ix_chat_runs_status_created_at",
    }
    assert {index["name"] for index in inspector.get_indexes("chat_run_events")} >= {
        "ix_chat_run_events_run_started",
    }
    with factory() as session:
        assert (
            session.execute(
                text("SELECT content_hash FROM documents WHERE id = 'legacy-doc'")
            ).scalar_one()
            == "legacy-hash"
        )
        assert (
            session.execute(
                text("SELECT rootpage FROM sqlite_master WHERE name = 'documents'")
            ).scalar_one()
            == original_root_page
        )


def test_chat_run_events_cascade_and_enforce_foreign_keys(tmp_path: Path):
    factory = create_session_factory(tmp_path / "index.sqlite3")
    started_at = datetime(2026, 11, 20, 8, 30, tzinfo=UTC)
    with factory() as session:
        session.add(
            ChatRunRecord(
                id="run-1",
                conversation_id="conversation-1",
                question="question",
                status="running",
                created_at=started_at,
            )
        )
        session.add(
            ChatRunEventRecord(
                run_id="run-1",
                stage="dense",
                status="succeeded",
                started_at=started_at,
                ended_at=started_at + timedelta(milliseconds=12),
                duration_ms=12,
                details_json={"count": 3},
            )
        )
        session.commit()
        event_id = session.scalars(
            text("SELECT id FROM chat_run_events WHERE run_id = 'run-1'")
        ).one_or_none()
        assert event_id is not None

        stored = session.get(ChatRunRecord, "run-1")
        assert stored.created_at.utcoffset() == timedelta(0)
        assert stored.events[0].details_json == {"count": 3}
        session.delete(stored)
        session.commit()
        assert session.get(ChatRunEventRecord, event_id) is None

        session.add(
            ChatRunEventRecord(
                run_id="missing",
                stage="dense",
                status="succeeded",
                started_at=started_at,
                ended_at=started_at,
                duration_ms=0,
                details_json={},
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()


@pytest.mark.parametrize("status", ["running", "succeeded", "failed", "cancelled"])
def test_chat_run_accepts_terminal_and_running_statuses(tmp_path: Path, status: str):
    factory = create_session_factory(tmp_path / f"{status}.sqlite3")
    with factory() as session:
        session.add(ChatRunRecord(id=f"run-{status}", question="q", status=status))
        session.commit()
        assert session.get(ChatRunRecord, f"run-{status}").status == status


def test_chat_run_rejects_unknown_status(tmp_path: Path):
    factory = create_session_factory(tmp_path / "index.sqlite3")
    with factory() as session:
        session.add(ChatRunRecord(id="run-1", question="q", status="unknown"))
        with pytest.raises(IntegrityError):
            session.commit()
