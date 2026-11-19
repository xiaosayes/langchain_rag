import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.db.models import DocumentRecord
from app.db.session import create_session_factory, ensure_web_schema
from app.web.records import (
    AdminRecord,
    AuthSessionRecord,
    ConversationRecord,
    DocumentVersionRecord,
    FeedbackRecord,
    MessageRecord,
    TaskRecord,
)


def assert_utc(value: datetime) -> None:
    assert value.tzinfo is not None
    assert value.utcoffset() == timedelta(0)


def test_web_records_and_legacy_document_columns_round_trip(tmp_path: Path):
    factory = create_session_factory(tmp_path / "web.sqlite3")
    ensure_web_schema(factory)
    with factory() as session:
        session.add(
            DocumentRecord(
                id="doc-1",
                content_hash="abc",
                metadata_json={},
                filename="a.pdf",
                media_type="application/pdf",
                status="ready",
            )
        )
        session.add(
            DocumentVersionRecord(
                id="v1",
                document_id="doc-1",
                version=1,
                path="raw/a.pdf",
                sha256="abc",
                status="ready",
            )
        )
        session.add(
            TaskRecord(
                id="t1", document_id="doc-1", kind="ingest", status="queued"
            )
        )
        session.commit()
        assert session.get(DocumentVersionRecord, "v1").path == "raw/a.pdf"
        assert session.get(TaskRecord, "t1").status == "queued"

    with factory() as session:
        document = session.get(DocumentRecord, "doc-1")
        version = session.get(DocumentVersionRecord, "v1")
        task = session.get(TaskRecord, "t1")
        assert document.deleted is False
        assert document.current_version_id is None
        assert document.error is None
        assert version.error is None
        assert task.attempts == 0
        assert task.max_attempts == 3
        assert task.payload_json == {}
        assert task.lease_owner is None
        assert task.lease_expires_at is None
        for record in (document, version, task):
            assert_utc(record.created_at)
            assert_utc(record.updated_at)


def test_auth_chat_and_feedback_records_round_trip(tmp_path: Path):
    factory = create_session_factory(tmp_path / "web.sqlite3")
    expires_at = datetime.now(UTC) + timedelta(hours=12)
    with factory() as session:
        session.add(
            AdminRecord(id="admin-1", username="admin", password_hash="argon2-hash")
        )
        session.add(
            AuthSessionRecord(
                id="session-1",
                token_hash="sha256-hash",
                csrf_token="csrf-token",
                expires_at=expires_at,
            )
        )
        session.add(ConversationRecord(id="conversation-1", title="新会话"))
        session.add(
            MessageRecord(
                id="message-1",
                conversation_id="conversation-1",
                role="assistant",
                content="answer",
                citations_json=[{"source": "a.pdf"}],
                degraded=True,
                generation_id=None,
            )
        )
        session.add(
            FeedbackRecord(id="feedback-1", message_id="message-1", score=1)
        )
        session.commit()

    with factory() as session:
        admin = session.get(AdminRecord, "admin-1")
        auth_session = session.get(AuthSessionRecord, "session-1")
        conversation = session.get(ConversationRecord, "conversation-1")
        message = session.get(MessageRecord, "message-1")
        feedback = session.get(FeedbackRecord, "feedback-1")
        assert admin.username == "admin"
        assert_utc(admin.created_at)
        assert_utc(auth_session.created_at)
        assert_utc(auth_session.expires_at)
        assert_utc(conversation.created_at)
        assert_utc(conversation.updated_at)
        assert message.citations_json == [{"source": "a.pdf"}]
        assert message.cancelled is False
        assert message.error is None
        assert_utc(message.created_at)
        assert feedback.score == 1
        assert feedback.comment is None
        assert_utc(feedback.created_at)


def test_feedback_message_id_is_unique_and_conversation_delete_cascades(
    tmp_path: Path,
):
    factory = create_session_factory(tmp_path / "web.sqlite3")
    with factory() as session:
        conversation = ConversationRecord(id="conversation-1", title="新会话")
        message = MessageRecord(
            id="message-1",
            conversation=conversation,
            role="assistant",
            content="answer",
        )
        message.feedback = FeedbackRecord(id="feedback-1", score=-1)
        session.add(conversation)
        session.commit()
        session.add(FeedbackRecord(id="feedback-2", message_id="message-1", score=1))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()

        session.delete(conversation)
        session.commit()
        assert session.get(MessageRecord, "message-1") is None
        assert session.get(FeedbackRecord, "feedback-1") is None


@pytest.mark.parametrize(
    "status",
    [
        "uploaded",
        "parsing",
        "chunking",
        "embedding",
        "indexing",
        "ready",
        "failed",
    ],
)
def test_document_version_accepts_required_statuses(tmp_path: Path, status: str):
    factory = create_session_factory(tmp_path / "web.sqlite3")
    with factory() as session:
        session.add(DocumentRecord(id="doc-1", content_hash="abc", metadata_json={}))
        session.add(
            DocumentVersionRecord(
                id="v1",
                document_id="doc-1",
                version=1,
                path="raw/a.pdf",
                sha256="abc",
                status=status,
            )
        )
        session.commit()


@pytest.mark.parametrize("status", ["queued", "running", "succeeded", "failed"])
def test_task_accepts_required_statuses(tmp_path: Path, status: str):
    factory = create_session_factory(tmp_path / "web.sqlite3")
    with factory() as session:
        session.add(DocumentRecord(id="doc-1", content_hash="abc", metadata_json={}))
        session.add(
            TaskRecord(id="t1", document_id="doc-1", kind="ingest", status=status)
        )
        session.commit()


@pytest.mark.parametrize(
    ("record_type", "status"),
    [(DocumentVersionRecord, "unknown"), (TaskRecord, "unknown")],
)
def test_status_check_rejects_unknown_values(
    tmp_path: Path, record_type: type, status: str
):
    factory = create_session_factory(tmp_path / "web.sqlite3")
    with factory() as session:
        session.add(DocumentRecord(id="doc-1", content_hash="abc", metadata_json={}))
        if record_type is DocumentVersionRecord:
            record = DocumentVersionRecord(
                id="v1",
                document_id="doc-1",
                version=1,
                path="raw/a.pdf",
                sha256="abc",
                status=status,
            )
        else:
            record = TaskRecord(
                id="t1", document_id="doc-1", kind="ingest", status=status
            )
        session.add(record)
        with pytest.raises(IntegrityError):
            session.commit()


def test_legacy_documents_are_migrated_additively_and_idempotently(tmp_path: Path):
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

    with factory() as session:
        rows = session.execute(text("PRAGMA table_info(documents)")).all()
        columns = {row[1]: row for row in rows}
        assert set(columns) == {
            "id",
            "content_hash",
            "metadata_json",
            "filename",
            "media_type",
            "status",
            "current_version_id",
            "error",
            "created_at",
            "updated_at",
            "deleted",
        }
        assert session.execute(
            text("SELECT rootpage FROM sqlite_master WHERE name = 'documents'")
        ).scalar_one() == original_root_page
        raw = session.execute(
            text(
                "SELECT content_hash, metadata_json, filename, media_type, status, "
                "current_version_id, error, deleted FROM documents WHERE id = 'legacy-doc'"
            )
        ).one()
        assert raw == ("legacy-hash", '{"legacy": true}', "", "", "ready", None, None, 0)
        legacy = session.get(DocumentRecord, "legacy-doc")
        assert legacy.metadata_json == {"legacy": True}
        assert legacy.deleted is False
        assert_utc(legacy.created_at)
        assert_utc(legacy.updated_at)


def test_web_tables_have_no_roles_or_tenant_columns(tmp_path: Path):
    factory = create_session_factory(tmp_path / "web.sqlite3")
    with factory() as session:
        tables = (
            "admins",
            "auth_sessions",
            "documents",
            "document_versions",
            "tasks",
            "conversations",
            "messages",
            "feedback",
        )
        for table in tables:
            columns = {
                row[1]
                for row in session.execute(
                    text(f"PRAGMA table_info({table})")
                ).all()
            }
            assert not {name for name in columns if "tenant" in name}
            if table == "admins":
                assert not {name for name in columns if "role" in name}
