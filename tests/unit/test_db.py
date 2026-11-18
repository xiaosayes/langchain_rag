from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.db.models import DocumentRecord, IndexGenerationRecord, ParentChunkRecord
from app.db.session import create_session_factory


def make_parent(generation_id: str, text_value: str = "差旅标准为每天500元"):
    return ParentChunkRecord(
        id="parent-1",
        generation_id=generation_id,
        document_id="doc-1",
        ordinal=0,
        text=text_value,
        content_hash="abc",
        metadata_json={},
    )


def test_generation_and_parent_records_round_trip(tmp_path: Path):
    session_factory = create_session_factory(tmp_path / "rag.sqlite3")
    with session_factory() as session:
        generation = IndexGenerationRecord(id="gen-1", status="active")
        session.add(generation)
        session.add(
            ParentChunkRecord(
                id="parent-1",
                generation_id="gen-1",
                document_id="doc-1",
                ordinal=0,
                text="差旅标准为每天500元",
                content_hash="abc",
                metadata_json={},
            )
        )
        session.commit()
        stored = session.get(ParentChunkRecord, ("parent-1", "gen-1"))
        assert stored.text == "差旅标准为每天500元"


def test_composite_parent_key_keeps_old_generations_queryable(tmp_path: Path):
    session_factory = create_session_factory(tmp_path / "rag.sqlite3")
    with session_factory() as session:
        session.add_all(
            [
                IndexGenerationRecord(id="gen-1", status="retired"),
                IndexGenerationRecord(id="gen-2", status="active"),
                make_parent("gen-1", "旧标准"),
                make_parent("gen-2", "新标准"),
            ]
        )
        session.commit()

        old = session.get(ParentChunkRecord, ("parent-1", "gen-1"))
        new = session.get(ParentChunkRecord, ("parent-1", "gen-2"))
        assert old.text == "旧标准"
        assert new.text == "新标准"


def test_document_record_round_trip(tmp_path: Path):
    session_factory = create_session_factory(tmp_path / "rag.sqlite3")
    with session_factory() as session:
        session.add(
            DocumentRecord(
                id="doc-1",
                content_hash="document-hash",
                metadata_json={"file_name": "policy.txt"},
            )
        )
        session.commit()

        stored = session.get(DocumentRecord, "doc-1")
        assert stored.content_hash == "document-hash"
        assert stored.metadata_json == {"file_name": "policy.txt"}


def test_sqlite_enables_foreign_keys_and_wal(tmp_path: Path):
    session_factory = create_session_factory(tmp_path / "rag.sqlite3")
    with session_factory() as session:
        assert session.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
        assert session.execute(text("PRAGMA journal_mode")).scalar_one() == "wal"


def test_parent_requires_existing_generation(tmp_path: Path):
    session_factory = create_session_factory(tmp_path / "rag.sqlite3")
    with session_factory() as session:
        session.add(make_parent("missing"))
        with pytest.raises(IntegrityError):
            session.commit()


@pytest.mark.parametrize("status", ["building", "active", "retired", "failed"])
def test_generation_status_accepts_allowed_values(tmp_path: Path, status: str):
    session_factory = create_session_factory(tmp_path / "rag.sqlite3")
    with session_factory() as session:
        session.add(IndexGenerationRecord(id=f"gen-{status}", status=status))
        session.commit()
        assert session.get(IndexGenerationRecord, f"gen-{status}").status == status


def test_generation_status_rejects_unknown_value(tmp_path: Path):
    session_factory = create_session_factory(tmp_path / "rag.sqlite3")
    with session_factory() as session:
        session.add(IndexGenerationRecord(id="gen-1", status="unknown"))
        with pytest.raises(IntegrityError):
            session.commit()
