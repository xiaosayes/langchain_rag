from __future__ import annotations

import io
import zipfile
from pathlib import Path
from uuid import uuid4

import pytest
from docx import Document
from lxml.etree import XMLSyntaxError
from openpyxl import Workbook

from app.config import AppSettings, WebSettings
from app.db.models import DocumentRecord
from app.db.session import create_session_factory
from app.documents.service import DocumentService
from app.indexing.repository import IndexRepository
from app.ingestion.text_office import QA_COLUMNS
from app.web.records import DocumentVersionRecord, TaskRecord


class ActivatingBuilder:
    def __init__(self, factory) -> None:
        self.repository = IndexRepository(factory)
        self.fail = False
        self.calls: list[dict[str, list[str]]] = []

    def build(self, documents, progress=None):
        self.calls.append(
            {
                document_id: [block.text for block in blocks]
                for document_id, blocks in documents.items()
            }
        )
        generation_id = str(uuid4())
        self.repository.prepare_generation(generation_id)
        if self.fail:
            self.repository.mark_failed(generation_id)
            raise RuntimeError("forced builder failure")
        self.repository.activate_generation(generation_id, [], [])
        return generation_id


@pytest.fixture
def factory(tmp_path: Path):
    return create_session_factory(tmp_path / "documents.sqlite3")


@pytest.fixture
def settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        app_env="test",
        model_provider="fake",
        web=WebSettings(raw_file_dir=tmp_path / "raw"),
    )


@pytest.fixture
def builder(factory):
    return ActivatingBuilder(factory)


@pytest.fixture
def service(factory, settings, builder) -> DocumentService:
    return DocumentService(
        factory,
        settings,
        builder_factory=lambda current_settings: builder,
    )


def document_state(factory, document_id: str):
    with factory() as session:
        document = session.get(DocumentRecord, document_id)
        return (
            document.current_version_id,
            document.status,
            document.content_hash,
            document.filename,
        )


def test_first_upload_creates_version_task_and_activates_content(
    factory, service: DocumentService, builder: ActivatingBuilder
) -> None:
    version, task = service.upload(
        "notes.txt", io.BytesIO(b"first paragraph\n\nsecond paragraph"), "document"
    )

    assert version.version == 1
    assert version.status == "uploaded"
    assert version.path == str(
        service.storage.raw_file_dir / version.document_id / version.id / "notes.txt"
    )
    assert task.status == "queued"
    assert task.version_id == version.id
    assert document_state(factory, version.document_id) == (
        None,
        "uploaded",
        version.sha256,
        "notes.txt",
    )

    generation_id = service.build_pending("worker-a")

    assert builder.repository.active_generation() == generation_id
    assert document_state(factory, version.document_id) == (
        version.id,
        "ready",
        version.sha256,
        "notes.txt",
    )
    with factory() as session:
        assert session.get(DocumentVersionRecord, version.id).status == "ready"
        saved_task = session.get(TaskRecord, task.id)
        assert saved_task.status == "succeeded"
        assert saved_task.payload_json["generation_id"] == generation_id
    assert builder.calls[-1][version.document_id] == [
        "first paragraph",
        "second paragraph",
    ]


def test_replacement_success_uses_new_version_without_overwriting_old_raw_file(
    factory, service: DocumentService, builder: ActivatingBuilder
) -> None:
    first, _ = service.upload("notes.txt", io.BytesIO(b"accepted old"), "document")
    service.build_pending("worker-a")
    old_path = Path(first.path)
    old_bytes = old_path.read_bytes()

    second, _ = service.upload("notes.txt", io.BytesIO(b"accepted new"), "document")

    assert second.document_id == first.document_id
    assert second.version == 2
    assert Path(second.path) != old_path
    assert document_state(factory, first.document_id)[0] == first.id
    assert old_path.read_bytes() == old_bytes

    service.build_pending("worker-a")

    assert document_state(factory, first.document_id)[0] == second.id
    assert builder.calls[-1][first.document_id] == ["accepted new"]
    assert old_path.read_bytes() == old_bytes
    assert Path(second.path).read_bytes() == b"accepted new"


def test_latest_pending_version_is_built_and_older_task_is_superseded(
    factory, service: DocumentService, builder: ActivatingBuilder
) -> None:
    first, _ = service.upload("notes.txt", io.BytesIO(b"current"), "document")
    service.build_pending("worker-a")
    older, older_task = service.upload(
        "notes.txt", io.BytesIO(b"older pending"), "document"
    )
    newest, newest_task = service.upload(
        "notes.txt", io.BytesIO(b"newest pending"), "document"
    )

    service.build_pending("worker-a")

    assert builder.calls[-1][first.document_id] == ["newest pending"]
    assert document_state(factory, first.document_id)[0] == newest.id
    with factory() as session:
        old_record = session.get(TaskRecord, older_task.id)
        assert old_record.status == "succeeded"
        assert old_record.payload_json["superseded_by"] == newest_task.id
        assert session.get(DocumentVersionRecord, older.id).status == "uploaded"


def docx_bytes(text: str) -> bytes:
    output = io.BytesIO()
    document = Document()
    document.add_paragraph(text)
    document.save(output)
    return output.getvalue()


def corrupt_docx_bytes() -> bytes:
    valid = io.BytesIO(docx_bytes("candidate"))
    output = io.BytesIO()
    with zipfile.ZipFile(valid) as source, zipfile.ZipFile(output, "w") as target:
        for info in source.infolist():
            content = (
                b"<broken>"
                if info.filename == "word/document.xml"
                else source.read(info.filename)
            )
            target.writestr(info, content)
    return output.getvalue()


def test_replacement_parse_failure_preserves_accepted_version_and_active_generation(
    factory, service: DocumentService, builder: ActivatingBuilder
) -> None:
    first, _ = service.upload(
        "manual.docx", io.BytesIO(docx_bytes("accepted old")), "document"
    )
    old_generation = service.build_pending("worker-a")
    old_path = Path(first.path)
    old_bytes = old_path.read_bytes()
    calls_before_failure = len(builder.calls)

    failed, failed_task = service.upload(
        "manual.docx", io.BytesIO(corrupt_docx_bytes()), "document"
    )
    with pytest.raises(XMLSyntaxError):
        service.build_pending("worker-a")

    assert document_state(factory, first.document_id)[:2] == (first.id, "failed")
    assert builder.repository.active_generation() == old_generation
    assert len(builder.calls) == calls_before_failure
    assert old_path.read_bytes() == old_bytes
    assert service.documents_for_tasks([])[first.document_id][0].text == "accepted old"
    with factory() as session:
        assert session.get(DocumentVersionRecord, first.id).status == "ready"
        assert session.get(DocumentVersionRecord, failed.id).status == "failed"
        assert session.get(TaskRecord, failed_task.id).status == "failed"


def test_builder_failure_preserves_accepted_version_and_active_generation(
    factory, service: DocumentService, builder: ActivatingBuilder
) -> None:
    first, _ = service.upload("notes.txt", io.BytesIO(b"accepted old"), "document")
    old_generation = service.build_pending("worker-a")
    old_hash = first.sha256

    failed, failed_task = service.upload(
        "notes.txt", io.BytesIO(b"candidate new"), "document"
    )
    builder.fail = True
    with pytest.raises(RuntimeError, match="forced builder failure"):
        service.build_pending("worker-a")

    assert document_state(factory, first.document_id) == (
        first.id,
        "failed",
        old_hash,
        "notes.txt",
    )
    assert builder.repository.active_generation() == old_generation
    assert Path(first.path).read_bytes() == b"accepted old"
    assert service.documents_for_tasks([])[first.document_id][0].text == "accepted old"
    with factory() as session:
        assert session.get(DocumentVersionRecord, first.id).status == "ready"
        assert session.get(DocumentVersionRecord, failed.id).status == "failed"
        assert session.get(TaskRecord, failed_task.id).status == "failed"


def test_failed_replacement_upload_preserves_current_version_and_generation(
    factory, service: DocumentService, builder: ActivatingBuilder
) -> None:
    first, first_task = service.upload(
        "notes.txt", io.BytesIO(b"accepted old"), "document"
    )
    generation_id = service.build_pending("worker-a")

    with pytest.raises(ValueError, match="TXT"):
        service.upload("notes.txt", io.BytesIO(b"\xff"), "document")

    assert document_state(factory, first.document_id) == (
        first.id,
        "ready",
        first.sha256,
        "notes.txt",
    )
    assert builder.repository.active_generation() == generation_id
    assert Path(first.path).read_bytes() == b"accepted old"
    with factory() as session:
        assert session.get(TaskRecord, first_task.id).status == "succeeded"
        assert len(session.query(DocumentVersionRecord).all()) == 1


def test_document_loader_excludes_deleted_failed_and_evaluation_records(
    factory, service: DocumentService
) -> None:
    cases = (
        ("ordinary", False, "ready", "ordinary.txt", b"included"),
        ("deleted", True, "ready", "deleted.txt", b"deleted"),
        ("failed", False, "failed", "failed.txt", b"failed"),
        ("evaluation", False, "ready", "evaluation.jsonl", b'{"question": "no"}\n'),
    )
    with factory.begin() as session:
        for document_id, deleted, status, filename, content in cases:
            version_id = f"{document_id}-v1"
            path = service.storage.raw_file_dir / document_id / version_id / filename
            path.parent.mkdir(parents=True)
            path.write_bytes(content)
            session.add(
                DocumentRecord(
                    id=document_id,
                    content_hash=document_id,
                    metadata_json={"kind": "document"},
                    filename=filename,
                    current_version_id=version_id,
                    deleted=deleted,
                )
            )
            session.add(
                DocumentVersionRecord(
                    id=version_id,
                    document_id=document_id,
                    version=1,
                    path=str(path),
                    sha256=document_id,
                    status=status,
                )
            )

    documents = service.documents_for_tasks([])

    assert set(documents) == {"ordinary"}
    assert documents["ordinary"][0].text == "included"


def qa_bytes() -> bytes:
    output = io.BytesIO()
    workbook = Workbook()
    workbook.active.append(QA_COLUMNS)
    workbook.active.append(
        ("qa-1", "Question?", "Answer.", "doc-1", "Evidence.", "domain")
    )
    workbook.save(output)
    return output.getvalue()


def test_ready_qa_uses_qa_parser_when_another_document_is_replaced(
    service: DocumentService, builder: ActivatingBuilder
) -> None:
    qa, _ = service.upload("knowledge.xlsx", io.BytesIO(qa_bytes()), "qa")
    ordinary, _ = service.upload("notes.txt", io.BytesIO(b"ordinary old"), "document")
    service.build_pending("worker-a")

    replacement, _ = service.upload(
        "notes.txt", io.BytesIO(b"ordinary new"), "document"
    )
    service.build_pending("worker-a")

    latest = builder.calls[-1]
    assert latest[qa.document_id][0].startswith("Question: Question?")
    assert latest[ordinary.document_id] == ["ordinary new"]
    assert replacement.version == 2
