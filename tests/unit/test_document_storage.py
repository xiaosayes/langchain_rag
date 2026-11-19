from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest
from openpyxl import Workbook

from app.documents.storage import DocumentStorage
from app.ingestion.text_office import QA_COLUMNS


class ChunkedStream:
    def __init__(self, content: bytes, chunk_size: int = 7) -> None:
        self._content = io.BytesIO(content)
        self.chunk_size = chunk_size
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        assert size > 0, "uploads must be streamed with bounded reads"
        return self._content.read(min(size, self.chunk_size))


class RepeatedBytes:
    def __init__(self, byte: bytes, remaining: int) -> None:
        self.byte = byte
        self.remaining = remaining

    def read(self, size: int = -1) -> bytes:
        assert size > 0
        count = min(size, self.remaining)
        self.remaining -= count
        return self.byte * count


@pytest.fixture
def storage(tmp_path: Path) -> DocumentStorage:
    return DocumentStorage(tmp_path / "raw")


def office_bytes(kind: str) -> bytes:
    output = io.BytesIO()
    member = "word/document.xml" if kind == "docx" else "xl/workbook.xml"
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr(member, "<document/>")
    return output.getvalue()


def qa_bytes(headers: tuple[str, ...] = QA_COLUMNS) -> bytes:
    output = io.BytesIO()
    workbook = Workbook()
    workbook.active.append(headers)
    workbook.save(output)
    return output.getvalue()


@pytest.mark.parametrize(
    "filename",
    [
        "../secret.txt",
        "..\\secret.txt",
        "/tmp/secret.txt",
        "C:\\secret.txt",
        "C:secret.txt",
        "bad\nname.txt",
        "bad\x7fname.txt",
        "CON.txt",
        "com1.txt",
        "NUL",
        ".",
        "name.txt ",
        "evaluation.jsonl",
        "malware.exe",
        "notes.md",
    ],
)
def test_rejects_unsafe_or_unsupported_filenames(
    storage: DocumentStorage, filename: str
) -> None:
    with pytest.raises(ValueError):
        storage.store_upload(filename, io.BytesIO(b"plain text"), "document")


def test_qa_accepts_only_xlsx(storage: DocumentStorage) -> None:
    with pytest.raises(ValueError, match="Q&A"):
        storage.store_upload("qa.txt", io.BytesIO(b"not a workbook"), "qa")


def test_streams_hash_and_stores_under_versioned_path(
    storage: DocumentStorage,
) -> None:
    content = "first paragraph\n\n第二段".encode()
    stream = ChunkedStream(content)

    stored = storage.store_upload("notes.txt", stream, "document")

    assert stored.sha256 == hashlib.sha256(content).hexdigest()
    assert stored.media_type == "text/plain"
    assert stored.kind == "document"
    assert stored.path == (
        storage.raw_file_dir / stored.document_id / stored.version_id / "notes.txt"
    )
    assert stored.path.read_bytes() == content
    assert stream.read_sizes
    assert all(size > 0 for size in stream.read_sizes)


def test_version_id_collision_never_removes_or_overwrites_existing_file(
    storage: DocumentStorage,
) -> None:
    stored = storage.store_upload(
        "notes.txt",
        io.BytesIO(b"accepted"),
        "document",
        document_id="document-1",
        version_id="version-1",
    )

    with pytest.raises(FileExistsError):
        storage.store_upload(
            "notes.txt",
            io.BytesIO(b"replacement"),
            "document",
            document_id="document-1",
            version_id="version-1",
        )

    assert stored.path.read_bytes() == b"accepted"


def test_rejects_files_over_50_mib_without_buffering(tmp_path: Path) -> None:
    maximum = 50 * 1024 * 1024
    storage = DocumentStorage(tmp_path / "raw", max_upload_bytes=maximum)
    stream = RepeatedBytes(b"x", maximum + 1)

    with pytest.raises(ValueError, match="50"):
        storage.store_upload("large.txt", stream, "document")

    assert stream.remaining == 0
    assert not list((tmp_path / "raw").rglob("*.txt"))


@pytest.mark.parametrize("prefix", [b"MZ\x90\x00", b"PK\x03\x04"])
def test_rejects_executable_or_zip_disguised_as_text(
    storage: DocumentStorage, prefix: bytes
) -> None:
    with pytest.raises(ValueError, match="executable|TXT|signature"):
        storage.store_upload(
            "disguised.txt", io.BytesIO(prefix + b"payload"), "document"
        )


def test_validates_pdf_signature(storage: DocumentStorage) -> None:
    stored = storage.store_upload(
        "manual.pdf", io.BytesIO(b"%PDF-1.7\nremainder"), "document"
    )
    assert stored.media_type == "application/pdf"

    for content in (b"MZ\x90\x00not a pdf", b"PK\x03\x04not a pdf"):
        with pytest.raises(ValueError, match="PDF|executable"):
            storage.store_upload("bad.pdf", io.BytesIO(content), "document")


@pytest.mark.parametrize("kind", ["docx", "xlsx"])
def test_validates_office_zip_signatures_and_contents(
    storage: DocumentStorage, kind: str
) -> None:
    stored = storage.store_upload(
        f"valid.{kind}", io.BytesIO(office_bytes(kind)), "document"
    )
    assert stored.path.suffix == f".{kind}"

    with pytest.raises(ValueError, match="Office"):
        storage.store_upload(
            f"bad.{kind}", io.BytesIO(b"PK\x03\x04not really zip"), "document"
        )

    other_kind = "xlsx" if kind == "docx" else "docx"
    with pytest.raises(ValueError, match="Office"):
        storage.store_upload(
            f"wrong.{kind}", io.BytesIO(office_bytes(other_kind)), "document"
        )


def test_txt_decoding_matches_supported_parsers(storage: DocumentStorage) -> None:
    for content in ("UTF-8 文档".encode(), "GB18030 文档".encode("gb18030")):
        stored = storage.store_upload("text.txt", io.BytesIO(content), "document")
        assert stored.path.read_bytes() == content

    with pytest.raises(ValueError, match="TXT"):
        storage.store_upload("bad.txt", io.BytesIO(b"\xff"), "document")


def test_qa_requires_exact_six_column_header(storage: DocumentStorage) -> None:
    stored = storage.store_upload("qa.xlsx", io.BytesIO(qa_bytes()), "qa")
    assert stored.kind == "qa"

    for headers in (
        QA_COLUMNS[:5],
        (*QA_COLUMNS, "extra"),
        ("question", "answer", "source_document_id", "evidence", "domain", "qa_id"),
    ):
        with pytest.raises(ValueError, match="exactly six columns"):
            storage.store_upload("bad-qa.xlsx", io.BytesIO(qa_bytes(headers)), "qa")
