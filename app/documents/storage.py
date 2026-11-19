"""Streamed, validated uploads stored under immutable version identifiers."""

from __future__ import annotations

import codecs
import shutil
import unicodedata
import zipfile
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from typing import BinaryIO
from uuid import uuid4

from openpyxl import load_workbook  # type: ignore[import-untyped]

from app.documents.schemas import DocumentKind, StoredUpload
from app.ingestion.text_office import QA_COLUMNS

_CHUNK_SIZE = 1024 * 1024
_DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MEDIA_TYPES = {
    ".txt": "text/plain",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pdf": "application/pdf",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


class DocumentStorage:
    """Validate and persist uploads without buffering their complete contents."""

    def __init__(
        self,
        raw_file_dir: str | Path,
        *,
        max_upload_bytes: int = _DEFAULT_MAX_UPLOAD_BYTES,
    ) -> None:
        if max_upload_bytes <= 0:
            raise ValueError("Maximum upload size must be positive")
        self.raw_file_dir = Path(raw_file_dir)
        self.max_upload_bytes = max_upload_bytes

    def store_upload(
        self,
        filename: str,
        content: BinaryIO,
        kind: DocumentKind,
        *,
        document_id: str | None = None,
        version_id: str | None = None,
    ) -> StoredUpload:
        """Stream one candidate to its final immutable path, then validate it."""
        original_name = self._validate_filename(filename, kind)
        document_id = self._validate_identifier(
            document_id or str(uuid4()), "document_id"
        )
        version_id = self._validate_identifier(version_id or str(uuid4()), "version_id")
        version_dir = self.raw_file_dir / document_id / version_id
        path = version_dir / original_name
        digest = sha256()
        size = 0

        created_version_dir = False
        try:
            version_dir.mkdir(parents=True, exist_ok=False)
            created_version_dir = True
            with path.open("xb") as destination:
                while chunk := content.read(_CHUNK_SIZE):
                    size += len(chunk)
                    if size > self.max_upload_bytes:
                        self._raise_too_large()
                    digest.update(chunk)
                    destination.write(chunk)
            self._validate_content(path, original_name, kind)
        except Exception:
            if created_version_dir:
                shutil.rmtree(version_dir, ignore_errors=True)
                try:
                    (self.raw_file_dir / document_id).rmdir()
                except OSError:
                    pass
            raise

        return StoredUpload(
            document_id=document_id,
            version_id=version_id,
            path=path,
            sha256=digest.hexdigest(),
            media_type=MEDIA_TYPES[path.suffix.lower()],
            kind=kind,
        )

    @staticmethod
    def _validate_identifier(value: str, label: str) -> str:
        if (
            not value
            or value in {".", ".."}
            or any(character in value for character in ("/", "\\"))
            or any(unicodedata.category(character) == "Cc" for character in value)
            or PureWindowsPath(value).drive
        ):
            raise ValueError(f"{label} must be one safe path component")
        return value

    def _raise_too_large(self) -> None:
        maximum_mb = self.max_upload_bytes / (1024 * 1024)
        raise ValueError(f"Upload exceeds the {maximum_mb:g} MB limit")

    @staticmethod
    def _validate_filename(filename: str, kind: DocumentKind) -> str:
        if not filename or any(character in filename for character in ("/", "\\")):
            raise ValueError("Filename must be one path component")
        if any(unicodedata.category(character) == "Cc" for character in filename):
            raise ValueError("Filename contains control characters")
        windows_path = PureWindowsPath(filename)
        if (
            Path(filename).is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
        ):
            raise ValueError("Absolute and drive-qualified paths are not allowed")
        if filename in {".", ".."} or filename.endswith((" ", ".")):
            raise ValueError("Filename is reserved")
        if filename.split(".", 1)[0].upper() in _RESERVED_NAMES:
            raise ValueError("Filename is reserved")
        if filename.casefold() == "evaluation.jsonl":
            raise ValueError("Evaluation data cannot be ingested as a document")

        original_name = Path(filename).name
        suffix = Path(original_name).suffix.lower()
        if suffix not in MEDIA_TYPES:
            raise ValueError("Only TXT, DOCX, PDF, and XLSX documents are supported")
        if kind not in {"document", "qa"}:
            raise ValueError("Document kind must be document or qa")
        if kind == "qa" and suffix != ".xlsx":
            raise ValueError("Q&A uploads must be XLSX workbooks")
        return original_name

    def _validate_content(
        self, path: Path, original_name: str, kind: DocumentKind
    ) -> None:
        suffix = path.suffix.lower()
        with path.open("rb") as source:
            signature = source.read(8)
        if signature.startswith(b"MZ"):
            raise ValueError(f"{original_name}: executable content is not allowed")

        if suffix == ".txt":
            if signature.startswith(b"PK"):
                raise ValueError(f"{original_name}: TXT file has a ZIP signature")
            self._validate_txt(path, original_name)
        elif suffix == ".pdf":
            if not signature.startswith(b"%PDF-"):
                raise ValueError(f"{original_name}: invalid PDF signature")
        else:
            self._validate_office(path, original_name, suffix)
            if kind == "qa":
                self._validate_qa(path, original_name)

    @staticmethod
    def _validate_txt(path: Path, original_name: str) -> None:
        last_error: UnicodeDecodeError | None = None
        for encoding in ("utf-8-sig", "gb18030"):
            decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
            try:
                with path.open("rb") as source:
                    while chunk := source.read(_CHUNK_SIZE):
                        decoder.decode(chunk)
                    decoder.decode(b"", final=True)
                return
            except UnicodeDecodeError as error:
                last_error = error
        raise ValueError(
            f"{original_name}: TXT encoding is neither UTF-8 nor GB18030"
        ) from last_error

    @staticmethod
    def _validate_office(path: Path, original_name: str, suffix: str) -> None:
        required_member = (
            "word/document.xml" if suffix == ".docx" else "xl/workbook.xml"
        )
        try:
            with path.open("rb") as source:
                if not source.read(4).startswith(b"PK"):
                    raise ValueError(f"{original_name}: invalid Office ZIP signature")
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                if "[Content_Types].xml" not in names or required_member not in names:
                    raise ValueError(
                        f"{original_name}: invalid Office {suffix} package"
                    )
        except (zipfile.BadZipFile, OSError) as error:
            raise ValueError(f"{original_name}: invalid Office ZIP package") from error

    @staticmethod
    def _validate_qa(path: Path, original_name: str) -> None:
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            header = next(workbook.active.iter_rows(values_only=True), None)
        finally:
            workbook.close()
        if header != QA_COLUMNS:
            raise ValueError(
                f"{original_name}: Q&A workbook must contain exactly six columns: "
                + ", ".join(QA_COLUMNS)
            )
