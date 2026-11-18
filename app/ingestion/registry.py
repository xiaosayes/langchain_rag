"""Filename-suffix routing for ordinary document parsers."""

from collections.abc import Iterable
from pathlib import Path

from app.domain.models import ParsedBlock
from app.ingestion.base import DocumentParser
from app.ingestion.pdf import PDFParser
from app.ingestion.text_office import DocxParser, TextParser, XlsxParser


class UnsupportedDocumentType(ValueError):
    """Raised when no parser is registered for a document suffix."""


class ParserRegistry:
    def __init__(self, parsers: Iterable[DocumentParser]):
        self._parsers = {
            suffix: parser for parser in parsers for suffix in parser.suffixes
        }

    @classmethod
    def default(cls) -> "ParserRegistry":
        return cls([TextParser(), DocxParser(), XlsxParser(), PDFParser()])

    def parse(self, path: Path, document_id: str) -> list[ParsedBlock]:
        parser = self._parsers.get(path.suffix.lower())
        if parser is None:
            raise UnsupportedDocumentType(path.suffix)
        return parser.parse(path, document_id)
