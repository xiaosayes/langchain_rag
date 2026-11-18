"""Parser contract for converting source documents into typed blocks."""

from pathlib import Path
from typing import Protocol

from app.domain.models import ParsedBlock


class DocumentParser(Protocol):
    """A parser supporting one or more case-insensitive filename suffixes."""

    suffixes: tuple[str, ...]

    def parse(self, path: Path, document_id: str) -> list[ParsedBlock]:
        """Parse a document into ordered, non-empty source blocks."""
        ...
