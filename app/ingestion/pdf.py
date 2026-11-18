"""PDF parser with selective OCR for pages lacking a usable text layer."""

from pathlib import Path

import fitz

from app.domain.models import ParsedBlock, SourceLocation
from app.ingestion.ocr import OCREngine, OCRProcessingError, PaddleOCREngine

_MINIMUM_TEXT_CHARACTERS = 20
_OCR_DPI = 200


def has_usable_text_layer(text: str) -> bool:
    """Require meaningful characters and reject common corrupted extraction layers."""
    meaningful = sum(character.isalnum() for character in text)
    corrupted = sum(
        character == "\ufffd"
        or (not character.isspace() and ord(character) < 32)
        for character in text
    )
    return (
        meaningful >= _MINIMUM_TEXT_CHARACTERS
        and "\x00" not in text
        and corrupted / max(len(text), 1) <= 0.05
    )


class PDFParser:
    suffixes: tuple[str, ...] = (".pdf",)

    def __init__(self, ocr: OCREngine | None = None) -> None:
        self._ocr = ocr

    @property
    def ocr_engine(self) -> OCREngine:
        if self._ocr is None:
            self._ocr = PaddleOCREngine()
        return self._ocr

    def parse(self, path: Path, document_id: str) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        with fitz.open(path) as document:
            for page_index, page in enumerate(document):
                extracted = page.get_text("text").strip()
                if has_usable_text_layer(extracted):
                    text = extracted
                else:
                    pixmap = page.get_pixmap(dpi=_OCR_DPI)
                    try:
                        text = self.ocr_engine.recognize(pixmap.tobytes("png")).strip()
                    except Exception as exc:
                        raise OCRProcessingError(
                            f"{path.name}: page {page_index + 1}"
                        ) from exc
                if text:
                    blocks.append(
                        ParsedBlock(
                            document_id=document_id,
                            sequence=len(blocks),
                            text=text,
                            location=SourceLocation(page=page_index + 1),
                            metadata={"parser": "pdf"},
                        )
                    )
        return blocks
