from pathlib import Path

import fitz
import pytest

from app.ingestion.ocr import OCRProcessingError
from app.ingestion.pdf import PDFParser, has_usable_text_layer
from app.ingestion.registry import ParserRegistry
from app.mockgen.catalog import build_catalog

CORPUS = Path("mock_data/corpus")


class FakeOCR:
    def __init__(self):
        self.calls = 0

    def recognize(self, image_png: bytes) -> str:
        self.calls += 1
        assert image_png.startswith(b"\x89PNG")
        return "扫描件中的报销标准为每天500元"


def test_text_pdf_skips_ocr_and_keeps_page_numbers():
    spec = next(
        item for item in build_catalog() if item.fmt == "pdf" and not item.scanned
    )
    ocr = FakeOCR()

    blocks = PDFParser(ocr=ocr).parse(CORPUS / f"{spec.id}.pdf", spec.id)

    assert ocr.calls == 0
    assert [block.location.page for block in blocks] == [1, 2, 3]
    assert [block.sequence for block in blocks] == [0, 1, 2]
    assert spec.title in blocks[0].text
    assert all(block.document_id == spec.id for block in blocks)
    assert all(block.metadata == {"parser": "pdf"} for block in blocks)


def test_scanned_pdf_uses_ocr_and_keeps_page_numbers():
    spec = next(item for item in build_catalog() if item.scanned)
    ocr = FakeOCR()
    path = CORPUS / f"{spec.id}.pdf"
    with fitz.open(path) as document:
        page_count = len(document)

    blocks = PDFParser(ocr=ocr).parse(path, spec.id)

    assert ocr.calls == page_count
    assert "每天500元" in "\n".join(block.text for block in blocks)
    assert [block.location.page for block in blocks] == list(range(1, page_count + 1))


def test_page_with_at_least_20_meaningful_characters_skips_ocr(tmp_path):
    path = tmp_path / "mixed.pdf"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "This page has at least twenty characters.")
        page = document.new_page()
        page.insert_text((72, 72), "short")
        document.save(path)
    ocr = FakeOCR()

    blocks = PDFParser(ocr=ocr).parse(path, "mixed")

    assert ocr.calls == 1
    assert [block.location.page for block in blocks] == [1, 2]
    assert "This page has at least twenty characters." in blocks[0].text
    assert "每天500元" in blocks[1].text


@pytest.mark.parametrize(
    "text",
    [
        "..............................",
        " \n\t" * 20,
        "abcdefghijklmnopqrstuvwxyz\ufffd\ufffd\ufffd\ufffd",
        "\x00abcdefghijklmnopqrstuvwxyz",
    ],
)
def test_corrupted_or_non_meaningful_text_layer_is_not_usable(text):
    assert not has_usable_text_layer(text)


def test_ocr_is_selected_for_corrupted_long_text_layer(tmp_path, monkeypatch):
    path = tmp_path / "corrupt.pdf"
    with fitz.open() as document:
        document.new_page()
        document.save(path)
    monkeypatch.setattr(
        fitz.Page,
        "get_text",
        lambda self, *args, **kwargs: "abcdefghijklmnopqrstuvwxyz\ufffd\ufffd\ufffd\ufffd",
    )
    ocr = FakeOCR()

    blocks = PDFParser(ocr=ocr).parse(path, "corrupt")

    assert ocr.calls == 1
    assert "每天500元" in blocks[0].text


def test_default_engine_is_created_only_when_ocr_is_required(tmp_path, monkeypatch):
    text_spec = next(
        item for item in build_catalog() if item.fmt == "pdf" and not item.scanned
    )
    ocr = FakeOCR()
    creations = 0

    def create_engine():
        nonlocal creations
        creations += 1
        return ocr

    monkeypatch.setattr("app.ingestion.pdf.PaddleOCREngine", create_engine)
    parser = PDFParser()
    parser.parse(CORPUS / f"{text_spec.id}.pdf", text_spec.id)
    assert creations == 0

    path = tmp_path / "blank.pdf"
    with fitz.open() as document:
        document.new_page()
        document.save(path)
    blocks = parser.parse(path, "blank")
    assert [block.location.page for block in blocks] == [1]
    assert "每天500元" in blocks[0].text
    assert creations == 1
    assert ocr.calls == 1


def test_ocr_error_identifies_document_and_page(tmp_path):
    class BrokenOCR:
        def recognize(self, image_png: bytes) -> str:
            raise RuntimeError("recognizer failed")

    path = tmp_path / "broken.pdf"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "This page has enough text to skip OCR.")
        document.new_page()
        document.save(path)

    with pytest.raises(OCRProcessingError, match=r"broken\.pdf: page 2") as error:
        PDFParser(ocr=BrokenOCR()).parse(path, "broken")

    assert isinstance(error.value.__cause__, RuntimeError)


def test_default_registry_routes_pdf_without_replacing_existing_parsers():
    spec = next(
        item for item in build_catalog() if item.fmt == "pdf" and not item.scanned
    )
    registry = ParserRegistry.default()

    blocks = registry.parse(CORPUS / f"{spec.id}.pdf", spec.id)

    assert spec.title in blocks[0].text
    assert registry.parse(CORPUS / "DOC-000.txt", "DOC-000")
