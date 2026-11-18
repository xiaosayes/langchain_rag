import os
import shutil
import subprocess
from io import BytesIO
from pathlib import Path

import fitz
import pytest
from PIL import Image

from app.mockgen.catalog import build_catalog
from app.mockgen.pdf import render_pdf

FONT_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "fonts"
    / "NotoSansCJKsc-Regular.otf"
)


def test_text_mock_pdf_preserves_metadata_and_extractable_evidence(tmp_path):
    spec = next(
        item for item in build_catalog() if item.fmt == "pdf" and not item.scanned
    )

    path = render_pdf(spec, tmp_path, FONT_PATH)

    with fitz.open(path) as document:
        page_text = [page.get_text() for page in document]
        text = "".join(page_text)
        assert path == tmp_path / f"{spec.id}.pdf"
        assert len(document) > 1
        assert spec.title in text
        assert spec.version in text
        assert spec.effective_date in text
        assert all(fact.evidence in text for fact in spec.facts)
        assert all(
            f"Page {number} / {len(document)}" in page
            for number, page in enumerate(page_text, start=1)
        )


def test_text_mock_pdf_is_readable_by_independent_pdftotext(tmp_path):
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        pytest.skip("pdftotext is unavailable")
    spec = next(
        item for item in build_catalog() if item.fmt == "pdf" and not item.scanned
    )
    path = render_pdf(spec, tmp_path, FONT_PATH)
    for filename in (".xpdfrc", "xpdfrc"):
        (tmp_path / filename).write_text("textEncoding UTF-8\n", encoding="ascii")
    environment = os.environ.copy()
    environment.update({"HOME": str(tmp_path), "USERPROFILE": str(tmp_path)})

    result = subprocess.run(
        [pdftotext, str(path), "-"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "Syntax Error" not in result.stderr
    assert all(fact.evidence in result.stdout for fact in spec.facts)


def test_scanned_mock_pdf_has_image_but_no_text_layer(tmp_path):
    spec = next(item for item in build_catalog() if item.scanned)
    path = render_pdf(spec, tmp_path, FONT_PATH)
    with fitz.open(path) as document:
        assert len("".join(page.get_text() for page in document).strip()) < 20
        assert any(page.get_images() for page in document)
        images = [image for page in document for image in page.get_images(full=True)]
        assert len(images) == len(document)
        for image in images:
            extracted = document.extract_image(image[0])
            with Image.open(BytesIO(extracted["image"])) as png:
                assert png.mode == "RGB"
