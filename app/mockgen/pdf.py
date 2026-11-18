"""Render selectable-text and image-only scanned PDF specifications."""

from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont

from app.mockgen.catalog import DocumentSpec

_PAGE_WIDTH = 595
_PAGE_HEIGHT = 842
_MARGIN = 42
_FONT_SIZE = 10.5
_LINE_HEIGHT = 24
_LINES_PER_PAGE = 27
_CJK_FONT_NAME = "china-s"
_IMAGE_SCALE = 2


def _document_lines(spec: DocumentSpec) -> list[str]:
    lines = [
        spec.title,
        f"Document ID: {spec.id}",
        f"Domain: {spec.domain}",
        f"Version: {spec.version}",
        f"Effective Date: {spec.effective_date}",
        "",
        "Traceable facts",
    ]
    for number, fact in enumerate(spec.facts, start=1):
        lines.extend(
            (
                f"Fact {number}",
                f"Question: {fact.question}",
                f"Paraphrase: {fact.paraphrase}",
                f"Answer: {fact.answer}",
                f"Evidence: {fact.evidence}",
            )
        )
    return lines


def paginate_spec(spec: DocumentSpec) -> list[list[str]]:
    """Group document lines into fixed-size, deterministic pages."""
    lines = _document_lines(spec)
    return [
        lines[start : start + _LINES_PER_PAGE]
        for start in range(0, len(lines), _LINES_PER_PAGE)
    ]


def render_lines_to_png(
    lines: list[str], page_number: int, font_path: Path, *, page_count: int = 1
) -> bytes:
    """Render one page into an RGB PNG, with no separate text layer."""
    image = Image.new(
        "RGB",
        (_PAGE_WIDTH * _IMAGE_SCALE, _PAGE_HEIGHT * _IMAGE_SCALE),
        "white",
    )
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), round(_FONT_SIZE * _IMAGE_SCALE))
    for index, line in enumerate(lines):
        draw.text(
            (
                _MARGIN * _IMAGE_SCALE,
                (_MARGIN + index * _LINE_HEIGHT) * _IMAGE_SCALE,
            ),
            line,
            font=font,
            fill="black",
        )
    draw.text(
        (_MARGIN * _IMAGE_SCALE, (_PAGE_HEIGHT - 30) * _IMAGE_SCALE),
        f"Page {page_number} / {page_count}",
        font=font,
        fill="black",
    )
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def insert_selectable_chinese(
    page: fitz.Page,
    lines: list[str],
    page_number: int,
    font_path: Path,
    *,
    page_count: int = 1,
) -> None:
    """Place Unicode text so evidence remains directly extractable."""
    for index, line in enumerate(lines):
        if not line:
            continue
        fontname = (
            _CJK_FONT_NAME
            if any(ord(character) > 127 for character in line)
            else "helv"
        )
        page.insert_text(
            (_MARGIN, _MARGIN + index * _LINE_HEIGHT),
            line,
            fontname=fontname,
            fontsize=_FONT_SIZE,
        )
    page.insert_text(
        (_MARGIN, _PAGE_HEIGHT - 30),
        f"Page {page_number} / {page_count}",
        fontname="helv",
        fontsize=_FONT_SIZE,
    )


def render_pdf(spec: DocumentSpec, output_dir: Path, font_path: Path) -> Path:
    """Render one text or scanned PDF without changing the public dispatcher."""
    if spec.fmt != "pdf":
        raise ValueError("PDF renderer requires a PDF document specification")
    if not font_path.is_file():
        raise FileNotFoundError(font_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{spec.id}.pdf"
    pages = paginate_spec(spec)
    with fitz.open() as document:
        for page_number, lines in enumerate(pages, start=1):
            page = document.new_page(width=_PAGE_WIDTH, height=_PAGE_HEIGHT)
            if spec.scanned:
                png = render_lines_to_png(
                    lines,
                    page_number,
                    font_path,
                    page_count=len(pages),
                )
                page.insert_image(page.rect, stream=png)
            else:
                insert_selectable_chinese(
                    page,
                    lines,
                    page_number,
                    font_path,
                    page_count=len(pages),
                )
        document.set_metadata(
            {"title": spec.title, "creationDate": "D:20261118000000Z"}
        )
        document.save(path, garbage=4, deflate=True)
    return path
