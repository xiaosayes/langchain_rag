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
_FONT_NAME = "NotoSansCJKsc"
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
        page.insert_text(
            (_MARGIN, _MARGIN + index * _LINE_HEIGHT),
            line,
            fontname=_FONT_NAME,
            fontfile=str(font_path),
            fontsize=_FONT_SIZE,
        )
    page.insert_text(
        (_MARGIN, _PAGE_HEIGHT - 30),
        f"Page {page_number} / {page_count}",
        fontname=_FONT_NAME,
        fontfile=str(font_path),
        fontsize=_FONT_SIZE,
    )


def _update_to_unicode(
    document: fitz.Document, pages: list[list[str]], font_path: Path
) -> None:
    """Map used glyph IDs back to their original, exact Unicode codepoints."""
    font = fitz.Font(fontfile=str(font_path))
    page_numbers = [
        f"Page {number} / {len(pages)}" for number in range(1, len(pages) + 1)
    ]
    text = "\n".join(
        [line for page_lines in pages for line in page_lines] + page_numbers
    )
    mappings: dict[int, str] = {}
    for character in text:
        if character == "\n":
            continue
        glyph_id = font.has_glyph(ord(character))
        if glyph_id == 0:
            raise ValueError(f"Pinned font has no glyph for U+{ord(character):04X}")
        previous = mappings.get(glyph_id)
        if previous is not None and previous != character:
            raise ValueError("Pinned font maps two source characters to one glyph")
        mappings[glyph_id] = character

    items = sorted(mappings.items())
    cmap = [
        "/CIDInit /ProcSet findresource begin\n",
        "12 dict begin\n",
        "begincmap\n",
        "/CIDSystemInfo <</Registry(Adobe)/Ordering(UCS)/Supplement 0>> def\n",
        "/CMapName /Adobe-Identity-UCS def\n",
        "/CMapType 2 def\n",
        "1 begincodespacerange\n",
        "<0000> <FFFF>\n",
        "endcodespacerange\n",
    ]
    for start in range(0, len(items), 100):
        chunk = items[start : start + 100]
        cmap.append(f"{len(chunk)} beginbfchar\n")
        cmap.extend(
            f"<{glyph_id:04X}> <{character.encode('utf-16-be').hex().upper()}>\n"
            for glyph_id, character in chunk
        )
        cmap.append("endbfchar\n")
    cmap.extend(
        (
            "endcmap\n",
            "CMapName currentdict /CMap defineresource pop\n",
            "end\n",
            "end\n",
        )
    )
    stream = "".join(cmap).encode("ascii")

    font_xrefs = {
        font_info[0]
        for page in document
        for font_info in page.get_fonts(full=True)
        if font_info[4] == _FONT_NAME
    }
    for font_xref in font_xrefs:
        kind, value = document.xref_get_key(font_xref, "ToUnicode")
        if kind != "xref":
            raise RuntimeError("Embedded Noto font has no indirect ToUnicode map")
        document.update_stream(int(value.split()[0]), stream)


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
        if not spec.scanned:
            document.subset_fonts()
            _update_to_unicode(document, pages, font_path)
        document.save(path, garbage=4, deflate=True)
    return path
