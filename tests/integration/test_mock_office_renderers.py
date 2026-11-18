from pathlib import Path

import pytest
from docx import Document
from openpyxl import load_workbook

from app.mockgen.catalog import DocumentSpec, build_catalog
from app.mockgen.renderers import render_document


def _docx_text(path: Path) -> str:
    document = Document(path)
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    return (
        text
        + "\n"
        + "\n".join(
            cell.text
            for table in document.tables
            for row in table.rows
            for cell in row.cells
        )
    )


def _xlsx_text(path: Path) -> str:
    workbook = load_workbook(path, data_only=False)
    try:
        return "\n".join(
            str(cell.value)
            for worksheet in workbook.worksheets
            for row in worksheet.iter_rows()
            for cell in row
            if cell.value is not None
        )
    finally:
        workbook.close()


def test_txt_renderer_preserves_metadata_and_all_evidence_as_utf8(tmp_path):
    spec = next(item for item in build_catalog() if item.fmt == "txt")

    path = render_document(spec, tmp_path)
    text = path.read_text(encoding="utf-8")

    assert path == tmp_path / f"{spec.id}.txt"
    assert spec.title in text
    assert spec.version in text
    assert spec.effective_date in text
    assert all(fact.evidence in text for fact in spec.facts)


def test_docx_renderer_preserves_all_evidence(tmp_path):
    spec = next(item for item in build_catalog() if item.fmt == "docx")

    path = render_document(spec, tmp_path)
    document = Document(path)
    text = _docx_text(path)

    assert path == tmp_path / f"{spec.id}.docx"
    assert spec.title in text
    assert spec.version in text
    assert spec.effective_date in text
    assert document.tables
    assert all(fact.evidence in text for fact in spec.facts)


def test_xlsx_renderer_preserves_metadata_evidence_and_row_numbers(tmp_path):
    spec = next(item for item in build_catalog() if item.fmt == "xlsx")

    path = render_document(spec, tmp_path)
    workbook = load_workbook(path, data_only=False)
    try:
        assert workbook.sheetnames == ["Overview", "facts"]
        overview = workbook["Overview"]
        facts = workbook["facts"]
        text = _xlsx_text(path)

        assert path == tmp_path / f"{spec.id}.xlsx"
        assert spec.title in text
        assert spec.version in text
        assert spec.effective_date in text
        assert [cell.value for cell in facts[1]] == [
            "Row",
            "Question",
            "Paraphrase",
            "Answer",
            "Evidence",
        ]
        assert [facts.cell(row=row, column=1).value for row in range(2, 12)] == list(
            range(1, 11)
        )
        assert all(fact.evidence in text for fact in spec.facts)
        assert all(
            not (isinstance(cell.value, str) and cell.value.startswith("="))
            for worksheet in workbook.worksheets
            for row in worksheet.iter_rows()
            for cell in row
        )
        assert overview["A1"].value == spec.title
    finally:
        workbook.close()


@pytest.mark.parametrize(
    "spec",
    [spec for spec in build_catalog() if spec.fmt != "pdf"],
    ids=lambda spec: spec.id,
)
def test_every_non_pdf_spec_renders_literal_evidence(spec: DocumentSpec, tmp_path):
    path = render_document(spec, tmp_path / "nested")

    if spec.fmt == "txt":
        text = path.read_text(encoding="utf-8")
    elif spec.fmt == "docx":
        text = _docx_text(path)
    else:
        text = _xlsx_text(path)

    assert path.is_file()
    assert spec.title in text
    assert all(fact.evidence in text for fact in spec.facts)


def test_pdf_renderer_is_explicitly_deferred(tmp_path):
    spec = next(item for item in build_catalog() if item.fmt == "pdf")

    with pytest.raises(ValueError, match="^PDF renderer is not registered$"):
        render_document(spec, tmp_path)

    assert not (tmp_path / f"{spec.id}.pdf").exists()
