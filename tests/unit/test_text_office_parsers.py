from pathlib import Path

import pytest
from docx import Document
from openpyxl import Workbook

from app.domain.models import ParsedBlock, SourceLocation
from app.ingestion.registry import ParserRegistry, UnsupportedDocumentType
from app.ingestion.text_office import QAParser, TextParser, XlsxParser
from app.mockgen.catalog import DocumentSpec, build_catalog

CORPUS = Path("mock_data/corpus")
KNOWLEDGE_QA = Path("mock_data/qa/knowledge_qa.xlsx")
QA_COLUMNS = (
    "qa_id",
    "question",
    "answer",
    "source_document_id",
    "evidence",
    "domain",
)


def _assert_typed_blocks(blocks: list[ParsedBlock], document_id: str) -> None:
    assert blocks
    assert [block.sequence for block in blocks] == list(range(len(blocks)))
    assert all(block.document_id == document_id for block in blocks)
    assert all(block.text.strip() for block in blocks)
    assert all(isinstance(block.location, SourceLocation) for block in blocks)


@pytest.mark.parametrize(
    "spec",
    [
        next(item for item in build_catalog() if item.fmt == fmt)
        for fmt in ("txt", "docx", "xlsx")
    ],
    ids=lambda spec: spec.fmt,
)
def test_registry_parses_office_samples_into_typed_blocks(spec: DocumentSpec):
    path = CORPUS / f"{spec.id}.{spec.fmt}"

    blocks = ParserRegistry.default().parse(path, spec.id)

    _assert_typed_blocks(blocks, spec.id)
    combined = "\n".join(block.text for block in blocks)
    assert all(fact.evidence in combined for fact in spec.facts)


def test_txt_parser_strips_blank_paragraphs_without_merging_sections(tmp_path):
    path = tmp_path / "sections.txt"
    path.write_text(
        "Title\n\nFirst paragraph\ncontinued here\n\n   \nLast paragraph\n",
        encoding="utf-8",
    )

    blocks = TextParser().parse(path, "text-doc")

    _assert_typed_blocks(blocks, "text-doc")
    assert [block.text for block in blocks] == [
        "Title",
        "First paragraph\ncontinued here",
        "Last paragraph",
    ]


def test_docx_parser_preserves_heading_paths_and_table_rows():
    spec = next(item for item in build_catalog() if item.fmt == "docx")
    path = CORPUS / f"{spec.id}.docx"
    document = Document(path)
    assert document.tables

    blocks = ParserRegistry.default().parse(path, spec.id)

    _assert_typed_blocks(blocks, spec.id)
    assert any(block.location.heading_path == (spec.title,) for block in blocks)
    fact_blocks = [
        block
        for block in blocks
        if block.location.heading_path == (spec.title, "Traceable facts")
    ]
    assert fact_blocks
    assert any(
        "Row\tQuestion\tParaphrase\tAnswer\tEvidence" == block.text
        for block in fact_blocks
    )
    combined = "\n".join(block.text for block in fact_blocks)
    assert all(fact.evidence in combined for fact in spec.facts)


def test_xlsx_parser_preserves_sheet_header_and_consecutive_row_ranges(tmp_path):
    path = tmp_path / "rows.xlsx"
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "facts"
    worksheet.append(["Name", "Value"])
    worksheet.append(["alpha", 1])
    worksheet.append([None, None])
    worksheet.append(["beta", 2])
    worksheet.append(["formula", "=1+1"])
    workbook.save(path)
    workbook.close()

    blocks = XlsxParser().parse(path, "sheet-doc")

    _assert_typed_blocks(blocks, "sheet-doc")
    assert [
        (block.location.sheet, block.location.row_start, block.location.row_end)
        for block in blocks
    ] == [
        ("facts", 1, 2),
        ("facts", 4, 5),
    ]
    assert "Row 1: Name | Value" in blocks[0].text
    assert "Row 1: Name | Value" in blocks[1].text
    assert "Row 2: alpha | 1" in blocks[0].text
    assert "Row 4: beta | 2" in blocks[1].text
    assert "=1+1" not in "\n".join(block.text for block in blocks)


def test_qa_parser_emits_one_atomic_lineage_block_per_row():
    blocks = QAParser().parse(KNOWLEDGE_QA)

    _assert_typed_blocks(blocks, "knowledge_qa")
    assert len(blocks) == 1000
    assert [block.location.row_start for block in blocks] == list(range(2, 1002))
    assert all(block.location.sheet == "knowledge_qa" for block in blocks)
    assert all(block.location.row_start == block.location.row_end for block in blocks)
    assert len({block.metadata["qa_id"] for block in blocks}) == 1000

    catalog = {spec.id: spec for spec in build_catalog()}
    for block in blocks:
        assert block.metadata["record_type"] == "qa"
        source_id = block.metadata["source_document_id"]
        assert source_id in catalog
        fact_index = int(block.metadata["qa_id"].rsplit("-Q", 1)[1]) - 1
        fact = catalog[source_id].facts[fact_index]
        assert f"Question: {fact.question}" in block.text
        assert f"Answer: {fact.answer}" in block.text
        assert f"Evidence: {fact.evidence}" in block.text


def test_qa_parser_requires_exact_columns(tmp_path):
    path = tmp_path / "invalid_qa.xlsx"
    workbook = Workbook()
    workbook.active.append(QA_COLUMNS[:-1])
    workbook.save(path)
    workbook.close()

    with pytest.raises(ValueError, match="exactly six columns"):
        QAParser().parse(path)


def test_qa_parser_rejects_duplicate_qa_ids(tmp_path):
    path = tmp_path / "duplicate_qa.xlsx"
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(QA_COLUMNS)
    row = ("QA-1", "Question", "Answer", "DOC-000", "Evidence", "Domain")
    worksheet.append(row)
    worksheet.append(row)
    workbook.save(path)
    workbook.close()

    with pytest.raises(ValueError, match="Duplicate qa_id: QA-1"):
        QAParser().parse(path)


@pytest.mark.parametrize(
    "path",
    [CORPUS / "DOC-050.pdf", Path("mock_data/eval/evaluation.jsonl")],
)
def test_default_registry_excludes_pdf_and_evaluation(path):
    with pytest.raises(UnsupportedDocumentType):
        ParserRegistry.default().parse(path, "unsupported")


@pytest.mark.parametrize(
    "spec",
    [item for item in build_catalog() if item.fmt != "pdf"],
    ids=lambda spec: spec.id,
)
def test_every_generated_office_document_parses(spec: DocumentSpec):
    path = CORPUS / f"{spec.id}.{spec.fmt}"

    blocks = ParserRegistry.default().parse(path, spec.id)

    _assert_typed_blocks(blocks, spec.id)
    combined = "\n".join(block.text for block in blocks)
    assert all(fact.evidence in combined for fact in spec.facts)
