"""Parsers for plain text, Office documents, and knowledge Q&A workbooks."""

import re
from pathlib import Path
from typing import Any

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from openpyxl import load_workbook  # type: ignore[import-untyped]

from app.domain.models import ParsedBlock, SourceLocation

QA_COLUMNS = (
    "qa_id",
    "question",
    "answer",
    "source_document_id",
    "evidence",
    "domain",
)


class TextParser:
    suffixes: tuple[str, ...] = (".txt",)

    def parse(self, path: Path, document_id: str) -> list[ParsedBlock]:
        raw = path.read_bytes()
        source = ""
        last_error: UnicodeDecodeError | None = None
        # Deliberately bounded: UTF-8 (with optional BOM), then Chinese GB18030.
        for encoding in ("utf-8-sig", "gb18030"):
            try:
                source = raw.decode(encoding)
                break
            except UnicodeDecodeError as error:
                last_error = error
        else:
            raise UnicodeError(
                f"{path.name}: TXT encoding is neither UTF-8 nor GB18030"
            ) from last_error
        source = source.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = []
        for paragraph in re.split(r"\n\s*\n", source):
            text = "\n".join(
                line.strip() for line in paragraph.splitlines() if line.strip()
            )
            if text:
                paragraphs.append(text)

        return [
            ParsedBlock(
                document_id=document_id,
                sequence=sequence,
                text=text,
                location=SourceLocation(),
                metadata={"parser": "txt"},
            )
            for sequence, text in enumerate(paragraphs)
        ]


def _heading_level(paragraph: Paragraph) -> int | None:
    style_name = paragraph.style.name if paragraph.style is not None else ""
    if style_name == "Title":
        return 0
    if style_name and style_name.startswith("Heading "):
        try:
            return int(style_name.removeprefix("Heading "))
        except ValueError:
            return None
    return None


def _docx_paragraph_text(paragraph: Paragraph) -> str:
    return paragraph.text.strip()


def _docx_table_row_text(table: Table) -> list[str]:
    rows = []
    for row in table.rows:
        text = "\t".join(cell.text.strip() for cell in row.cells).strip()
        if text:
            rows.append(text)
    return rows


class DocxParser:
    suffixes: tuple[str, ...] = (".docx",)

    def parse(self, path: Path, document_id: str) -> list[ParsedBlock]:
        document = Document(str(path))
        blocks: list[ParsedBlock] = []
        heading_path: tuple[str, ...] = ()

        for item in document.iter_inner_content():
            if isinstance(item, Paragraph):
                text = _docx_paragraph_text(item)
                if not text:
                    continue
                level = _heading_level(item)
                if level is not None:
                    if level == 0:
                        heading_path = (text,)
                    else:
                        heading_path = (*heading_path[:level], text)
                texts = [text]
            elif isinstance(item, Table):
                texts = _docx_table_row_text(item)
            else:
                continue

            for text in texts:
                blocks.append(
                    ParsedBlock(
                        document_id=document_id,
                        sequence=len(blocks),
                        text=text,
                        location=SourceLocation(heading_path=heading_path),
                        metadata={"parser": "docx"},
                    )
                )

        return blocks


def _has_value(values: tuple[Any, ...]) -> bool:
    return any(value is not None and bool(str(value).strip()) for value in values)


def _xlsx_row_text(row_number: int, values: tuple[Any, ...]) -> str:
    cells = ["" if value is None else str(value).strip() for value in values]
    while cells and not cells[-1]:
        cells.pop()
    return f"Row {row_number}: " + " | ".join(cells)


class XlsxParser:
    suffixes: tuple[str, ...] = (".xlsx",)

    def parse(self, path: Path, document_id: str) -> list[ParsedBlock]:
        workbook = load_workbook(path, read_only=True, data_only=True)
        blocks: list[ParsedBlock] = []
        try:
            for worksheet in workbook.worksheets:
                groups: list[list[tuple[int, tuple[Any, ...]]]] = []
                current_group: list[tuple[int, tuple[Any, ...]]] = []
                for row_number, values in enumerate(
                    worksheet.iter_rows(values_only=True), start=1
                ):
                    if _has_value(values):
                        current_group.append((row_number, values))
                    elif current_group:
                        groups.append(current_group)
                        current_group = []
                if current_group:
                    groups.append(current_group)
                if not groups:
                    continue

                header = groups[0][0]
                for group in groups:
                    rows = group if group[0][0] == header[0] else [header, *group]
                    text = "\n".join(
                        [f"Sheet: {worksheet.title}"]
                        + [_xlsx_row_text(number, values) for number, values in rows]
                    )
                    blocks.append(
                        ParsedBlock(
                            document_id=document_id,
                            sequence=len(blocks),
                            text=text,
                            location=SourceLocation(
                                sheet=worksheet.title,
                                row_start=group[0][0],
                                row_end=group[-1][0],
                            ),
                            metadata={"parser": "xlsx"},
                        )
                    )
        finally:
            workbook.close()
        return blocks


class QAParser:
    """Parse the dedicated knowledge workbook without registry auto-routing."""

    def parse(self, path: Path, document_id: str = "knowledge_qa") -> list[ParsedBlock]:
        workbook = load_workbook(path, read_only=True, data_only=True)
        blocks: list[ParsedBlock] = []
        seen_qa_ids: set[str] = set()
        try:
            worksheet = workbook.active
            rows = worksheet.iter_rows(values_only=True)
            header = next(rows, None)
            if header != QA_COLUMNS:
                raise ValueError(
                    f"{path.name}: Q&A workbook must contain exactly six columns: "
                    + ", ".join(QA_COLUMNS)
                )

            for row_number, values in enumerate(rows, start=2):
                if not _has_value(values):
                    continue
                cleaned = [
                    value.strip() if isinstance(value, str) else "" for value in values
                ]
                for column, value in zip(QA_COLUMNS, cleaned, strict=True):
                    if not value:
                        raise ValueError(
                            f"{path.name}: row {row_number} has an invalid {column}"
                        )
                qa_id, question, answer, source_id, evidence, domain = cleaned
                if qa_id in seen_qa_ids:
                    raise ValueError(f"Duplicate qa_id: {qa_id}")
                seen_qa_ids.add(qa_id)

                text = "\n".join(
                    (
                        f"Question: {question}",
                        f"Answer: {answer}",
                        f"Evidence: {evidence}",
                        f"Source Document ID: {source_id}",
                        f"Domain: {domain}",
                    )
                )
                blocks.append(
                    ParsedBlock(
                        document_id=document_id,
                        sequence=len(blocks),
                        text=text,
                        location=SourceLocation(
                            sheet=worksheet.title,
                            row_start=row_number,
                            row_end=row_number,
                        ),
                        metadata={
                            "record_type": "qa",
                            "qa_id": qa_id,
                            "source_document_id": source_id,
                            "domain": domain,
                        },
                    )
                )
        finally:
            workbook.close()
        return blocks
