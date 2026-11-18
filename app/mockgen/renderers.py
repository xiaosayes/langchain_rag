"""Render deterministic non-PDF document specifications to local files."""

from pathlib import Path

from docx import Document
from openpyxl import Workbook

from app.mockgen.catalog import DocumentSpec


def _render_txt(spec: DocumentSpec, path: Path) -> None:
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
                "",
                f"Fact {number}",
                f"Question: {fact.question}",
                f"Paraphrase: {fact.paraphrase}",
                f"Answer: {fact.answer}",
                f"Evidence: {fact.evidence}",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_docx(spec: DocumentSpec, path: Path) -> None:
    document = Document()
    document.add_heading(spec.title, level=0)
    document.add_paragraph(f"Document ID: {spec.id}")
    document.add_paragraph(f"Domain: {spec.domain}")
    document.add_paragraph(f"Version: {spec.version}")
    document.add_paragraph(f"Effective Date: {spec.effective_date}")
    document.add_paragraph(
        "This synthetic enterprise document contains ten explicit, traceable facts."
    )
    document.add_heading("Traceable facts", level=1)
    table = document.add_table(rows=1, cols=5)
    for cell, heading in zip(
        table.rows[0].cells,
        ("Row", "Question", "Paraphrase", "Answer", "Evidence"),
        strict=True,
    ):
        cell.text = heading
    for number, fact in enumerate(spec.facts, start=1):
        cells = table.add_row().cells
        for cell, value in zip(
            cells,
            (
                str(number),
                fact.question,
                fact.paraphrase,
                fact.answer,
                fact.evidence,
            ),
            strict=True,
        ):
            cell.text = value
    document.save(path)


def _render_xlsx(spec: DocumentSpec, path: Path) -> None:
    workbook = Workbook()
    overview = workbook.active
    overview.title = "Overview"
    overview.append([spec.title])
    overview.append(["Document ID", spec.id])
    overview.append(["Domain", spec.domain])
    overview.append(["Version", spec.version])
    overview.append(["Effective Date", spec.effective_date])

    facts = workbook.create_sheet("facts")
    facts.append(["Row", "Question", "Paraphrase", "Answer", "Evidence"])
    for number, fact in enumerate(spec.facts, start=1):
        facts.append(
            [number, fact.question, fact.paraphrase, fact.answer, fact.evidence]
        )
    workbook.save(path)


def render_document(spec: DocumentSpec, output_dir: Path) -> Path:
    """Render TXT, DOCX, or XLSX without registering a PDF implementation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{spec.id}.{spec.fmt}"
    renderers = {"txt": _render_txt, "docx": _render_docx, "xlsx": _render_xlsx}
    if spec.fmt not in renderers:
        raise ValueError("PDF renderer is not registered")
    renderers[spec.fmt](spec, path)
    return path
