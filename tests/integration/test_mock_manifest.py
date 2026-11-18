import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import fitz
from docx import Document
from openpyxl import load_workbook

from app.mockgen.catalog import DocumentSpec, build_catalog
from app.mockgen.generate import CorpusManifest, generate_corpus

FONT_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "fonts"
    / "NotoSansCJKsc-Regular.otf"
)


def _normalized_text(path: Path, spec: DocumentSpec) -> str:
    if spec.fmt == "txt":
        text = path.read_text(encoding="utf-8")
    elif spec.fmt == "docx":
        document = Document(path)
        text = "\n".join(
            [paragraph.text for paragraph in document.paragraphs]
            + [
                cell.text
                for table in document.tables
                for row in table.rows
                for cell in row.cells
            ]
        )
    elif spec.fmt == "xlsx":
        workbook = load_workbook(path, data_only=False)
        try:
            text = "\n".join(
                str(cell.value)
                for worksheet in workbook.worksheets
                for row in worksheet.iter_rows()
                for cell in row
                if cell.value is not None
            )
        finally:
            workbook.close()
    else:
        with fitz.open(path) as document:
            text = "".join(page.get_text() for page in document)
    return "\n".join(text.split())


def _semantic_entries(manifest: CorpusManifest) -> list[dict[str, object]]:
    return [
        {key: value for key, value in asdict(document).items() if key != "sha256"}
        for document in manifest.documents
    ]


def test_generated_manifest_has_all_required_documents(tmp_path):
    manifest = generate_corpus(tmp_path, FONT_PATH)
    assert manifest.total_documents == 100
    assert manifest.counts == {"txt": 25, "docx": 25, "pdf": 25, "xlsx": 25}
    assert manifest.scanned_pdf_count == 5
    assert all((tmp_path / item.relative_path).is_file() for item in manifest.documents)
    assert len({item.sha256 for item in manifest.documents}) == 100
    assert len(list((tmp_path / "corpus").iterdir())) == 100
    assert len(list((tmp_path / "source").glob("*.json"))) == 100


def test_manifest_records_real_hashes_metadata_and_source_specs(tmp_path):
    manifest = generate_corpus(tmp_path, FONT_PATH)
    entries = {item.id: item for item in manifest.documents}
    serialized = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert serialized["total_documents"] == 100
    assert len(serialized["documents"]) == 100
    for spec in build_catalog():
        item = entries[spec.id]
        artifact = tmp_path / item.relative_path
        source = tmp_path / item.source_path
        assert item.relative_path == f"corpus/{spec.id}.{spec.fmt}"
        assert item.sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert item.format == spec.fmt
        assert item.scanned is spec.scanned
        assert item.domain == spec.domain
        assert item.version == spec.version
        assert item.effective_date == spec.effective_date
        assert item.source_path == f"source/{spec.id}.json"
        assert json.loads(source.read_text(encoding="utf-8")) == json.loads(
            json.dumps(asdict(spec), ensure_ascii=False)
        )


def test_regeneration_is_semantically_deterministic(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = generate_corpus(first_root, FONT_PATH)
    second = generate_corpus(second_root, FONT_PATH)

    assert first.total_documents == second.total_documents
    assert first.counts == second.counts
    assert first.scanned_pdf_count == second.scanned_pdf_count
    assert _semantic_entries(first) == _semantic_entries(second)
    for spec, first_item, second_item in zip(
        build_catalog(), first.documents, second.documents, strict=True
    ):
        assert _normalized_text(first_root / first_item.relative_path, spec) == (
            _normalized_text(second_root / second_item.relative_path, spec)
        )
