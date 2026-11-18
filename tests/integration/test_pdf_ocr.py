import json
import traceback
from difflib import SequenceMatcher
from pathlib import Path

import fitz
import pytest

from app.ingestion.ocr import PaddleOCREngine
from app.ingestion.pdf import PDFParser
from app.mockgen.catalog import DocumentSpec, build_catalog

CORPUS = Path("mock_data/corpus")
ARTIFACTS = Path("var/test-artifacts/pdf-ocr")
SCANNED_SPECS = [item for item in build_catalog() if item.scanned]

pytestmark = pytest.mark.ocr


def _normalize(text: str) -> str:
    return "".join(text.split())


def _character_coverage(expected: str, actual: str) -> float:
    matcher = SequenceMatcher(None, expected, actual, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(expected)


def _preserve_failure_artifacts(
    spec: DocumentSpec, blocks, coverage: float, error: str = ""
) -> Path:
    destination = ARTIFACTS / spec.id
    destination.mkdir(parents=True, exist_ok=True)
    extracted = "\n\n".join(
        f"Page {block.location.page}\n{block.text}" for block in blocks
    )
    (destination / "extracted.txt").write_text(extracted, encoding="utf-8")
    if error:
        (destination / "error.txt").write_text(error, encoding="utf-8")
    (destination / "coverage.json").write_text(
        json.dumps(
            {
                "document_id": spec.id,
                "title": spec.title,
                "character_coverage": coverage,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    path = CORPUS / f"{spec.id}.pdf"
    with fitz.open(path) as document:
        for page_index, page in enumerate(document, start=1):
            page.get_pixmap(dpi=200).save(destination / f"page-{page_index}.png")
    return destination


@pytest.fixture(scope="module")
def paddle_engine() -> PaddleOCREngine:
    return PaddleOCREngine()


@pytest.mark.parametrize("spec", SCANNED_SPECS, ids=lambda spec: spec.id)
def test_real_cpu_ocr_recovers_scanned_title_and_evidence(
    spec: DocumentSpec, paddle_engine: PaddleOCREngine
):
    path = CORPUS / f"{spec.id}.pdf"

    try:
        blocks = PDFParser(ocr=paddle_engine).parse(path, spec.id)
    except Exception:
        _preserve_failure_artifacts(
            spec, [], coverage=0.0, error=traceback.format_exc()
        )
        raise

    actual = _normalize("\n".join(block.text for block in blocks))
    expected = _normalize("".join(fact.evidence for fact in spec.facts))
    coverage = _character_coverage(expected, actual)
    title_present = _normalize(spec.title) in actual
    page_numbers = [block.location.page for block in blocks]
    print(f"{spec.id} evidence character coverage: {coverage:.2%}")
    if not title_present or coverage < 0.90 or page_numbers != [1, 2, 3]:
        artifacts = _preserve_failure_artifacts(spec, blocks, coverage)
        pytest.fail(
            f"{spec.id}: title present={title_present}, "
            f"evidence character coverage={coverage:.2%}, "
            f"page numbers={page_numbers}; artifacts={artifacts}"
        )
