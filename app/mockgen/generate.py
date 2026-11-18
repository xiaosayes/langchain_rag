"""Generate the complete synthetic corpus and its integrity manifest."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from app.mockgen.catalog import DocumentSpec, build_catalog
from app.mockgen.pdf import render_pdf
from app.mockgen.qa import generate_qa
from app.mockgen.renderers import render_document


@dataclass(frozen=True)
class ManifestDocument:
    """Integrity and catalog metadata for one generated ordinary document."""

    id: str
    relative_path: str
    sha256: str
    format: str
    scanned: bool
    domain: str
    version: str
    effective_date: str
    source_path: str


@dataclass(frozen=True)
class CorpusManifest:
    """Summary and ordered document entries for the complete corpus."""

    total_documents: int
    counts: dict[str, int]
    scanned_pdf_count: int
    documents: tuple[ManifestDocument, ...]


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_source_spec(spec: DocumentSpec, source_dir: Path) -> Path:
    path = source_dir / f"{spec.id}.json"
    _write_json(path, asdict(spec))
    return path


def _render(spec: DocumentSpec, corpus_dir: Path, font_path: Path) -> Path:
    if spec.fmt == "pdf":
        return render_pdf(spec, corpus_dir, font_path)
    return render_document(spec, corpus_dir)


def generate_corpus(output_dir: Path, font_path: Path) -> CorpusManifest:
    """Render all catalog entries, source specifications, and manifest.json."""
    corpus_dir = output_dir / "corpus"
    source_dir = output_dir / "source"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    documents = []
    for spec in build_catalog():
        source_path = _write_source_spec(spec, source_dir)
        artifact_path = _render(spec, corpus_dir, font_path)
        documents.append(
            ManifestDocument(
                id=spec.id,
                relative_path=artifact_path.relative_to(output_dir).as_posix(),
                sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                format=spec.fmt,
                scanned=spec.scanned,
                domain=spec.domain,
                version=spec.version,
                effective_date=spec.effective_date,
                source_path=source_path.relative_to(output_dir).as_posix(),
            )
        )

    counts = Counter(spec.fmt for spec in build_catalog())
    manifest = CorpusManifest(
        total_documents=len(documents),
        counts=dict(counts),
        scanned_pdf_count=sum(document.scanned for document in documents),
        documents=tuple(documents),
    )
    _write_json(output_dir / "manifest.json", asdict(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = generate_corpus(arguments.output, arguments.font)
    qa_manifest = generate_qa(build_catalog(), arguments.output)
    print(
        f"Generated {manifest.total_documents} documents "
        f"({manifest.scanned_pdf_count} scanned PDFs), "
        f"{qa_manifest.knowledge_qa_count} knowledge Q&A, and "
        f"{qa_manifest.evaluation_count} evaluation questions "
        f"({qa_manifest.answerable_count} answerable)."
    )


if __name__ == "__main__":
    main()
