"""Synchronous, structured integrity checks for a complete core installation."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fitz
from qdrant_client import models
from sqlalchemy import select

from app.bootstrap import _qdrant, _repository
from app.config import AppSettings
from app.db.models import ChildChunkRecord, ParentChunkRecord
from app.indexing.repository import IndexRepository
from app.ingestion.text_office import QAParser
from app.retrieval.bm25 import PersistentBM25

EXPECTED_ORDINARY_DOCUMENTS = 100
EXPECTED_KNOWLEDGE_QA = 1000
_SUPPORTED_SUFFIXES = frozenset({".txt", ".docx", ".xlsx", ".pdf"})
_MINIMUM_PDF_TEXT_CHARACTERS = 20


def ordinary_document_paths(corpus: Path) -> list[Path]:
    """Select only ordinary corpus documents, never the held-out evaluation set."""
    if not corpus.is_dir():
        return []
    return sorted(
        path
        for path in corpus.iterdir()
        if path.is_file()
        and path.suffix.lower() in _SUPPORTED_SUFFIXES
        and path.name != "knowledge_qa.xlsx"
    )


@dataclass(frozen=True)
class VerificationCheck:
    name: str
    ok: bool
    expected: Any
    actual: Any
    message: str


@dataclass(frozen=True)
class VerificationReport:
    checks: list[VerificationCheck]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": [asdict(check) for check in self.checks]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def _error_check(
    name: str, expected: Any, error: Exception, actual: Any = None
) -> VerificationCheck:
    return VerificationCheck(
        name=name,
        ok=False,
        expected=expected,
        actual=actual,
        message=f"{type(error).__name__}: {error}",
    )


def _pdf_counts(paths: list[Path]) -> dict[str, int]:
    scanned = 0
    for path in paths:
        with fitz.open(path) as document:
            has_text_layer = any(
                len(page.get_text("text").strip()) >= _MINIMUM_PDF_TEXT_CHARACTERS
                for page in document
            )
        if not has_text_layer:
            scanned += 1
    return {"scanned": scanned, "text": len(paths) - scanned}


class CoreVerifier:
    def __init__(self, settings: AppSettings, corpus: Path, qa: Path) -> None:
        self.settings = settings
        self.corpus = corpus
        self.qa = qa

    def run(self) -> VerificationReport:
        """Return all check results; command exit policy belongs to the CLI."""
        checks = [self._check_corpus(), self._check_qa()]
        try:
            repository = _repository()
            generation_id = repository.active_generation()
        except Exception as error:  # noqa: BLE001 - storage failures are check data.
            checks.append(
                _error_check("active_generation", "one active generation", error)
            )
            checks.extend(
                [
                    VerificationCheck(
                        name="parent_child_integrity",
                        ok=False,
                        expected="every child references a parent in the same generation/document",
                        actual=None,
                        message="not checked: active generation unavailable",
                    ),
                    VerificationCheck(
                        name="qdrant_dimension",
                        ok=False,
                        expected=self.settings.embedding.dimension,
                        actual=None,
                        message="not checked: active generation unavailable",
                    ),
                    VerificationCheck(
                        name="bm25_manifest",
                        ok=False,
                        expected="compatible active-generation manifest",
                        actual=None,
                        message="not checked: active generation unavailable",
                    ),
                ]
            )
            return VerificationReport(checks)

        checks.append(
            VerificationCheck(
                name="active_generation",
                ok=generation_id is not None,
                expected="one active generation",
                actual=generation_id,
                message=(
                    f"active generation: {generation_id}"
                    if generation_id is not None
                    else "no active generation"
                ),
            )
        )
        try:
            checks.append(self._check_parent_children(repository, generation_id))
        except Exception as error:  # noqa: BLE001 - query failures are check data.
            checks.append(
                _error_check(
                    "parent_child_integrity",
                    "every child references a parent in the same generation/document",
                    error,
                )
            )

        try:
            qdrant = _qdrant(self.settings)
        except Exception as error:  # noqa: BLE001 - client failures are check data.
            checks.append(
                _error_check(
                    "qdrant_dimension", self.settings.embedding.dimension, error
                )
            )
        else:
            try:
                checks.append(self._check_qdrant(qdrant, repository, generation_id))
            finally:
                qdrant.client.close()
        checks.append(self._check_bm25(repository, generation_id))
        return VerificationReport(checks)

    def _check_corpus(self) -> VerificationCheck:
        actual: dict[str, Any] = {}
        try:
            paths = ordinary_document_paths(self.corpus)
            by_extension = {
                suffix: sum(path.suffix.lower() == suffix for path in paths)
                for suffix in sorted(_SUPPORTED_SUFFIXES)
            }
            actual = {
                "total": len(paths),
                "by_extension": by_extension,
            }
            actual["pdf"] = _pdf_counts(
                [path for path in paths if path.suffix.lower() == ".pdf"]
            )
        except Exception as error:  # noqa: BLE001 - checks must report bad sources.
            return _error_check(
                "ordinary_documents",
                EXPECTED_ORDINARY_DOCUMENTS,
                error,
                actual or None,
            )
        return VerificationCheck(
            name="ordinary_documents",
            ok=len(paths) == EXPECTED_ORDINARY_DOCUMENTS,
            expected=EXPECTED_ORDINARY_DOCUMENTS,
            actual=actual,
            message=(
                f"expected {EXPECTED_ORDINARY_DOCUMENTS} ordinary documents, "
                f"found {len(paths)}"
            ),
        )

    def _check_qa(self) -> VerificationCheck:
        try:
            count = len(QAParser().parse(self.qa))
        except Exception as error:  # noqa: BLE001 - malformed workbooks are data.
            return _error_check("knowledge_qa", EXPECTED_KNOWLEDGE_QA, error)
        return VerificationCheck(
            name="knowledge_qa",
            ok=count == EXPECTED_KNOWLEDGE_QA,
            expected=EXPECTED_KNOWLEDGE_QA,
            actual=count,
            message=f"expected {EXPECTED_KNOWLEDGE_QA} knowledge Q&A rows, found {count}",
        )

    @staticmethod
    def _check_parent_children(
        repository: IndexRepository, generation_id: str | None
    ) -> VerificationCheck:
        if generation_id is None:
            return VerificationCheck(
                name="parent_child_integrity",
                ok=False,
                expected="every child references a parent in the same generation/document",
                actual=None,
                message="not checked: no active generation",
            )
        with repository.session_factory() as session:
            parents = session.scalars(
                select(ParentChunkRecord).where(
                    ParentChunkRecord.generation_id == generation_id
                )
            ).all()
            children = session.scalars(
                select(ChildChunkRecord).where(
                    ChildChunkRecord.generation_id == generation_id
                )
            ).all()
        parent_by_id = {parent.id: parent for parent in parents}
        missing_parents = sorted(
            {
                child.parent_id
                for child in children
                if child.parent_id not in parent_by_id
            }
        )
        document_mismatches = sorted(
            child.id
            for child in children
            if child.parent_id in parent_by_id
            and parent_by_id[child.parent_id].document_id != child.document_id
        )
        actual = {
            "generation_id": generation_id,
            "parent_count": len(parents),
            "child_count": len(children),
            "missing_parents": missing_parents,
            "document_mismatches": document_mismatches,
        }
        return VerificationCheck(
            name="parent_child_integrity",
            ok=not missing_parents and not document_mismatches,
            expected="every child references a parent in the same generation/document",
            actual=actual,
            message=(
                "parent/child integrity passed"
                if not missing_parents and not document_mismatches
                else f"missing parent IDs: {missing_parents}; "
                f"document mismatches: {document_mismatches}"
            ),
        )

    def _check_qdrant(
        self,
        qdrant: Any,
        repository: IndexRepository,
        generation_id: str | None,
    ) -> VerificationCheck:
        if generation_id is None:
            return VerificationCheck(
                name="qdrant_dimension",
                ok=False,
                expected=self.settings.embedding.dimension,
                actual=None,
                message="not checked: no active generation",
            )
        try:
            collection = qdrant.client.get_collection(
                qdrant.collection_name(generation_id)
            )
            vectors = collection.config.params.vectors
            dimension = (
                vectors.size if isinstance(vectors, models.VectorParams) else None
            )
            point_count = qdrant.client.count(
                qdrant.collection_name(generation_id), exact=True
            ).count
            with repository.session_factory() as session:
                child_count = len(
                    session.scalars(
                        select(ChildChunkRecord.id).where(
                            ChildChunkRecord.generation_id == generation_id
                        )
                    ).all()
                )
        except Exception as error:  # noqa: BLE001 - unavailable storage is a check.
            return _error_check(
                "qdrant_dimension", self.settings.embedding.dimension, error
            )
        actual = {
            "generation_id": generation_id,
            "dimension": dimension,
            "point_count": point_count,
            "child_count": child_count,
        }
        ok = (
            dimension == self.settings.embedding.dimension
            and point_count == child_count
        )
        return VerificationCheck(
            name="qdrant_dimension",
            ok=ok,
            expected={
                "dimension": self.settings.embedding.dimension,
                "point_count": child_count,
            },
            actual=actual,
            message=(
                f"Qdrant dimension {dimension}, points {point_count}, "
                f"database children {child_count}"
            ),
        )

    def _check_bm25(
        self, repository: IndexRepository, generation_id: str | None
    ) -> VerificationCheck:
        if generation_id is None:
            return VerificationCheck(
                name="bm25_manifest",
                ok=False,
                expected="compatible manifest matching the active child count",
                actual=None,
                message="not checked: no active generation",
            )
        try:
            from app.bootstrap import state_directory

            generation_dir = state_directory() / "bm25" / generation_id
            PersistentBM25.load(generation_dir)
            manifest = json.loads(
                (generation_dir / "manifest.json").read_text(encoding="utf-8")
            )
            with repository.session_factory() as session:
                child_count = len(
                    session.scalars(
                        select(ChildChunkRecord.id).where(
                            ChildChunkRecord.generation_id == generation_id
                        )
                    ).all()
                )
        except Exception as error:  # noqa: BLE001 - invalid state is a check result.
            return _error_check(
                "bm25_manifest", "compatible active-generation manifest", error
            )
        actual = {**manifest, "generation_id": generation_id}
        ok = manifest.get("chunk_count") == child_count
        return VerificationCheck(
            name="bm25_manifest",
            ok=ok,
            expected={
                "chunk_count": child_count,
                "schema_version": 1,
                "tokenizer": "jieba",
            },
            actual=actual,
            message=(
                f"BM25 manifest chunks {manifest.get('chunk_count')}, "
                f"database children {child_count}"
            ),
        )
