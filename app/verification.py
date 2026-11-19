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
from app.error_sanitization import sanitized_error_message
from app.indexing.repository import IndexRepository
from app.indexing.vector_space import VectorSpace
from app.ingestion.pdf import has_usable_text_layer
from app.ingestion.text_office import QAParser
from app.retrieval.bm25 import PersistentBM25

EXPECTED_ORDINARY_DOCUMENTS = 100
EXPECTED_KNOWLEDGE_QA = 1000
_SUPPORTED_SUFFIXES = frozenset({".txt", ".docx", ".xlsx", ".pdf"})
_EXPECTED_BY_EXTENSION = {".txt": 25, ".docx": 25, ".xlsx": 25, ".pdf": 25}
_EXPECTED_PDF_COUNTS = {"scanned": 5, "text": 20}


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
        message=sanitized_error_message(error),
    )


def _pdf_counts(paths: list[Path]) -> dict[str, int]:
    scanned = 0
    for path in paths:
        with fitz.open(path) as document:
            has_text_layer = any(
                has_usable_text_layer(page.get_text("text").strip())
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
                        name="vector_space_compatibility",
                        ok=False,
                        expected=VectorSpace.from_settings(self.settings).fingerprint,
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
        checks.append(self._check_vector_space(repository, generation_id))
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

    def _check_vector_space(
        self, repository: IndexRepository, generation_id: str | None
    ) -> VerificationCheck:
        expected_space = VectorSpace.from_settings(self.settings)
        expected = {
            "provider": expected_space.provider,
            "model": expected_space.model,
            "dimension": expected_space.dimension,
            "distance": expected_space.distance,
            "fingerprint": expected_space.fingerprint,
        }
        actual = (
            repository.vector_space_fingerprint(generation_id)
            if generation_id is not None
            else None
        )
        ok = generation_id is not None and actual == expected_space.fingerprint
        return VerificationCheck(
            name="vector_space_compatibility",
            ok=ok,
            expected=expected,
            actual={"generation_id": generation_id, "fingerprint": actual},
            message=(
                "active generation vector space is compatible"
                if ok
                else "active generation vector space is unknown or incompatible; "
                "run rebuild with the current provider, model, dimension, and distance"
            ),
        )

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
        ok = (
            len(paths) == EXPECTED_ORDINARY_DOCUMENTS
            and actual["by_extension"] == _EXPECTED_BY_EXTENSION
            and actual["pdf"] == _EXPECTED_PDF_COUNTS
        )
        expected = {
            "total": EXPECTED_ORDINARY_DOCUMENTS,
            "by_extension": _EXPECTED_BY_EXTENSION,
            "pdf": _EXPECTED_PDF_COUNTS,
        }
        return VerificationCheck(
            name="ordinary_documents",
            ok=ok,
            expected=expected,
            actual=actual,
            message=(
                "ordinary document and PDF shape passed"
                if ok
                else f"expected {expected}, found {actual}"
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
            distance = (
                vectors.distance
                if isinstance(vectors, models.VectorParams)
                else None
            )
            distance_name = (
                distance.value.lower() if distance is not None else None
            )
            point_count = qdrant.client.count(
                qdrant.collection_name(generation_id), exact=True
            ).count
            qdrant_child_ids = qdrant.child_ids(generation_id)
            with repository.session_factory() as session:
                database_child_ids = set(
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
        child_count = len(database_child_ids)
        child_ids_match = qdrant_child_ids == database_child_ids
        actual = {
            "generation_id": generation_id,
            "dimension": dimension,
            "distance": distance_name,
            "point_count": point_count,
            "child_count": child_count,
            "child_ids_match": child_ids_match,
            "missing_child_ids": sorted(database_child_ids - qdrant_child_ids),
            "unexpected_child_ids": sorted(qdrant_child_ids - database_child_ids),
        }
        ok = (
            dimension == self.settings.embedding.dimension
            and distance_name == self.settings.qdrant.distance
            and point_count == child_count
            and child_ids_match
        )
        return VerificationCheck(
            name="qdrant_dimension",
            ok=ok,
            expected={
                "dimension": self.settings.embedding.dimension,
                "distance": self.settings.qdrant.distance,
                "point_count": child_count,
                "child_ids_match": True,
            },
            actual=actual,
            message=(
                f"Qdrant dimension {dimension}, distance {distance_name}, "
                f"points {point_count}, database children {child_count}, "
                f"child IDs match: {child_ids_match}"
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
            bm25 = PersistentBM25.load(generation_dir)
            manifest = json.loads(
                (generation_dir / "manifest.json").read_text(encoding="utf-8")
            )
            with repository.session_factory() as session:
                database_child_ids = set(
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
        child_count = len(database_child_ids)
        child_ids_match = bm25.child_ids == database_child_ids
        actual = {
            **manifest,
            "generation_id": generation_id,
            "child_ids_match": child_ids_match,
            "missing_child_ids": sorted(database_child_ids - bm25.child_ids),
            "unexpected_child_ids": sorted(bm25.child_ids - database_child_ids),
        }
        ok = manifest.get("chunk_count") == child_count and child_ids_match
        return VerificationCheck(
            name="bm25_manifest",
            ok=ok,
            expected={
                "chunk_count": child_count,
                "schema_version": 1,
                "tokenizer": "jieba",
                "child_ids_match": True,
            },
            actual=actual,
            message=(
                f"BM25 manifest chunks {manifest.get('chunk_count')}, "
                f"database children {child_count}, child IDs match: {child_ids_match}"
            ),
        )
