"""Transactional persistence for immutable, generation-scoped index data."""

import hashlib
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import (
    ChildChunkRecord,
    GenerationDocumentRecord,
    IndexGenerationRecord,
    ParentChunkRecord,
)
from app.domain.models import ChildChunk, ParentChunk


class GenerationStateError(RuntimeError):
    """Raised when a generation is missing or not in the required state."""


class IndexRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def prepare_generation(self, generation_id: str) -> None:
        with self.session_factory.begin() as session:
            session.add(IndexGenerationRecord(id=generation_id, status="building"))

    def active_generation(self) -> str | None:
        with self.session_factory() as session:
            return session.scalars(
                select(IndexGenerationRecord.id).where(
                    IndexGenerationRecord.status == "active"
                )
            ).one_or_none()

    def parents(self, generation_id: str, parent_ids: list[str]) -> list[ParentChunk]:
        """Resolve IDs only within the requested generation and preserve ID order."""
        if not parent_ids:
            return []
        with self.session_factory() as session:
            records = session.scalars(
                select(ParentChunkRecord).where(
                    ParentChunkRecord.generation_id == generation_id,
                    ParentChunkRecord.id.in_(set(parent_ids)),
                )
            ).all()
        by_id = {record.id: record for record in records}
        return [
            self._parent(by_id[parent_id])
            for parent_id in parent_ids
            if parent_id in by_id
        ]

    def reusable_vectors(self, children: list[ChildChunk]) -> set[str]:
        """Return child IDs with identical content hashes in the active generation."""
        if not children:
            return set()
        by_id = {child.id: child for child in children}
        with self.session_factory() as session:
            generation_id = session.scalars(
                select(IndexGenerationRecord.id).where(
                    IndexGenerationRecord.status == "active"
                )
            ).one_or_none()
            if generation_id is None:
                return set()
            rows = session.execute(
                select(ChildChunkRecord.id, ChildChunkRecord.content_hash).where(
                    ChildChunkRecord.generation_id == generation_id,
                    ChildChunkRecord.id.in_(by_id),
                )
            ).all()
        previous_hashes = {row.id: row.content_hash for row in rows}
        return {
            child_id
            for child_id, content_hash in previous_hashes.items()
            if by_id[child_id].content_hash == content_hash
        }

    def activate_generation(
        self,
        generation_id: str,
        parents: Sequence[ParentChunk],
        children: Sequence[ChildChunk] = (),
    ) -> None:
        """Persist a staged generation and atomically make it the sole active one."""
        with self.session_factory.begin() as session:
            staged = session.get(IndexGenerationRecord, generation_id)
            if staged is None or staged.status != "building":
                raise GenerationStateError(generation_id)

            previous = session.scalars(
                select(IndexGenerationRecord).where(
                    IndexGenerationRecord.status == "active"
                )
            ).one_or_none()

            session.add_all(self._document_records(generation_id, parents))
            session.add_all(
                ParentChunkRecord(
                    id=parent.id,
                    generation_id=generation_id,
                    document_id=parent.document_id,
                    ordinal=parent.ordinal,
                    text=parent.text,
                    content_hash=parent.content_hash,
                    metadata_json=dict(parent.metadata),
                )
                for parent in parents
            )
            session.flush()
            staged_parent_count = session.scalar(
                select(func.count())
                .select_from(ParentChunkRecord)
                .where(ParentChunkRecord.generation_id == generation_id)
            )
            if staged_parent_count != len(parents):
                raise GenerationStateError(
                    f"{generation_id}: staged parent count does not match input"
                )

            session.add_all(
                ChildChunkRecord(
                    id=child.id,
                    generation_id=generation_id,
                    parent_id=child.parent_id,
                    document_id=child.document_id,
                    ordinal=child.ordinal,
                    content_hash=child.content_hash,
                )
                for child in children
            )
            session.flush()

            if previous is not None:
                previous.status = "retired"
                session.flush()
            staged.status = "active"

    def mark_failed(self, generation_id: str) -> None:
        """Fail only a still-building generation, never an activated predecessor."""
        with self.session_factory.begin() as session:
            staged = session.get(IndexGenerationRecord, generation_id)
            if staged is not None and staged.status == "building":
                staged.status = "failed"

    @staticmethod
    def _parent(record: ParentChunkRecord) -> ParentChunk:
        return ParentChunk(
            id=record.id,
            document_id=record.document_id,
            ordinal=record.ordinal,
            text=record.text,
            content_hash=record.content_hash,
            metadata=dict(record.metadata_json),
        )

    @staticmethod
    def _document_records(
        generation_id: str, parents: Sequence[ParentChunk]
    ) -> list[GenerationDocumentRecord]:
        parents_by_document: dict[str, list[ParentChunk]] = {}
        for parent in parents:
            parents_by_document.setdefault(parent.document_id, []).append(parent)

        records: list[GenerationDocumentRecord] = []
        for document_id, document_parents in parents_by_document.items():
            ordered = sorted(
                document_parents, key=lambda parent: (parent.ordinal, parent.id)
            )
            content_hash = hashlib.sha256(
                "\x1f".join(parent.content_hash for parent in ordered).encode("utf-8")
            ).hexdigest()
            records.append(
                GenerationDocumentRecord(
                    id=document_id,
                    generation_id=generation_id,
                    content_hash=content_hash,
                    metadata_json={},
                )
            )
        return records
