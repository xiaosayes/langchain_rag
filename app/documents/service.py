"""Versioned document persistence and generation-atomic ingestion orchestration."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import BinaryIO, Protocol
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.bootstrap import build_index_builder
from app.config import AppSettings
from app.db.models import DocumentRecord
from app.documents.schemas import DocumentKind
from app.documents.storage import MEDIA_TYPES, DocumentStorage
from app.domain.models import ParsedBlock
from app.error_sanitization import sanitized_error_message
from app.ingestion.registry import ParserRegistry
from app.ingestion.text_office import QAParser
from app.web.progress import mark_failed
from app.web.queue import TaskQueue
from app.web.records import DocumentVersionRecord, TaskRecord


class Builder(Protocol):
    def build(self, documents: dict[str, list[ParsedBlock]]) -> str: ...


TaskSelection = Mapping[str, TaskRecord] | Iterable[TaskRecord]


class DocumentService:
    """Coordinate immutable candidates with the existing leased task queue."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        settings: AppSettings,
        *,
        storage: DocumentStorage | None = None,
        builder_factory: Callable[[AppSettings], Builder] | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.settings = settings
        self.storage = storage or DocumentStorage(
            settings.web.raw_file_dir,
            max_upload_bytes=settings.web.max_upload_mb * 1024 * 1024,
        )
        self.builder_factory = builder_factory or build_index_builder
        self.queue = TaskQueue(session_factory)

    def upload(
        self, filename: str, content: BinaryIO, kind: DocumentKind
    ) -> tuple[DocumentVersionRecord, TaskRecord]:
        """Persist a new candidate and queued task without replacing accepted data."""
        with self.session_factory() as session:
            existing = session.scalar(
                select(DocumentRecord)
                .where(
                    DocumentRecord.filename == filename,
                    DocumentRecord.deleted.is_(False),
                )
                .order_by(DocumentRecord.created_at, DocumentRecord.id)
                .limit(1)
            )
        document_id = existing.id if existing is not None else str(uuid4())
        stored = self.storage.store_upload(
            filename,
            content,
            kind,
            document_id=document_id,
        )

        session = self.session_factory()
        try:
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            document = session.get(DocumentRecord, stored.document_id)
            if document is None:
                document = DocumentRecord(
                    id=stored.document_id,
                    content_hash=stored.sha256,
                    metadata_json={"kind": stored.kind},
                    filename=filename,
                    media_type=stored.media_type,
                    status="uploaded",
                    current_version_id=None,
                )
                session.add(document)
            else:
                document.status = "uploaded"
                document.error = None

            next_version = (
                session.scalar(
                    select(
                        func.coalesce(func.max(DocumentVersionRecord.version), 0)
                    ).where(DocumentVersionRecord.document_id == stored.document_id)
                )
                or 0
            ) + 1
            version = DocumentVersionRecord(
                id=stored.version_id,
                document_id=stored.document_id,
                version=next_version,
                path=str(stored.path),
                sha256=stored.sha256,
                status="uploaded",
            )
            task = TaskRecord(
                id=str(uuid4()),
                document_id=stored.document_id,
                version_id=stored.version_id,
                kind=stored.kind,
                status="queued",
                payload_json={},
            )
            session.add_all([version, task])
            session.commit()
            return version, task
        except Exception:
            session.rollback()
            shutil.rmtree(stored.path.parent, ignore_errors=True)
            raise
        finally:
            session.close()

    def documents_for_tasks(self, tasks: TaskSelection) -> dict[str, list[ParsedBlock]]:
        """Parse ready content with only the newest viable candidate per document."""
        task_list = list(tasks.values()) if isinstance(tasks, Mapping) else list(tasks)
        selected = self._newest_tasks(task_list)
        with self.session_factory() as session:
            documents = list(
                session.scalars(
                    select(DocumentRecord)
                    .where(DocumentRecord.deleted.is_(False))
                    .order_by(DocumentRecord.id)
                )
            )
            version_ids = {
                version_id
                for version_id in (task.version_id for task in selected.values())
                if version_id is not None
            }
            version_ids.update(
                document.current_version_id
                for document in documents
                if document.current_version_id is not None
            )
            versions = (
                {
                    version.id: version
                    for version in session.scalars(
                        select(DocumentVersionRecord).where(
                            DocumentVersionRecord.id.in_(version_ids)
                        )
                    )
                }
                if version_ids
                else {}
            )

            sources: list[tuple[str, Path, str]] = []
            for document in documents:
                task = selected.get(document.id)
                version = (
                    versions.get(task.version_id)
                    if task is not None and task.version_id is not None
                    else None
                )
                kind = task.kind if task is not None else "document"
                if version is None or version.status == "failed":
                    current_id = document.current_version_id
                    version = (
                        versions.get(current_id) if current_id is not None else None
                    )
                    if version is None or version.status != "ready":
                        continue
                    kind = str(document.metadata_json.get("kind", "document"))
                if Path(version.path).name.casefold() == "evaluation.jsonl":
                    continue
                sources.append((document.id, Path(version.path), kind))

        registry = ParserRegistry.default()
        parsed: dict[str, list[ParsedBlock]] = {}
        for document_id, path, kind in sources:
            parser = QAParser() if kind == "qa" else registry
            parsed[document_id] = parser.parse(path, document_id)
        return parsed

    def build_pending(self, owner: str) -> str:
        """Build one claimed batch and publish candidates only after activation."""
        self.queue.recover_expired()
        claimed = self.queue.claim_batch(owner, self.settings.web.task_lease_seconds)
        if not claimed:
            return ""
        selected = self._newest_tasks(claimed)
        newest = list(selected.values())
        try:
            documents = self.documents_for_tasks(selected)
            generation_id = self.builder_factory(self.settings).build(documents)
        except Exception as error:
            message = sanitized_error_message(error)
            for task in newest:
                self.queue.fail(task.id, message, owner=owner)
                mark_failed(self.session_factory, task.id, message)
            raise

        for task in newest:
            self._publish(task.id, generation_id)
            self.queue.complete(task.id, owner=owner)
        return generation_id

    def _newest_tasks(self, tasks: Iterable[TaskRecord]) -> dict[str, TaskRecord]:
        candidates = [
            task
            for task in tasks
            if task.status in {"queued", "running"} and task.version_id is not None
        ]
        version_ids = [task.version_id for task in candidates]
        with self.session_factory() as session:
            versions = (
                {
                    version.id: version
                    for version in session.scalars(
                        select(DocumentVersionRecord).where(
                            DocumentVersionRecord.id.in_(version_ids)
                        )
                    )
                }
                if version_ids
                else {}
            )
        selected: dict[str, TaskRecord] = {}
        order: dict[str, tuple[int, object, str]] = {}
        for task in candidates:
            version = versions.get(task.version_id)
            if version is None:
                raise RuntimeError(f"Document version {task.version_id} does not exist")
            if version.status == "failed":
                continue
            task_order = (version.version, task.created_at, task.id)
            if task.document_id not in selected or task_order > order[task.document_id]:
                selected[task.document_id] = task
                order[task.document_id] = task_order
        return selected

    def _publish(self, task_id: str, generation_id: str) -> None:
        with self.session_factory.begin() as session:
            task = session.get(TaskRecord, task_id)
            if task is None or task.version_id is None:
                raise RuntimeError(f"Task {task_id} does not reference a version")
            version = session.get(DocumentVersionRecord, task.version_id)
            document = session.get(DocumentRecord, task.document_id)
            if version is None or document is None:
                raise RuntimeError(f"Task {task_id} references missing document state")
            task.payload_json = {
                **task.payload_json,
                "progress": "ready",
                "generation_id": generation_id,
            }
            version.status = "ready"
            version.error = None
            document.current_version_id = version.id
            document.content_hash = version.sha256
            document.filename = Path(version.path).name
            document.media_type = MEDIA_TYPES[Path(version.path).suffix.lower()]
            document.metadata_json = {
                **document.metadata_json,
                "kind": task.kind,
            }
            document.status = "ready"
            document.error = None
