"""Single ingestion worker for persistent, generation-atomic task batches."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from sqlalchemy import select

from app.bootstrap import build_index_builder, state_directory
from app.config import load_settings
from app.db.models import DocumentRecord
from app.db.session import create_session_factory
from app.domain.models import ParsedBlock
from app.error_sanitization import sanitized_error_message
from app.ingestion.registry import ParserRegistry
from app.ingestion.text_office import QAParser
from app.web.progress import IndexProgress, mark_failed, mark_ready
from app.web.queue import TaskQueue
from app.web.records import DocumentVersionRecord, TaskRecord


class Builder(Protocol):
    def build(
        self,
        documents: dict[str, list[ParsedBlock]],
        progress: Callable[[str], None] | None = None,
    ) -> str: ...


DocumentLoader = Callable[[Mapping[str, TaskRecord]], dict[str, list[ParsedBlock]]]


class IngestionWorker:
    """Recover and build one coalesced batch; no background tasks or subprocesses."""

    def __init__(
        self,
        queue: TaskQueue,
        builder: Builder,
        *,
        owner: str = "ingestion-worker",
        lease_seconds: int = 300,
        document_loader: DocumentLoader | None = None,
    ) -> None:
        self.queue = queue
        self.builder = builder
        self.owner = owner
        self.lease_seconds = lease_seconds
        self.document_loader = document_loader or self._load_documents

    def run_once(self) -> int:
        """Recover expired leases and attempt at most one generation build."""
        self.queue.recover_expired()
        claimed = self.queue.claim_batch(self.owner, self.lease_seconds)
        if not claimed:
            return 0
        selected = self._newest_tasks(claimed)
        newest = list(selected.values())
        try:
            documents = self.document_loader(selected)
            generation_id = self.builder.build(
                documents,
                progress=IndexProgress(
                    self.queue.session_factory, [task.id for task in newest]
                ),
            )
        except Exception as error:  # noqa: BLE001 - every batch failure is persisted.
            message = sanitized_error_message(error)
            for task in newest:
                self.queue.fail(task.id, message, owner=self.owner)
                mark_failed(self.queue.session_factory, task.id, message)
            return 1

        # IndexBuilder has activated the generation before this terminal transition.
        for task in newest:
            mark_ready(self.queue.session_factory, task.id, generation_id)
            self.queue.complete(task.id, owner=self.owner)
        return 1

    def _newest_tasks(self, claimed: list[TaskRecord]) -> dict[str, TaskRecord]:
        version_ids = [
            task.version_id for task in claimed if task.version_id is not None
        ]
        with self.queue.session_factory() as session:
            versions = (
                dict(
                    session.execute(
                        select(
                            DocumentVersionRecord.id, DocumentVersionRecord.version
                        ).where(DocumentVersionRecord.id.in_(version_ids))
                    ).all()
                )
                if version_ids
                else {}
            )
        selected: dict[str, TaskRecord] = {}
        order: dict[str, tuple[int, object, str]] = {}
        for task in claimed:
            version = (
                versions.get(task.version_id, -1) if task.version_id is not None else -1
            )
            task_order = (version, task.created_at, task.id)
            if task.document_id not in selected or task_order > order[task.document_id]:
                selected[task.document_id] = task
                order[task.document_id] = task_order
        return selected

    def _load_documents(
        self, selected: Mapping[str, TaskRecord]
    ) -> dict[str, list[ParsedBlock]]:
        """Parse the full corpus, replacing current versions with claimed versions."""
        pending: list[tuple[str, Path, str]] = []
        with self.queue.session_factory() as session:
            documents = list(
                session.scalars(
                    select(DocumentRecord)
                    .where(DocumentRecord.deleted.is_(False))
                    .order_by(DocumentRecord.id)
                )
            )
            for document in documents:
                task = selected.get(document.id)
                version_id = (
                    task.version_id if task is not None else document.current_version_id
                )
                if version_id is None:
                    continue
                version = session.get(DocumentVersionRecord, version_id)
                if version is None:
                    raise RuntimeError(f"Document version {version_id} does not exist")
                kind = task.kind if task is not None else "ingest"
                pending.append((document.id, Path(version.path), kind))

        registry = ParserRegistry.default()
        parsed: dict[str, list[ParsedBlock]] = {}
        for document_id, path, kind in pending:
            parser = (
                QAParser()
                if kind == "qa" or document_id == "knowledge_qa"
                else registry
            )
            parsed[document_id] = parser.parse(path, document_id)
        return parsed


def main() -> None:
    """Run the one ingestion worker against the configured persistent state."""
    settings = load_settings()
    root = state_directory()
    root.mkdir(parents=True, exist_ok=True)
    queue = TaskQueue(create_session_factory(root / "index.sqlite3"))
    builder = build_index_builder(settings)
    worker = IngestionWorker(
        queue,
        builder,
        lease_seconds=settings.web.task_lease_seconds,
    )
    try:
        while True:
            if worker.run_once() == 0:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        builder.qdrant.client.close()


if __name__ == "__main__":
    main()
