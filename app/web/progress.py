"""Persist indexing stage callbacks and terminal document-version state."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.orm import Session, sessionmaker

from app.db.models import DocumentRecord
from app.web.records import DocumentVersionRecord, TaskRecord

_INDEX_STAGES = frozenset({"chunking", "embedding", "indexing"})


class IndexProgress:
    """Apply one builder stage to every newest task in a coalesced batch."""

    def __init__(
        self, session_factory: sessionmaker[Session], task_ids: Sequence[str]
    ) -> None:
        self.session_factory = session_factory
        self.task_ids = tuple(task_ids)

    def __call__(self, stage: str) -> None:
        if stage not in _INDEX_STAGES:
            raise ValueError(f"Unsupported index progress stage: {stage}")
        with self.session_factory.begin() as session:
            for task_id in self.task_ids:
                task = _require_task(session, task_id)
                task.payload_json = {**task.payload_json, "progress": stage}
                version = _version(session, task)
                if version is not None and task.kind != "delete":
                    version.status = stage
                    version.error = None
                document = session.get(DocumentRecord, task.document_id)
                if document is not None:
                    document.status = stage
                    document.error = None


def mark_ready(
    session_factory: sessionmaker[Session], task_id: str, generation_id: str
) -> None:
    """Publish terminal web state only after the generation has activated."""
    with session_factory.begin() as session:
        task = _require_task(session, task_id)
        task.payload_json = {
            **task.payload_json,
            "progress": "ready",
            "generation_id": generation_id,
        }
        version = _version(session, task)
        if version is not None:
            version.status = "ready"
            version.error = None
        document = session.get(DocumentRecord, task.document_id)
        if document is not None:
            if task.kind == "delete":
                document.deleted = True
                document.status = "deleted"
            else:
                document.status = "ready"
                document.current_version_id = task.version_id
            document.error = None


def mark_failed(
    session_factory: sessionmaker[Session], task_id: str, error: str
) -> None:
    """Expose a failed candidate without replacing the current ready version."""
    with session_factory.begin() as session:
        task = _require_task(session, task_id)
        task.payload_json = {**task.payload_json, "progress": "failed"}
        version = _version(session, task)
        document = session.get(DocumentRecord, task.document_id)
        if task.kind == "delete":
            # A failed delete leaves the accepted version ready and downloadable.
            if version is not None:
                version.status = "ready"
                version.error = None
            if document is not None:
                document.status = "ready"
                document.error = None
            return
        if version is not None:
            version.status = "failed"
            version.error = error
        if document is not None:
            document.status = "failed"
            document.error = error


def _require_task(session: Session, task_id: str) -> TaskRecord:
    task = session.get(TaskRecord, task_id)
    if task is None:
        raise RuntimeError(f"Task {task_id} does not exist")
    return task


def _version(session: Session, task: TaskRecord) -> DocumentVersionRecord | None:
    if task.version_id is None:
        return None
    return session.get(DocumentVersionRecord, task.version_id)
