"""Persistent SQLite task leases with atomic claims and crash recovery."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from app.web.records import DocumentVersionRecord, TaskRecord


class TaskLeaseError(RuntimeError):
    """Raised when a terminal transition does not hold the current task lease."""


class TaskQueue:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def enqueue(
        self, document_id: str, version_id: str | None, kind: str
    ) -> TaskRecord:
        task = TaskRecord(
            id=str(uuid4()),
            document_id=document_id,
            version_id=version_id,
            kind=kind,
            status="queued",
            payload_json={},
        )
        with self.session_factory.begin() as session:
            session.add(task)
        return task

    def claim_next(self, owner: str, lease_seconds: int) -> TaskRecord | None:
        claimed = self._claim(owner, lease_seconds, batch=False)
        return claimed[0] if claimed else None

    def claim_batch(self, owner: str, lease_seconds: int) -> list[TaskRecord]:
        """Claim all currently eligible queued tasks in one SQLite transaction."""
        return self._claim(owner, lease_seconds, batch=True)

    def _claim(
        self, owner: str, lease_seconds: int, *, batch: bool
    ) -> list[TaskRecord]:
        if not owner:
            raise ValueError("Lease owner must not be empty")
        if lease_seconds <= 0:
            raise ValueError("Lease duration must be positive")

        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=lease_seconds)
        with self.session_factory() as session:
            # SQLite's immediate write lock makes the select/update claim indivisible.
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            id_query = (
                select(TaskRecord.id)
                .where(
                    TaskRecord.status == "queued",
                    TaskRecord.attempts < TaskRecord.max_attempts,
                )
                .order_by(TaskRecord.created_at, TaskRecord.id)
            )
            if not batch:
                id_query = id_query.limit(1)
            task_ids = list(session.scalars(id_query))
            if task_ids:
                session.execute(
                    update(TaskRecord)
                    .where(
                        TaskRecord.id.in_(task_ids),
                        TaskRecord.status == "queued",
                        TaskRecord.attempts < TaskRecord.max_attempts,
                    )
                    .values(
                        status="running",
                        lease_owner=owner,
                        lease_expires_at=expires_at,
                        attempts=TaskRecord.attempts + 1,
                        error=None,
                    )
                    .execution_options(synchronize_session=False)
                )
            claimed = list(
                session.scalars(
                    select(TaskRecord)
                    .where(TaskRecord.id.in_(task_ids))
                    .order_by(TaskRecord.created_at, TaskRecord.id)
                )
            )
            session.commit()
        return claimed

    def complete(self, task_id: str, owner: str) -> TaskRecord:
        """Finish a leased task and supersede only its older pending versions."""
        with self.session_factory() as session:
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            task = session.get(TaskRecord, task_id)
            self._require_lease(task, owner, task_id)
            candidates = list(
                session.scalars(
                    select(TaskRecord).where(
                        TaskRecord.document_id == task.document_id,
                        TaskRecord.id != task.id,
                        TaskRecord.status.in_(("queued", "running")),
                    )
                )
            )
            versions = self._version_numbers(session, [task, *candidates])
            for candidate in candidates:
                if not self._is_older(candidate, task, versions):
                    continue
                candidate.status = "succeeded"
                candidate.payload_json = {
                    **candidate.payload_json,
                    "superseded_by": task.id,
                }
                candidate.lease_owner = None
                candidate.lease_expires_at = None
                candidate.error = None
            task.status = "succeeded"
            task.lease_owner = None
            task.lease_expires_at = None
            task.error = None
            session.commit()
            return task

    def fail(self, task_id: str, error: str, owner: str) -> TaskRecord:
        """Fail only the currently leased task; never supersede other versions."""
        with self.session_factory() as session:
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            task = session.get(TaskRecord, task_id)
            self._require_lease(task, owner, task_id)
            task.status = "failed"
            task.error = error
            task.lease_owner = None
            task.lease_expires_at = None
            session.commit()
            return task

    def recover_expired(self) -> int:
        """Requeue expired running leases without touching generation state."""
        now = datetime.now(UTC)
        with self.session_factory() as session:
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            result = session.execute(
                update(TaskRecord)
                .where(
                    TaskRecord.status == "running",
                    TaskRecord.lease_expires_at.is_not(None),
                    TaskRecord.lease_expires_at <= now,
                )
                .values(status="queued", lease_owner=None, lease_expires_at=None)
                .execution_options(synchronize_session=False)
            )
            session.commit()
            return result.rowcount or 0

    @staticmethod
    def _require_lease(task: TaskRecord | None, owner: str, task_id: str) -> None:
        if (
            task is None
            or task.status != "running"
            or task.lease_owner is None
            or task.lease_owner != owner
            or task.lease_expires_at is None
            or task.lease_expires_at <= datetime.now(UTC)
        ):
            raise TaskLeaseError(
                f"Task {task_id} does not have an unexpired lease for this owner"
            )

    @staticmethod
    def _version_numbers(session: Session, tasks: list[TaskRecord]) -> dict[str, int]:
        version_ids = [task.version_id for task in tasks if task.version_id is not None]
        if not version_ids:
            return {}
        return dict(
            session.execute(
                select(DocumentVersionRecord.id, DocumentVersionRecord.version).where(
                    DocumentVersionRecord.id.in_(version_ids)
                )
            ).all()
        )

    @staticmethod
    def _is_older(
        candidate: TaskRecord,
        target: TaskRecord,
        versions: dict[str, int],
    ) -> bool:
        candidate_version = (
            versions.get(candidate.version_id)
            if candidate.version_id is not None
            else None
        )
        target_version = (
            versions.get(target.version_id) if target.version_id is not None else None
        )
        if (
            candidate_version is not None
            and target_version is not None
            and candidate_version != target_version
        ):
            return candidate_version < target_version
        return (candidate.created_at, candidate.id) < (target.created_at, target.id)
