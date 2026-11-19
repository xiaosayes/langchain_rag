from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

from app.db.models import DocumentRecord
from app.db.session import create_session_factory
from app.web.queue import TaskLeaseError, TaskQueue
from app.web.records import DocumentVersionRecord, TaskRecord


@pytest.fixture
def factory(tmp_path: Path):
    return create_session_factory(tmp_path / "tasks.sqlite3")


def add_version(factory, document_id: str = "doc-1", version: int = 1) -> str:
    version_id = f"{document_id}-v{version}"
    with factory.begin() as session:
        if session.get(DocumentRecord, document_id) is None:
            session.add(
                DocumentRecord(
                    id=document_id,
                    content_hash=f"hash-{document_id}",
                    metadata_json={},
                )
            )
        session.add(
            DocumentVersionRecord(
                id=version_id,
                document_id=document_id,
                version=version,
                path=f"raw/{version_id}.txt",
                sha256=f"hash-{version_id}",
            )
        )
    return version_id


def task_by_id(factory, task_id: str) -> TaskRecord:
    with factory() as session:
        return session.get(TaskRecord, task_id)


def test_only_one_worker_claims_task_and_expired_lease_recovers(factory):
    version_id = add_version(factory)
    queue = TaskQueue(factory)
    queued = queue.enqueue(document_id="doc-1", version_id=version_id, kind="ingest")

    first = queue.claim_next("worker-a", 1)
    assert first is not None
    assert first.id == queued.id
    assert first.status == "running"
    assert first.lease_owner == "worker-a"
    assert first.lease_expires_at > datetime.now(UTC)
    assert first.attempts == 1
    assert queue.claim_next("worker-b", 1) is None

    with factory.begin() as session:
        task = session.get(TaskRecord, first.id)
        task.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)

    assert queue.recover_expired() == 1
    recovered = task_by_id(factory, first.id)
    assert recovered.status == "queued"
    assert recovered.lease_owner is None
    assert recovered.lease_expires_at is None
    assert recovered.attempts == 1

    second = queue.claim_next("worker-b", 1)
    assert second is not None
    assert second.id == first.id
    assert second.attempts == 2


def test_competing_claims_are_serialized_by_sqlite(factory):
    version_id = add_version(factory)
    queue = TaskQueue(factory)
    queue.enqueue(document_id="doc-1", version_id=version_id, kind="ingest")

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(
            executor.map(
                lambda owner: queue.claim_next(owner, 30),
                ("worker-a", "worker-b"),
            )
        )

    assert sum(claim is not None for claim in claims) == 1
    claimed = next(claim for claim in claims if claim is not None)
    assert task_by_id(factory, claimed.id).attempts == 1


def test_claim_batch_claims_all_eligible_work_in_one_transaction(factory):
    first_version = add_version(factory, version=1)
    second_version = add_version(factory, version=2)
    other_version = add_version(factory, document_id="doc-2", version=1)
    queue = TaskQueue(factory)
    queued = [
        queue.enqueue("doc-1", first_version, "ingest"),
        queue.enqueue("doc-1", second_version, "ingest"),
        queue.enqueue("doc-2", other_version, "ingest"),
    ]

    claimed = queue.claim_batch("worker-a", 30)

    assert [task.id for task in claimed] == [task.id for task in queued]
    assert all(task.status == "running" for task in claimed)
    assert all(task.lease_owner == "worker-a" for task in claimed)
    assert all(task.attempts == 1 for task in claimed)
    assert queue.claim_batch("worker-b", 30) == []


def test_complete_requires_owner_and_supersedes_only_older_work(factory):
    old_version = add_version(factory, version=1)
    new_version = add_version(factory, version=2)
    queue = TaskQueue(factory)
    old = queue.enqueue("doc-1", old_version, "ingest")
    new = queue.enqueue("doc-1", new_version, "ingest")
    queue.claim_batch("worker-a", 30)

    with pytest.raises(TaskLeaseError):
        queue.complete(new.id, owner="worker-b")
    assert task_by_id(factory, old.id).status == "running"

    newest = queue.enqueue("doc-1", add_version(factory, version=3), "ingest")
    queue.complete(new.id, owner="worker-a")

    old_record = task_by_id(factory, old.id)
    new_record = task_by_id(factory, new.id)
    newest_record = task_by_id(factory, newest.id)
    assert old_record.status == "succeeded"
    assert old_record.payload_json["superseded_by"] == new.id
    assert old_record.lease_owner is None
    assert new_record.status == "succeeded"
    assert "superseded_by" not in new_record.payload_json
    assert newest_record.status == "queued"


@pytest.mark.parametrize("action", ["complete", "fail"])
def test_terminal_transition_rejects_omitted_owner_without_supersession(
    factory, action
):
    old_version = add_version(factory, version=1)
    new_version = add_version(factory, version=2)
    queue = TaskQueue(factory)
    old = queue.enqueue("doc-1", old_version, "ingest")
    new = queue.enqueue("doc-1", new_version, "ingest")
    queue.claim_batch("worker-a", 30)

    with pytest.raises(TypeError):
        if action == "complete":
            queue.complete(new.id)
        else:
            queue.fail(new.id, "must not be persisted")

    old_record = task_by_id(factory, old.id)
    new_record = task_by_id(factory, new.id)
    assert old_record.status == "running"
    assert old_record.payload_json == {}
    assert new_record.status == "running"
    assert new_record.lease_owner == "worker-a"
    assert new_record.lease_expires_at > datetime.now(UTC)
    assert new_record.attempts == 1
    assert new_record.error is None


@pytest.mark.parametrize("action", ["complete", "fail"])
@pytest.mark.parametrize(
    "condition", ["wrong-owner", "expired-lease", "missing-lease", "non-running"]
)
def test_terminal_transition_rejects_invalid_lease_without_supersession(
    factory, action, condition
):
    old_version = add_version(factory, version=1)
    new_version = add_version(factory, version=2)
    queue = TaskQueue(factory)
    old = queue.enqueue("doc-1", old_version, "ingest")
    new = queue.enqueue("doc-1", new_version, "ingest")
    queue.claim_batch("worker-a", 30)
    with factory.begin() as session:
        task = session.get(TaskRecord, new.id)
        if condition == "expired-lease":
            task.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        elif condition == "missing-lease":
            task.lease_expires_at = None
        elif condition == "non-running":
            task.status = "queued"

    owner = "worker-b" if condition == "wrong-owner" else "worker-a"
    with pytest.raises(TaskLeaseError):
        if action == "complete":
            queue.complete(new.id, owner=owner)
        else:
            queue.fail(new.id, "must not be persisted", owner=owner)

    old_record = task_by_id(factory, old.id)
    new_record = task_by_id(factory, new.id)
    assert old_record.status == "running"
    assert old_record.payload_json == {}
    assert new_record.status == ("queued" if condition == "non-running" else "running")
    assert new_record.lease_owner == "worker-a"
    if condition == "expired-lease":
        assert new_record.lease_expires_at <= datetime.now(UTC)
    elif condition == "missing-lease":
        assert new_record.lease_expires_at is None
    else:
        assert new_record.lease_expires_at > datetime.now(UTC)
    assert new_record.attempts == 1
    assert new_record.error is None


def test_failed_newest_task_does_not_supersede_older_work(factory):
    old_version = add_version(factory, version=1)
    new_version = add_version(factory, version=2)
    queue = TaskQueue(factory)
    old = queue.enqueue("doc-1", old_version, "ingest")
    new = queue.enqueue("doc-1", new_version, "ingest")
    queue.claim_batch("worker-a", 30)

    queue.fail(new.id, "embedding unavailable", owner="worker-a")

    old_record = task_by_id(factory, old.id)
    new_record = task_by_id(factory, new.id)
    assert old_record.status == "running"
    assert old_record.payload_json == {}
    assert new_record.status == "failed"
    assert new_record.error == "embedding unavailable"
    assert new_record.lease_owner is None


def test_claim_skips_tasks_that_exhausted_max_attempts(factory):
    version_id = add_version(factory)
    queue = TaskQueue(factory)
    queued = queue.enqueue("doc-1", version_id, "ingest")
    with factory.begin() as session:
        task = session.get(TaskRecord, queued.id)
        task.attempts = task.max_attempts

    assert queue.claim_next("worker-a", 30) is None
    assert queue.claim_batch("worker-a", 30) == []
    assert task_by_id(factory, queued.id).status == "queued"


def test_recovery_does_not_change_active_generation(factory):
    from app.db.models import IndexGenerationRecord

    version_id = add_version(factory)
    queue = TaskQueue(factory)
    claimed = queue.enqueue("doc-1", version_id, "ingest")
    queue.claim_next("worker-a", 30)
    with factory.begin() as session:
        session.add(IndexGenerationRecord(id="active", status="active"))
        task = session.get(TaskRecord, claimed.id)
        task.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)

    assert queue.recover_expired() == 1
    with factory() as session:
        active = session.scalars(
            select(IndexGenerationRecord).where(
                IndexGenerationRecord.status == "active"
            )
        ).one()
        assert active.id == "active"
