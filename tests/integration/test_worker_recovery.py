from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from app.config import ChunkSettings
from app.db.models import DocumentRecord, IndexGenerationRecord
from app.db.session import create_session_factory
from app.domain.models import ChildChunk, ParentChunk
from app.indexing.builder import IndexBuilder
from app.indexing.repository import IndexRepository
from app.indexing.vector_space import VectorSpace
from app.web.queue import TaskQueue
from app.web.records import DocumentVersionRecord, TaskRecord
from app.worker import IngestionWorker


@pytest.fixture
def factory(tmp_path: Path):
    return create_session_factory(tmp_path / "worker.sqlite3")


def add_version(factory, version: int) -> str:
    version_id = f"doc-1-v{version}"
    with factory.begin() as session:
        if session.get(DocumentRecord, "doc-1") is None:
            session.add(
                DocumentRecord(id="doc-1", content_hash="hash", metadata_json={})
            )
        session.add(
            DocumentVersionRecord(
                id=version_id,
                document_id="doc-1",
                version=version,
                path=f"raw/{version_id}.txt",
                sha256=f"hash-{version}",
            )
        )
    return version_id


def task_by_id(factory, task_id: str) -> TaskRecord:
    with factory() as session:
        return session.get(TaskRecord, task_id)


class ActivatingBuilder:
    def __init__(self, factory):
        self.repository = IndexRepository(factory)
        self.fail = False
        self.before_activation = None
        self.build_calls = 0

    def build(self, documents, progress=None):
        self.build_calls += 1
        assert documents == {"doc-1": []}
        for stage in ("chunking", "embedding", "indexing"):
            if progress is not None:
                progress(stage)
        generation_id = str(uuid4())
        self.repository.prepare_generation(generation_id)
        if self.before_activation is not None:
            self.before_activation()
        if self.fail:
            self.repository.mark_failed(generation_id)
            raise RuntimeError("forced indexing failure")
        self.repository.activate_generation(generation_id, [], [])
        return generation_id


def test_worker_coalesces_versions_and_supersedes_only_after_activation(factory):
    old_version = add_version(factory, 1)
    new_version = add_version(factory, 2)
    queue = TaskQueue(factory)
    old = queue.enqueue("doc-1", old_version, "ingest")
    new = queue.enqueue("doc-1", new_version, "ingest")
    builder = ActivatingBuilder(factory)

    def assert_not_activated_or_superseded():
        assert builder.repository.active_generation() is None
        assert task_by_id(factory, old.id).status == "running"
        assert task_by_id(factory, old.id).payload_json == {}
        new_during_build = task_by_id(factory, new.id)
        assert new_during_build.status == "running"
        assert new_during_build.payload_json["progress"] == "indexing"
        with factory() as session:
            assert session.get(DocumentVersionRecord, new_version).status == "indexing"

    builder.before_activation = assert_not_activated_or_superseded
    worker = IngestionWorker(
        queue,
        builder,
        owner="worker-a",
        lease_seconds=30,
        document_loader=lambda selected: {"doc-1": []},
    )

    assert worker.run_once() == 1

    assert builder.build_calls == 1
    old_record = task_by_id(factory, old.id)
    new_record = task_by_id(factory, new.id)
    assert old_record.status == "succeeded"
    assert old_record.payload_json["superseded_by"] == new.id
    assert new_record.status == "succeeded"
    generation_id = new_record.payload_json["generation_id"]
    assert builder.repository.active_generation() == generation_id
    with factory() as session:
        document = session.get(DocumentRecord, "doc-1")
        assert document.status == "ready"
        assert document.current_version_id == new_version
        assert session.get(DocumentVersionRecord, new_version).status == "ready"


def test_failed_newest_build_recovers_older_work_without_superseding(factory):
    old_version = add_version(factory, 1)
    new_version = add_version(factory, 2)
    queue = TaskQueue(factory)
    old = queue.enqueue("doc-1", old_version, "ingest")
    new = queue.enqueue("doc-1", new_version, "ingest")
    builder = ActivatingBuilder(factory)
    builder.fail = True
    worker = IngestionWorker(
        queue,
        builder,
        owner="worker-a",
        lease_seconds=30,
        document_loader=lambda selected: {"doc-1": []},
    )

    assert worker.run_once() == 1
    assert builder.repository.active_generation() is None
    assert task_by_id(factory, new.id).status == "failed"
    assert task_by_id(factory, old.id).status == "running"
    assert task_by_id(factory, old.id).payload_json == {}

    with factory.begin() as session:
        old_record = session.get(TaskRecord, old.id)
        old_record.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    builder.fail = False

    assert worker.run_once() == 1
    recovered = task_by_id(factory, old.id)
    assert recovered.status == "succeeded"
    assert recovered.attempts == 2
    assert "superseded_by" not in recovered.payload_json
    assert task_by_id(factory, new.id).status == "failed"
    assert (
        builder.repository.active_generation()
        == recovered.payload_json["generation_id"]
    )
    with factory() as session:
        document = session.get(DocumentRecord, "doc-1")
        assert document.current_version_id == old_version
        assert session.get(DocumentVersionRecord, old_version).status == "ready"
        assert session.get(DocumentVersionRecord, new_version).status == "failed"


class RecordingEmbeddings:
    dimension = 2

    def __init__(self, events):
        self.events = events

    def embed_documents(self, texts):
        self.events.append("actual-embedding")
        return [[0.25, 0.75] for _ in texts]


class RecordingQdrant:
    def __init__(self, events):
        self.events = events
        self.vectors = {}

    def create(self, generation_id, dimension):
        assert dimension == 2
        self.events.append("qdrant-create")
        self.vectors[generation_id] = {}

    def upsert(self, generation_id, children, vectors):
        self.events.append("qdrant-upsert")
        self.vectors[generation_id].update(
            {child.id: vector for child, vector in zip(children, vectors, strict=True)}
        )

    def fetch_vectors(self, generation_id, child_ids):
        return {
            child_id: self.vectors[generation_id][child_id] for child_id in child_ids
        }


def make_index_builder(factory, tmp_path, events):
    return IndexBuilder(
        repository=IndexRepository(factory),
        qdrant=RecordingQdrant(events),
        bm25_root=tmp_path / "bm25",
        embeddings=RecordingEmbeddings(events),
        chunk_settings=ChunkSettings(),
        vector_space=VectorSpace("fake", "model", 2, "cosine"),
    )


def stage_chunk(builder, monkeypatch):
    parent = ParentChunk("parent", "doc-1", 0, "text", "parent-hash", {})
    child = ChildChunk("child", "parent", "doc-1", 0, "text", "child-hash", {})
    monkeypatch.setattr(builder, "_chunk_all", lambda documents: ([parent], [child]))


def test_index_builder_emits_progress_at_real_pipeline_boundaries(
    factory, tmp_path, monkeypatch
):
    events = []
    builder = make_index_builder(factory, tmp_path, events)
    stage_chunk(builder, monkeypatch)

    generation_id = builder.build(
        {"doc-1": []}, progress=lambda stage: events.append(stage)
    )

    assert events == [
        "chunking",
        "embedding",
        "actual-embedding",
        "indexing",
        "qdrant-create",
        "qdrant-upsert",
    ]
    assert builder.repository.active_generation() == generation_id


def test_index_builder_validation_failure_never_emits_ready(
    factory, tmp_path, monkeypatch
):
    events = []
    builder = make_index_builder(factory, tmp_path, events)
    stage_chunk(builder, monkeypatch)

    def fail_validation(*args):
        raise ValueError("forced validation failure")

    monkeypatch.setattr(builder, "validate", fail_validation)
    with pytest.raises(ValueError, match="forced validation failure"):
        builder.build({"doc-1": []}, progress=lambda stage: events.append(stage))

    assert events == [
        "chunking",
        "embedding",
        "actual-embedding",
        "indexing",
        "qdrant-create",
        "qdrant-upsert",
    ]
    assert "ready" not in events
    assert builder.repository.active_generation() is None
    with factory() as session:
        from sqlalchemy import select

        failed = session.scalars(
            select(IndexGenerationRecord).where(
                IndexGenerationRecord.status == "failed"
            )
        ).one()
        assert failed.status == "failed"


def test_index_builder_default_call_remains_supported(factory, tmp_path):
    events = []
    builder = make_index_builder(factory, tmp_path, events)

    generation_id = builder.build({})

    assert builder.repository.active_generation() == generation_id
    assert events == ["qdrant-create", "qdrant-upsert"]
