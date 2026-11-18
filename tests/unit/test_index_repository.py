from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db.models import (
    GenerationDocumentRecord,
    IndexGenerationRecord,
    ParentChunkRecord,
)
from app.db.session import create_session_factory
from app.domain.models import ChildChunk, ParentChunk
from app.indexing.repository import GenerationStateError, IndexRepository


def make_parent(
    parent_id: str = "parent-1",
    *,
    document_id: str = "doc-1",
    text: str = "差旅标准为500元",
    content_hash: str = "parent-hash",
) -> ParentChunk:
    return ParentChunk(
        id=parent_id,
        document_id=document_id,
        ordinal=0,
        text=text,
        content_hash=content_hash,
        metadata={"source": "policy.txt"},
    )


def make_child(
    child_id: str = "child-1",
    *,
    parent_id: str = "parent-1",
    document_id: str = "doc-1",
    content_hash: str = "child-hash",
) -> ChildChunk:
    return ChildChunk(
        id=child_id,
        parent_id=parent_id,
        document_id=document_id,
        ordinal=0,
        text="差旅标准为500元",
        content_hash=content_hash,
        metadata={},
    )


def repository(tmp_path: Path) -> IndexRepository:
    return IndexRepository(create_session_factory(tmp_path / "rag.sqlite3"))


def activate(
    index_repository: IndexRepository,
    generation_id: str,
    *,
    parent: ParentChunk | None = None,
    children: list[ChildChunk] | None = None,
) -> None:
    index_repository.prepare_generation(generation_id)
    index_repository.activate_generation(
        generation_id,
        [make_parent() if parent is None else parent],
        [make_child()] if children is None else children,
    )


def test_first_activation_and_explicit_generation_parent_queries(tmp_path: Path):
    index_repository = repository(tmp_path)
    assert index_repository.active_generation() is None

    activate(index_repository, "gen-1")
    first_parent = make_parent(text="旧标准")
    index_repository.prepare_generation("gen-2")
    index_repository.activate_generation(
        "gen-2", [first_parent], [make_child(content_hash="new-hash")]
    )

    assert index_repository.active_generation() == "gen-2"
    assert index_repository.parents("gen-1", ["parent-1"]) == [make_parent()]
    assert index_repository.parents("gen-2", ["parent-1"]) == [first_parent]
    assert index_repository.parents("gen-1", ["missing"]) == []


def test_parent_results_follow_requested_id_order(tmp_path: Path):
    index_repository = repository(tmp_path)
    parents = [
        make_parent("parent-a", text="A", content_hash="hash-a"),
        make_parent("parent-b", text="B", content_hash="hash-b"),
    ]
    index_repository.prepare_generation("gen-1")
    index_repository.activate_generation("gen-1", parents)

    assert index_repository.parents("gen-1", ["parent-b", "parent-a", "parent-b"]) == [
        parents[1],
        parents[0],
        parents[1],
    ]


def test_reusable_vectors_require_matching_id_hash_and_active_generation(
    tmp_path: Path,
):
    index_repository = repository(tmp_path)
    unchanged = make_child("same", content_hash="same-hash")
    changed_hash = make_child("changed", content_hash="new-hash")
    unknown = make_child("new", content_hash="new-hash")

    activate(
        index_repository,
        "gen-1",
        children=[
            unchanged,
            make_child("changed", content_hash="old-hash"),
            make_child("retired-only", content_hash="retired-hash"),
        ],
    )
    index_repository.prepare_generation("gen-2")
    index_repository.activate_generation("gen-2", [make_parent()], [unchanged])

    assert index_repository.reusable_vectors([unchanged, changed_hash, unknown]) == {
        "same"
    }
    assert (
        index_repository.reusable_vectors(
            [make_child("retired-only", content_hash="retired-hash")]
        )
        == set()
    )


def test_document_rows_are_scoped_to_their_generation(tmp_path: Path):
    index_repository = repository(tmp_path)
    activate(index_repository, "gen-1")
    activate(index_repository, "gen-2")

    with index_repository.session_factory() as session:
        first = session.get(GenerationDocumentRecord, ("doc-1", "gen-1"))
        second = session.get(GenerationDocumentRecord, ("doc-1", "gen-2"))
        assert first is not None
        assert second is not None
        assert first.generation_id == "gen-1"
        assert second.generation_id == "gen-2"


def test_activation_retires_predecessor_before_activating_new_row(tmp_path: Path):
    index_repository = repository(tmp_path)
    activate(index_repository, "gen-z")

    activate(index_repository, "gen-a")

    assert index_repository.active_generation() == "gen-a"
    with index_repository.session_factory() as session:
        assert session.get(IndexGenerationRecord, "gen-z").status == "retired"


def test_only_one_active_generation_is_allowed(tmp_path: Path):
    index_repository = repository(tmp_path)
    with (
        pytest.raises(IntegrityError),
        index_repository.session_factory.begin() as session,
    ):
        session.add_all(
            [
                IndexGenerationRecord(id="gen-1", status="active"),
                IndexGenerationRecord(id="gen-2", status="active"),
            ]
        )


def test_activation_rejects_non_building_generation(tmp_path: Path):
    index_repository = repository(tmp_path)
    activate(index_repository, "gen-1")

    with pytest.raises(GenerationStateError, match="gen-1"):
        index_repository.activate_generation("gen-1", [make_parent()])
    with pytest.raises(GenerationStateError, match="missing"):
        index_repository.activate_generation("missing", [make_parent()])
    assert index_repository.active_generation() == "gen-1"


def test_failed_activation_rolls_back_parent_rows_and_previous_activation(
    tmp_path: Path,
):
    index_repository = repository(tmp_path)
    activate(index_repository, "gen-1")
    index_repository.prepare_generation("gen-2")

    with pytest.raises(IntegrityError):
        index_repository.activate_generation("gen-2", [make_parent(), make_parent()])

    index_repository.mark_failed("gen-2")

    assert index_repository.active_generation() == "gen-1"
    with index_repository.session_factory() as session:
        assert session.get(IndexGenerationRecord, "gen-2").status == "failed"
        assert session.get(IndexGenerationRecord, "gen-1").status == "active"
        assert (
            session.scalars(
                select(ParentChunkRecord).where(
                    ParentChunkRecord.generation_id == "gen-2"
                )
            ).all()
            == []
        )


def test_mark_failed_only_changes_the_staged_generation(tmp_path: Path):
    index_repository = repository(tmp_path)
    activate(index_repository, "gen-1")
    index_repository.prepare_generation("gen-2")

    index_repository.mark_failed("gen-2")
    index_repository.mark_failed("gen-1")

    assert index_repository.active_generation() == "gen-1"
    with index_repository.session_factory() as session:
        assert session.get(IndexGenerationRecord, "gen-2").status == "failed"
