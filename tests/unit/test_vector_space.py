from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import text

from app.config import AppSettings
from app.db.models import IndexGenerationRecord
from app.db.session import create_session_factory
from app.indexing.repository import IndexRepository
from app.indexing.vector_space import VectorSpace
from tests.unit.test_index_repository import (
    activate,
    make_child,
    make_parent,
    repository,
)


def test_vector_space_fingerprint_covers_every_compatibility_field():
    settings = AppSettings()
    space = VectorSpace.from_settings(settings)

    assert space.provider == "bailian"
    assert space.model == settings.embedding.model
    assert space.dimension == 1024
    assert space.distance == "cosine"
    assert len(space.fingerprint) == 64
    assert space.fingerprint != replace(space, model="other").fingerprint
    assert space.fingerprint != replace(space, provider="fake").fingerprint
    assert space.fingerprint != replace(space, dimension=512).fingerprint
    assert space.fingerprint != replace(space, distance="dot").fingerprint


def test_legacy_generation_is_unknown_and_never_reused(tmp_path: Path):
    index_repository = repository(tmp_path)
    child = make_child()
    activate(index_repository, "legacy")
    space = VectorSpace("fake", "model", 8, "cosine")

    assert index_repository.vector_space_fingerprint("legacy") is None
    assert index_repository.reusable_vectors([child], space) == set()
    with pytest.raises(RuntimeError, match="rebuild"):
        index_repository.require_compatible_vector_space("legacy", space)


def test_only_matching_active_vector_space_can_reuse_vectors(tmp_path: Path):
    index_repository = repository(tmp_path)
    child = make_child()
    space = VectorSpace("fake", "model", 8, "cosine")
    index_repository.prepare_generation("current", space)
    index_repository.activate_generation("current", [make_parent()], [child])

    assert index_repository.reusable_vectors([child], space) == {child.id}
    assert index_repository.reusable_vectors(
        [child], replace(space, model="different")
    ) == set()
    index_repository.require_compatible_vector_space("current", space)
    with pytest.raises(RuntimeError, match="rebuild"):
        index_repository.require_compatible_vector_space(
            "current", replace(space, distance="dot")
        )


def test_existing_sqlite_schema_gains_nullable_fingerprint(tmp_path: Path):
    database = tmp_path / "legacy.sqlite3"
    legacy_factory = create_session_factory(database)
    with legacy_factory.begin() as session:
        session.add(IndexGenerationRecord(id="legacy", status="active"))

    # Simulate the pre-fingerprint schema, then reopen through lightweight migration.
    with legacy_factory.begin() as session:
        session.execute(
            text("ALTER TABLE index_generations DROP COLUMN vector_space_fingerprint")
        )

    reopened = create_session_factory(database)
    index_repository = IndexRepository(reopened)
    assert index_repository.vector_space_fingerprint("legacy") is None
