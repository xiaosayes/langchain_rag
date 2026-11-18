import json

import pytest
from sqlalchemy import select

from app.db.models import IndexGenerationRecord
from app.retrieval.bm25 import PersistentBM25
from tests.integration.conftest import parsed_blocks

pytestmark = pytest.mark.integration


def test_first_generation_activates_complete_searchable_artifacts(
    index_builder, fake_embeddings
):
    documents = {
        "doc-a": parsed_blocks("doc-a", "差旅标准为500元"),
        "doc-b": parsed_blocks("doc-b", "年假为10天"),
    }

    generation_id = index_builder.build(documents)

    assert index_builder.repository.active_generation() == generation_id
    children_hits = index_builder.qdrant.search(
        generation_id,
        fake_embeddings.embed_documents(["差旅标准为500元"])[0],
        top_k=2,
    )
    assert {hit.text for hit in children_hits} == {
        "差旅标准为500元",
        "年假为10天",
    }
    bm25 = PersistentBM25.load(index_builder.bm25_root / generation_id)
    assert {hit.text for hit in bm25.search("差旅", top_k=2)} == {
        "差旅标准为500元",
        "年假为10天",
    }
    manifest = json.loads(
        (index_builder.bm25_root / generation_id / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["chunk_count"] == 2


def test_second_generation_reuses_unchanged_vectors(index_builder, fake_embeddings):
    documents_v1 = {
        "doc-a": parsed_blocks("doc-a", "差旅标准为500元"),
        "doc-b": parsed_blocks("doc-b", "年假为10天"),
    }
    first = index_builder.build(documents_v1)
    first_parents = index_builder.repository.parents(
        first, [parent.id for parent in _parents(index_builder, first)]
    )
    first_collection = index_builder.qdrant.collection_name(first)
    fake_embeddings.embedded_texts.clear()

    documents_v2 = dict(documents_v1, **{"doc-b": parsed_blocks("doc-b", "年假为15天")})
    second = index_builder.build(documents_v2)

    assert second != first
    assert index_builder.repository.active_generation() == second
    assert "差旅标准为500元" not in fake_embeddings.embedded_texts
    assert "年假为10天" not in fake_embeddings.embedded_texts
    assert fake_embeddings.embedded_texts == ["年假为15天"]

    old_child_id = _child_id(index_builder, first, "差旅标准为500元")
    new_child_id = _child_id(index_builder, second, "差旅标准为500元")
    assert old_child_id == new_child_id
    assert index_builder.qdrant.fetch_vectors(first, [old_child_id]) == (
        index_builder.qdrant.fetch_vectors(second, [new_child_id])
    )
    assert (
        index_builder.repository.parents(first, [parent.id for parent in first_parents])
        == first_parents
    )
    assert (index_builder.bm25_root / first / "manifest.json").is_file()
    assert index_builder.qdrant.client.get_collection(first_collection)

    fake_embeddings.embedded_texts.clear()
    third = index_builder.build(documents_v2)
    assert fake_embeddings.embedded_texts == []
    assert {
        hit.text
        for hit in index_builder.qdrant.search(
            third,
            fake_embeddings.embed_query("年假为15天"),
            top_k=2,
        )
    } == {"差旅标准为500元", "年假为15天"}


def test_bm25_validation_failure_preserves_active_generation_and_artifacts(
    index_builder, fake_embeddings, monkeypatch
):
    first = index_builder.build({"doc-a": parsed_blocks("doc-a", "差旅标准为500元")})
    fake_embeddings.embedded_texts.clear()

    def fail_validation(_generation_dir):
        raise ValueError("forced BM25 validation failure")

    monkeypatch.setattr(PersistentBM25, "load", fail_validation)
    with pytest.raises(ValueError, match="forced BM25 validation failure"):
        index_builder.build({"doc-a": parsed_blocks("doc-a", "差旅标准为600元")})

    assert index_builder.repository.active_generation() == first
    assert (
        index_builder.repository.parents(first, [_parents(index_builder, first)[0].id])[
            0
        ].text
        == "差旅标准为500元"
    )
    assert (index_builder.bm25_root / first / "manifest.json").is_file()

    with index_builder.repository.session_factory() as session:
        failed = session.scalars(
            select(IndexGenerationRecord).where(
                IndexGenerationRecord.status == "failed"
            )
        ).one()
        failed_id = failed.id
        assert session.get(IndexGenerationRecord, first).status == "active"

    assert failed_id != first
    assert (index_builder.bm25_root / failed_id / "manifest.json").is_file()
    failed_hit = index_builder.qdrant.search(
        failed_id,
        fake_embeddings.embed_query("差旅标准为600元"),
        top_k=1,
    )[0]
    assert failed_hit.text == "差旅标准为600元"
    assert index_builder.qdrant.fetch_vectors(failed_id, [failed_hit.child_id])


def _parents(index_builder, generation_id: str):
    with index_builder.repository.session_factory() as session:
        from app.db.models import ParentChunkRecord

        records = session.scalars(
            select(ParentChunkRecord)
            .where(ParentChunkRecord.generation_id == generation_id)
            .order_by(ParentChunkRecord.document_id, ParentChunkRecord.ordinal)
        ).all()
        return [
            index_builder.repository.parents(generation_id, [record.id])[0]
            for record in records
        ]


def _child_id(index_builder, generation_id: str, text: str) -> str:
    bm25 = PersistentBM25.load(index_builder.bm25_root / generation_id)
    return next(hit.child_id for hit in bm25.search(text, top_k=10) if hit.text == text)
