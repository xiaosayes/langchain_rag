import pytest
from qdrant_client import QdrantClient, models

from app.config import QdrantSettings
from app.retrieval.qdrant_store import QdrantGenerationStore
from tests.integration.conftest import three_test_chunks

pytestmark = pytest.mark.integration


def test_qdrant_generation_round_trip(qdrant_url):
    store = QdrantGenerationStore(QdrantSettings(url=qdrant_url))
    store.create("gen-1", dimension=3)
    store.upsert(
        "gen-1",
        three_test_chunks(),
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )

    hits = store.search("gen-1", [0.9, 0.1, 0], top_k=1)

    assert hits[0].child_id == "child-a"
    assert hits[0].source == "dense"
    assert hits[0].parent_id == "parent-a"
    assert hits[0].document_id == "document-a"
    assert hits[0].text == "alpha document"
    assert hits[0].metadata == {"ordinal": 0}
    assert store.fetch_vectors("gen-1", ["child-a"])["child-a"] == [1, 0, 0]

    store.upsert("gen-1", three_test_chunks()[:1], [[1, 0, 0]])
    client = QdrantClient(url=qdrant_url)
    collection_name = store.collection_name("gen-1")
    assert client.count(collection_name, exact=True).count == 3

    collection = client.get_collection(collection_name)
    vectors = collection.config.params.vectors
    assert isinstance(vectors, models.VectorParams)
    assert vectors.size == 3
    assert vectors.distance == models.Distance.COSINE
    assert collection.payload_schema["document_id"].data_type == "keyword"
    assert collection.payload_schema["parent_id"].data_type == "keyword"

    store.create("gen-2", dimension=3)
    store.delete("gen-1")
    remaining_collections = {
        item.name for item in client.get_collections().collections
    }
    assert collection_name not in remaining_collections
    assert store.collection_name("gen-2") in remaining_collections
    store.delete("gen-2")
    client.close()
