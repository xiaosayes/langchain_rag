"""Generation-scoped dense vector storage backed by Qdrant."""

from collections.abc import Sequence
from typing import Any, cast
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from app.config import QdrantSettings
from app.domain.models import ChildChunk, SearchHit

_UPSERT_BATCH_SIZE = 128
_QDRANT_DISTANCES = {
    "cosine": models.Distance.COSINE,
    "euclid": models.Distance.EUCLID,
    "dot": models.Distance.DOT,
}


class QdrantGenerationStore:
    def __init__(self, settings: QdrantSettings) -> None:
        self._settings = settings
        self.client = QdrantClient(url=settings.url)

    def collection_name(self, generation_id: str) -> str:
        return f"{self._settings.collection_prefix}_{generation_id}"

    @staticmethod
    def _point_id(child_id: str) -> str:
        return str(uuid5(NAMESPACE_URL, child_id))

    def create(self, generation_id: str, dimension: int) -> None:
        collection = self.collection_name(generation_id)
        self.client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=_QDRANT_DISTANCES[self._settings.distance],
            ),
        )
        self.client.create_payload_index(
            collection_name=collection,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=collection,
            field_name="parent_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    def upsert(
        self,
        generation_id: str,
        chunks: Sequence[ChildChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each child chunk must have exactly one vector")
        collection = self.collection_name(generation_id)
        for start in range(0, len(chunks), _UPSERT_BATCH_SIZE):
            batch_chunks = chunks[start : start + _UPSERT_BATCH_SIZE]
            batch_vectors = vectors[start : start + _UPSERT_BATCH_SIZE]
            points = [
                models.PointStruct(
                    id=self._point_id(chunk.id),
                    vector=list(vector),
                    payload={
                        "child_id": chunk.id,
                        "parent_id": chunk.parent_id,
                        "document_id": chunk.document_id,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                    },
                )
                for chunk, vector in zip(batch_chunks, batch_vectors, strict=True)
            ]
            self.client.upsert(
                collection_name=collection,
                points=points,
                wait=True,
            )

    def search(
        self, generation_id: str, vector: Sequence[float], top_k: int
    ) -> list[SearchHit]:
        if top_k <= 0:
            return []
        response = self.client.query_points(
            collection_name=self.collection_name(generation_id),
            query=list(vector),
            limit=top_k,
            with_payload=True,
        )
        return [self._search_hit(point) for point in response.points]

    @staticmethod
    def _search_hit(point: models.ScoredPoint) -> SearchHit:
        payload: dict[str, Any] = point.payload or {}
        return SearchHit(
            child_id=payload["child_id"],
            parent_id=payload["parent_id"],
            document_id=payload["document_id"],
            text=payload["text"],
            score=point.score,
            source="dense",
            metadata=payload.get("metadata", {}),
        )

    def fetch_vectors(
        self, generation_id: str, child_ids: Sequence[str]
    ) -> dict[str, list[float]]:
        if not child_ids:
            return {}
        records = self.client.retrieve(
            collection_name=self.collection_name(generation_id),
            ids=[self._point_id(child_id) for child_id in child_ids],
            with_payload=["child_id"],
            with_vectors=True,
        )
        vectors: dict[str, list[float]] = {}
        for record in records:
            vector = record.vector
            if not isinstance(vector, list) or any(
                isinstance(component, list) for component in vector
            ):
                raise TypeError("Expected an unnamed dense vector from Qdrant")
            payload = record.payload or {}
            vectors[payload["child_id"]] = cast(list[float], vector)
        return vectors

    def child_ids(self, generation_id: str) -> frozenset[str]:
        """Read every persisted payload ID for exact generation verification."""
        child_ids: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name(generation_id),
                limit=256,
                offset=offset,
                with_payload=["child_id"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                child_id = payload.get("child_id")
                if isinstance(child_id, str):
                    child_ids.add(child_id)
            if offset is None:
                return frozenset(child_ids)

    def delete(self, generation_id: str) -> None:
        self.client.delete_collection(self.collection_name(generation_id))
