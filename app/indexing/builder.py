"""Staged index builds with vector reuse and atomic generation activation."""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from app.config import ChunkSettings
from app.domain.models import ChildChunk, ParentChunk, ParsedBlock
from app.indexing.repository import IndexRepository
from app.ingestion.chunker import chunk_blocks
from app.retrieval.bm25 import PersistentBM25
from app.retrieval.qdrant_store import QdrantGenerationStore


class EmbeddingModel(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class IndexBuilder:
    def __init__(
        self,
        repository: IndexRepository,
        qdrant: QdrantGenerationStore,
        bm25_root: str | Path,
        embeddings: EmbeddingModel,
        chunk_settings: ChunkSettings,
    ) -> None:
        self.repository = repository
        self.qdrant = qdrant
        self.bm25_root = Path(bm25_root)
        self.embeddings = embeddings
        self.chunk_settings = chunk_settings

    def build(self, documents: dict[str, list[ParsedBlock]]) -> str:
        generation_id = str(uuid4())
        self.repository.prepare_generation(generation_id)
        try:
            parents, children = self._chunk_all(documents)
            reusable = self.repository.reusable_vectors(children)
            previous_generation = self.repository.active_generation()
            vectors = (
                self.qdrant.fetch_vectors(previous_generation, sorted(reusable))
                if previous_generation is not None and reusable
                else {}
            )
            missing = [child for child in children if child.id not in vectors]
            vectors.update(self.embed_new(missing))

            self.qdrant.create(generation_id, self.embeddings.dimension)
            self.qdrant.upsert(
                generation_id,
                children,
                [vectors[child.id] for child in children],
            )
            PersistentBM25.build(children, self.bm25_root / generation_id)
            self.validate(generation_id, parents, children, vectors)
            self.repository.activate_generation(generation_id, parents, children)
            return generation_id
        except Exception:
            self.repository.mark_failed(generation_id)
            raise

    def embed_new(self, children: Sequence[ChildChunk]) -> dict[str, list[float]]:
        if not children:
            return {}
        embedded = self.embeddings.embed_documents([child.text for child in children])
        if len(embedded) != len(children):
            raise ValueError("Embedding count does not match new child count")
        return {
            child.id: vector for child, vector in zip(children, embedded, strict=True)
        }

    def validate(
        self,
        generation_id: str,
        parents: Sequence[ParentChunk],
        children: Sequence[ChildChunk],
        vectors: dict[str, list[float]],
    ) -> None:
        parent_by_id = {parent.id: parent for parent in parents}
        child_ids = {child.id for child in children}
        if len(parent_by_id) != len(parents) or len(child_ids) != len(children):
            raise ValueError("Parent and child IDs must be unique within a generation")
        if set(vectors) != child_ids:
            raise ValueError("Every child must have exactly one staged vector")
        for child in children:
            parent = parent_by_id.get(child.parent_id)
            if parent is None or parent.document_id != child.document_id:
                raise ValueError(f"Child {child.id} references an invalid parent")
            vector = vectors[child.id]
            if len(vector) != self.embeddings.dimension or not all(
                math.isfinite(component) for component in vector
            ):
                raise ValueError(f"Child {child.id} has an invalid vector")

        stored_vectors = self.qdrant.fetch_vectors(generation_id, sorted(child_ids))
        if set(stored_vectors) != child_ids or any(
            len(vector) != self.embeddings.dimension
            for vector in stored_vectors.values()
        ):
            raise ValueError("Qdrant generation count or dimension is invalid")

        bm25 = PersistentBM25.load(self.bm25_root / generation_id)
        if children:
            hits = bm25.search(children[0].text, top_k=len(children))
            if {hit.child_id for hit in hits} != child_ids:
                raise ValueError("BM25 generation chunk count is invalid")

    def _chunk_all(
        self, documents: dict[str, list[ParsedBlock]]
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        for document_id, blocks in documents.items():
            document_parents, document_children = chunk_blocks(
                document_id, blocks, self.chunk_settings
            )
            parents.extend(document_parents)
            children.extend(document_children)
        return parents, children
