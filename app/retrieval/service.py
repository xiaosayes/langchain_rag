"""Generation-scoped hybrid retrieval, reranking, and parent resolution."""

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path

from langchain_core.documents import Document

from app.config import RetrievalSettings
from app.domain.models import ParentChunk, SearchHit
from app.indexing.repository import IndexRepository
from app.indexing.vector_space import VectorSpace
from app.models.bailian import BailianEmbeddings, BailianReranker
from app.retrieval.bm25 import PersistentBM25
from app.retrieval.qdrant_store import QdrantGenerationStore
from app.retrieval.rrf import reciprocal_rank_fusion

_RERANK_CANDIDATE_COUNT = 30


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[SearchHit]
    documents: list[Document]
    degraded: bool


class RetrievalUnavailable(RuntimeError):
    """Raised when neither hybrid retrieval store can provide results."""


def successful_result_sets(
    *,
    dense: list[SearchHit] | BaseException,
    bm25: list[SearchHit] | BaseException,
) -> tuple[list[list[SearchHit]], list[str]]:
    """Keep successful stores and retain failed stage/error categories."""
    result_sets: list[list[SearchHit]] = []
    errors: list[str] = []
    for stage, result in (("dense", dense), ("bm25", bm25)):
        if isinstance(result, Exception):
            errors.append(f"{stage}: {type(result).__name__}: {result}")
        elif isinstance(result, BaseException):
            raise result
        else:
            result_sets.append(result)
    return result_sets, errors


def select_parent_documents(
    generation_id: str,
    parents: list[ParentChunk],
    settings: RetrievalSettings,
) -> list[Document]:
    """Select unique parents in ranked order within all configured budgets."""
    documents: list[Document] = []
    seen_parent_ids: set[str] = set()
    document_counts: dict[str, int] = {}
    context_characters = 0

    for parent in parents:
        if len(documents) >= settings.final_parent_count:
            break
        if parent.id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent.id)
        if (
            document_counts.get(parent.document_id, 0)
            >= settings.max_parents_per_document
        ):
            continue
        if context_characters + len(parent.text) > settings.max_context_characters:
            continue

        documents.append(
            Document(
                page_content=parent.text,
                metadata={
                    **parent.metadata,
                    "generation_id": generation_id,
                    "document_id": parent.document_id,
                    "parent_id": parent.id,
                },
            )
        )
        document_counts[parent.document_id] = (
            document_counts.get(parent.document_id, 0) + 1
        )
        context_characters += len(parent.text)

    return documents


class RetrievalService:
    def __init__(
        self,
        repository: IndexRepository,
        qdrant: QdrantGenerationStore,
        bm25_root: str | Path,
        embeddings: BailianEmbeddings,
        reranker: BailianReranker,
        retrieval_settings: RetrievalSettings,
        vector_space: VectorSpace,
        rerank_candidate_count: int = _RERANK_CANDIDATE_COUNT,
    ) -> None:
        self.repository = repository
        self.qdrant = qdrant
        self.bm25_root = Path(bm25_root)
        self.embeddings = embeddings
        self.reranker = reranker
        self.settings = retrieval_settings
        self.vector_space = vector_space
        self.rerank_candidate_count = rerank_candidate_count

    def load_bm25(self, generation_id: str) -> PersistentBM25:
        return PersistentBM25.load(self.bm25_root / generation_id)

    def _search_bm25(self, generation_id: str, query: str) -> list[SearchHit]:
        return self.load_bm25(generation_id).search(query, self.settings.bm25_top_k)

    async def _search_dense(self, generation_id: str, query: str) -> list[SearchHit]:
        vector = await self.embeddings.aembed_query(query)
        return await asyncio.to_thread(
            self.qdrant.search,
            generation_id,
            vector,
            self.settings.dense_top_k,
        )

    async def _rerank_or_original(
        self, query: str, candidates: list[SearchHit]
    ) -> tuple[list[SearchHit], bool]:
        if not candidates:
            return [], False
        try:
            results = await self.reranker.arerank(
                query,
                [hit.text for hit in candidates],
                len(candidates),
            )
            ranked = [
                replace(candidates[result.index], score=result.relevance_score)
                for result in results
            ]
        except Exception:  # noqa: BLE001 - every rerank failure must degrade.
            return candidates, True
        return ranked, False

    async def asearch(self, query: str, generation_id: str) -> RetrievalResult:
        self.repository.require_compatible_vector_space(
            generation_id, self.vector_space
        )
        dense_result, bm25_result = await asyncio.gather(
            self._search_dense(generation_id, query),
            asyncio.to_thread(self._search_bm25, generation_id, query),
            return_exceptions=True,
        )
        result_sets, errors = successful_result_sets(
            dense=dense_result, bm25=bm25_result
        )
        if not result_sets:
            raise RetrievalUnavailable("; ".join(errors))

        fused = reciprocal_rank_fusion(result_sets, self.settings.rrf_k)
        ranked, rerank_failed = await self._rerank_or_original(
            query, fused[: self.rerank_candidate_count]
        )
        parents = self.repository.parents(
            generation_id, [hit.parent_id for hit in ranked]
        )
        documents = select_parent_documents(generation_id, parents, self.settings)
        return RetrievalResult(
            hits=ranked,
            documents=documents,
            degraded=bool(errors) or rerank_failed,
        )
