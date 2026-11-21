"""Generation-scoped hybrid retrieval, reranking, and parent resolution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

from app.config import RetrievalSettings
from app.domain.models import ParentChunk, SearchHit
from app.indexing.repository import IndexRepository
from app.indexing.vector_space import VectorSpace
from app.models.bailian import BailianEmbeddings, BailianReranker
from app.retrieval.bm25 import PersistentBM25
from app.retrieval.qdrant_store import QdrantGenerationStore
from app.retrieval.rrf import reciprocal_rank_fusion

if TYPE_CHECKING:
    from app.monitoring.service import MonitoringService

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
        *,
        monitoring_service: MonitoringService | None = None,
    ) -> None:
        self.repository = repository
        self.qdrant = qdrant
        self.bm25_root = Path(bm25_root)
        self.embeddings = embeddings
        self.reranker = reranker
        self.settings = retrieval_settings
        self.vector_space = vector_space
        self.rerank_candidate_count = rerank_candidate_count
        self.monitoring_service = monitoring_service

    def _stage_start(self, run_id: str | None) -> tuple[datetime, float] | None:
        if run_id is None or self.monitoring_service is None:
            return None
        return datetime.now(UTC), perf_counter()

    def _record_stage(
        self,
        run_id: str | None,
        stage: str,
        status: str,
        started: tuple[datetime, float] | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if started is None or run_id is None or self.monitoring_service is None:
            return
        started_at, started_clock = started
        try:
            self.monitoring_service.record_event(
                run_id,
                stage,
                status,
                started_at,
                max(0.0, (perf_counter() - started_clock) * 1000),
                details,
            )
        except Exception:  # noqa: BLE001, S110 - diagnostics must not affect retrieval
            pass

    @staticmethod
    def _failure_status(error: BaseException) -> str:
        return "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"

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

    async def _monitored_dense(
        self, generation_id: str, query: str, run_id: str
    ) -> list[SearchHit]:
        started = self._stage_start(run_id)
        try:
            hits = await self._search_dense(generation_id, query)
        except BaseException as error:
            self._record_stage(
                run_id,
                "dense",
                self._failure_status(error),
                started,
                {"error_type": type(error).__name__},
            )
            raise
        self._record_stage(
            run_id, "dense", "succeeded", started, {"candidates": len(hits)}
        )
        return hits

    async def _monitored_bm25(
        self, generation_id: str, query: str, run_id: str
    ) -> list[SearchHit]:
        started = self._stage_start(run_id)
        try:
            hits = await asyncio.to_thread(self._search_bm25, generation_id, query)
        except BaseException as error:
            self._record_stage(
                run_id,
                "bm25",
                self._failure_status(error),
                started,
                {"error_type": type(error).__name__},
            )
            raise
        self._record_stage(
            run_id, "bm25", "succeeded", started, {"candidates": len(hits)}
        )
        return hits

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

    async def asearch(
        self, query: str, generation_id: str, *, run_id: str | None = None
    ) -> RetrievalResult:
        self.repository.require_compatible_vector_space(
            generation_id, self.vector_space
        )
        if run_id is None or self.monitoring_service is None:
            dense_result, bm25_result = await asyncio.gather(
                self._search_dense(generation_id, query),
                asyncio.to_thread(self._search_bm25, generation_id, query),
                return_exceptions=True,
            )
        else:
            dense_result, bm25_result = await asyncio.gather(
                self._monitored_dense(generation_id, query, run_id),
                self._monitored_bm25(generation_id, query, run_id),
                return_exceptions=True,
            )
        result_sets, errors = successful_result_sets(
            dense=dense_result, bm25=bm25_result
        )
        if not result_sets:
            raise RetrievalUnavailable("; ".join(errors))

        started = self._stage_start(run_id)
        try:
            fused = reciprocal_rank_fusion(result_sets, self.settings.rrf_k)
        except BaseException as error:
            self._record_stage(
                run_id,
                "rrf",
                self._failure_status(error),
                started,
                {"error_type": type(error).__name__},
            )
            raise
        self._record_stage(
            run_id, "rrf", "succeeded", started, {"candidates": len(fused)}
        )

        started = self._stage_start(run_id)
        try:
            ranked, rerank_failed = await self._rerank_or_original(
                query, fused[: self.rerank_candidate_count]
            )
        except BaseException as error:
            self._record_stage(
                run_id,
                "rerank",
                self._failure_status(error),
                started,
                {"error_type": type(error).__name__},
            )
            raise
        self._record_stage(
            run_id,
            "rerank",
            "degraded" if rerank_failed else "succeeded",
            started,
            {"candidates": len(ranked)},
        )

        started = self._stage_start(run_id)
        try:
            parents = self.repository.parents(
                generation_id, [hit.parent_id for hit in ranked]
            )
        except BaseException as error:
            self._record_stage(
                run_id,
                "parent_recovery",
                self._failure_status(error),
                started,
                {"error_type": type(error).__name__},
            )
            raise
        self._record_stage(
            run_id,
            "parent_recovery",
            "succeeded",
            started,
            {"requested": len(ranked), "recovered": len(parents)},
        )
        documents = select_parent_documents(generation_id, parents, self.settings)
        return RetrievalResult(
            hits=ranked,
            documents=documents,
            degraded=bool(errors) or rerank_failed,
        )
