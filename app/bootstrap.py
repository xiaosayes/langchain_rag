"""Production composition root for indexing, retrieval, and grounded RAG."""

import hashlib
import os
from pathlib import Path
from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import BaseMessage

from app.config import AppSettings
from app.db.session import create_session_factory
from app.indexing.builder import IndexBuilder
from app.indexing.repository import IndexRepository
from app.indexing.vector_space import VectorSpace
from app.models.bailian import (
    BailianEmbeddings,
    BailianReranker,
    RerankResult,
    create_chat_model,
)
from app.rag.service import INSUFFICIENT_CONTEXT_REFUSAL, RAGService
from app.retrieval.qdrant_store import QdrantGenerationStore
from app.retrieval.service import RetrievalService


def state_directory() -> Path:
    """Return the mutable SQLite/BM25 root, independently of source documents."""
    return Path(os.environ.get("RAG_STATE_DIR", "var/core"))


def _repository() -> IndexRepository:
    root = state_directory()
    root.mkdir(parents=True, exist_ok=True)
    return IndexRepository(create_session_factory(root / "index.sqlite3"))


def _qdrant(settings: AppSettings) -> QdrantGenerationStore:
    return QdrantGenerationStore(settings.qdrant)


def _validate_provider(settings: AppSettings) -> None:
    if settings.model_provider == "fake" and settings.app_env != "test":
        raise ValueError("MODEL_PROVIDER=fake is allowed only when APP_ENV=test")


class FakeEmbeddings(Embeddings):
    """Deterministic local embeddings available only to explicit test settings."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        blocks: list[bytes] = []
        for counter in range((self.dimension + 31) // 32):
            blocks.append(hashlib.sha256(f"{counter}:{text}".encode()).digest())
        digest = b"".join(blocks)
        return [byte / 127.5 - 1 for byte in digest[: self.dimension]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeReranker:
    """Preserve hybrid retrieval order while exercising the reranker boundary."""

    def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[RerankResult]:
        return [
            RerankResult(index=index, relevance_score=1.0)
            for index in range(min(len(documents), top_n))
        ]

    async def arerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[RerankResult]:
        return self.rerank(query, documents, top_n)


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


class FakeChatModel(SimpleChatModel):
    """Return the first supplied source, with its real label, in test mode."""

    @property
    def _llm_type(self) -> str:
        return "grounded-test-double"

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        del stop, run_manager, kwargs
        prompt = _message_text(messages[-1])
        source_marker = "知识库来源：\n"
        if source_marker not in prompt:
            return prompt
        sources = prompt.split(source_marker, 1)[1]
        if not sources.startswith("[S1]\n"):
            return INSUFFICIENT_CONTEXT_REFUSAL
        first_source = sources.removeprefix("[S1]\n").split("\n\n[S", 1)[0]
        return f"{first_source}[S1]"


def _model_adapters(
    settings: AppSettings,
) -> tuple[Embeddings, BailianReranker | FakeReranker]:
    _validate_provider(settings)
    if settings.model_provider == "fake":
        return FakeEmbeddings(settings.embedding.dimension), FakeReranker()
    # Both constructors validate credentials before any database/index mutation.
    return BailianEmbeddings(settings), BailianReranker(settings)


def _chat_model(settings: AppSettings) -> BaseChatModel:
    _validate_provider(settings)
    if settings.model_provider == "fake":
        return FakeChatModel()
    return create_chat_model(settings)


def build_index_builder(settings: AppSettings) -> IndexBuilder:
    """Build the staged generation pipeline after validating model configuration."""
    embeddings, _ = _model_adapters(settings)
    repository = _repository()
    return IndexBuilder(
        repository=repository,
        qdrant=_qdrant(settings),
        bm25_root=state_directory() / "bm25",
        embeddings=cast(Any, embeddings),
        chunk_settings=settings.chunk,
        vector_space=VectorSpace.from_settings(settings),
    )


def build_rag_service(settings: AppSettings) -> RAGService:
    """Compose the complete active-generation grounded RAG service."""
    embeddings, reranker = _model_adapters(settings)
    chat_model = _chat_model(settings)
    repository = _repository()
    retrieval = RetrievalService(
        repository=repository,
        qdrant=_qdrant(settings),
        bm25_root=state_directory() / "bm25",
        embeddings=cast(BailianEmbeddings, embeddings),
        reranker=cast(BailianReranker, reranker),
        retrieval_settings=settings.retrieval,
        vector_space=VectorSpace.from_settings(settings),
        rerank_candidate_count=settings.rerank.candidate_count,
    )
    return RAGService(chat_model, retrieval, repository)
