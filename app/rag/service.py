"""Grounded RAG preparation and citation-safe answer streaming."""

import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from app.indexing.repository import IndexRepository
from app.rag.prompts import build_grounded_messages, build_rewrite_messages
from app.retrieval.service import RetrievalResult, RetrievalService

INSUFFICIENT_CONTEXT_REFUSAL = "根据当前知识库，无法可靠回答该问题。"

_CITATION_PATTERN = re.compile(r"\[S\d+\]")
_INCOMPLETE_CITATION_PATTERN = re.compile(r"\[S\d*$")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Citation:
    label: str
    document_id: str
    parent_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RAGPreparation:
    original_question: str
    retrieval_query: str
    documents: list[Document]
    citations: list[Citation]
    degraded: bool
    generation_id: str
    insufficient_context: bool


def make_preparation(
    original_question: str,
    retrieval_query: str,
    result: RetrievalResult,
    generation_id: str,
) -> RAGPreparation:
    """Create citations exclusively from the parents supplied by retrieval."""
    citations = [
        Citation(
            f"S{index}",
            document.metadata["document_id"],
            document.metadata["parent_id"],
            document.page_content,
            document.metadata,
        )
        for index, document in enumerate(result.documents, start=1)
    ]
    return RAGPreparation(
        original_question,
        retrieval_query,
        result.documents,
        citations,
        result.degraded,
        generation_id,
        insufficient_context=not result.documents,
    )


def _content_text(content: str | list[str | dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif block.get("type") == "text":
            text = block.get("text", "")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


class _CitationStreamFilter:
    """Remove invalid citations even when their markers span model chunks."""

    def __init__(self, valid_labels: set[str], generation_id: str) -> None:
        self._valid_labels = valid_labels
        self._generation_id = generation_id
        self._pending = ""

    def push(self, text: str) -> str:
        buffer = self._pending + text
        self._pending = ""
        output: list[str] = []
        position = 0

        for match in _CITATION_PATTERN.finditer(buffer):
            output.append(buffer[position : match.start()])
            label = match.group()[1:-1]
            if label in self._valid_labels:
                output.append(match.group())
            else:
                logger.warning(
                    "Removed unknown citation label %s from generated answer "
                    "for generation %s",
                    label,
                    self._generation_id,
                )
            position = match.end()

        tail = buffer[position:]
        incomplete = _INCOMPLETE_CITATION_PATTERN.search(tail)
        if tail.endswith("["):
            incomplete_start = len(tail) - 1
        elif incomplete is not None:
            incomplete_start = incomplete.start()
        else:
            output.append(tail)
            return "".join(output)

        output.append(tail[:incomplete_start])
        self._pending = tail[incomplete_start:]
        return "".join(output)

    def finish(self) -> str:
        pending = self._pending
        self._pending = ""
        if pending == "[":
            # A lone terminal bracket is literal text, not a citation prefix.
            return "["
        if pending:
            logger.warning(
                "Removed incomplete citation %s from generated answer for generation %s",
                pending,
                self._generation_id,
            )
        return ""


class RAGService:
    def __init__(
        self,
        chat_model: BaseChatModel,
        retrieval_service: RetrievalService,
        repository: IndexRepository,
    ) -> None:
        self.chat_model = chat_model
        self.retrieval_service = retrieval_service
        self.repository = repository

    async def prepare(
        self, question: str, history: list[BaseMessage], generation_id: str
    ) -> RAGPreparation:
        retrieval_query = question
        if history:
            rewritten = await self.chat_model.ainvoke(
                build_rewrite_messages(question, history)
            )
            rewritten_text = _content_text(rewritten.content).strip()
            if rewritten_text:
                retrieval_query = rewritten_text

        result = await self.retrieval_service.asearch(retrieval_query, generation_id)
        return make_preparation(question, retrieval_query, result, generation_id)

    async def stream_answer(self, prepared: RAGPreparation) -> AsyncIterator[str]:
        if prepared.insufficient_context:
            yield INSUFFICIENT_CONTEXT_REFUSAL
            return

        citation_filter = _CitationStreamFilter(
            {citation.label for citation in prepared.citations}, prepared.generation_id
        )
        messages = build_grounded_messages(
            prepared.original_question, prepared.documents
        )
        async for chunk in self.chat_model.astream(messages):
            text = citation_filter.push(_content_text(chunk.content))
            if text:
                yield text

        remaining = citation_filter.finish()
        if remaining:
            yield remaining
