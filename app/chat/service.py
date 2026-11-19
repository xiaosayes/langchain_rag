"""Chat preparation, citation-safe event streaming, and local feedback."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import select

from app.chat.schemas import ChatEvent
from app.conversations.service import ConversationNotFoundError, ConversationService
from app.db.models import IndexGenerationRecord
from app.error_sanitization import sanitized_error_message
from app.rag.service import RAGPreparation, RAGService
from app.web.records import ConversationRecord, FeedbackRecord, MessageRecord


class NoActiveGenerationError(RuntimeError):
    """Raised when chat cannot select an active knowledge-base generation."""


class MessageNotFoundError(LookupError):
    """Raised when feedback references an unknown message."""


class ChatService:
    def __init__(
        self, rag_service: RAGService, conversation_service: ConversationService
    ) -> None:
        self.rag_service = rag_service
        self.conversation_service = conversation_service
        self.session_factory = conversation_service.session_factory

    async def prepare(
        self, question: str, conversation_id: str | None
    ) -> RAGPreparation:
        """Persist the new turn before retrieval and supply only prior history."""
        if conversation_id is None:
            conversation_id = self.conversation_service.create().id

        with self.session_factory.begin() as session:
            if session.get(ConversationRecord, conversation_id) is None:
                raise ConversationNotFoundError(conversation_id)
            # Query history before inserting the new user message. The active
            # generation is read exactly once for this preparation.
            latest_first = session.scalars(
                select(MessageRecord)
                .where(MessageRecord.conversation_id == conversation_id)
                .order_by(MessageRecord.created_at.desc(), MessageRecord.id.desc())
                .limit(6)
            ).all()
            history_records = list(reversed(latest_first))
            generation_id = session.scalars(
                select(IndexGenerationRecord.id).where(
                    IndexGenerationRecord.status == "active"
                )
            ).one_or_none()
            session.add(
                MessageRecord(
                    id=uuid4().hex,
                    conversation_id=conversation_id,
                    role="user",
                    content=question,
                )
            )

        if generation_id is None:
            raise NoActiveGenerationError(
                "No active index generation is available; ingest documents first"
            )
        history = [self._langchain_message(message) for message in history_records]
        return await self.rag_service.prepare(question, history, generation_id)

    @staticmethod
    def _langchain_message(message: MessageRecord) -> BaseMessage:
        if message.role == "user":
            return HumanMessage(content=message.content)
        if message.role == "assistant":
            return AIMessage(content=message.content)
        raise ValueError(f"Unsupported stored message role: {message.role}")

    @staticmethod
    def _citations(prepared: RAGPreparation) -> list[dict[str, Any]]:
        return [
            {
                "label": citation.label,
                "document_id": citation.document_id,
                "parent_id": citation.parent_id,
                "text": citation.text,
                "metadata": dict(citation.metadata),
            }
            for citation in prepared.citations
        ]

    def _persist_assistant(
        self,
        message_id: str,
        conversation_id: str,
        prepared: RAGPreparation,
        content: str,
        *,
        cancelled: bool = False,
        error: str | None = None,
    ) -> None:
        with self.session_factory.begin() as session:
            session.add(
                MessageRecord(
                    id=message_id,
                    conversation_id=conversation_id,
                    role="assistant",
                    content=content,
                    citations_json=self._citations(prepared),
                    degraded=prepared.degraded,
                    generation_id=prepared.generation_id,
                    cancelled=cancelled,
                    error=error,
                )
            )

    async def stream(
        self, conversation_id: str, prepared: RAGPreparation
    ) -> AsyncIterator[ChatEvent]:
        """Emit answer events and persist only a terminal or cancelled answer."""
        message_id = uuid4().hex
        answer_parts: list[str] = []
        upstream = self.rag_service.stream_answer(prepared)
        terminal = False
        try:
            yield ChatEvent(
                "start",
                {
                    "conversation_id": conversation_id,
                    "message_id": message_id,
                    "generation_id": prepared.generation_id,
                    "degraded": prepared.degraded,
                },
            )
            async for token in upstream:
                answer_parts.append(token)
                yield ChatEvent("token", {"content": token})

            citations = self._citations(prepared)
            yield ChatEvent("citations", {"citations": citations})
            self._persist_assistant(
                message_id, conversation_id, prepared, "".join(answer_parts)
            )
            terminal = True
            yield ChatEvent(
                "done", {"conversation_id": conversation_id, "message_id": message_id}
            )
        except (asyncio.CancelledError, GeneratorExit):
            if not terminal:
                self._persist_assistant(
                    message_id,
                    conversation_id,
                    prepared,
                    "".join(answer_parts),
                    cancelled=True,
                )
            raise
        except Exception as error:  # noqa: BLE001 - sanitize every provider failure
            message = sanitized_error_message(error)
            self._persist_assistant(
                message_id,
                conversation_id,
                prepared,
                "".join(answer_parts),
                error=message,
            )
            terminal = True
            yield ChatEvent(
                "error", {"message_id": message_id, "message": message}
            )
        finally:
            await upstream.aclose()

    def upsert_feedback(
        self, message_id: str, score: int, comment: str | None
    ) -> FeedbackRecord:
        """Store one administrator-owned rating without external run association."""
        if score not in (-1, 1):
            raise ValueError("Feedback score must be -1 or 1")
        if comment is not None and len(comment) > 2000:
            raise ValueError("Feedback comment must be at most 2000 characters")

        with self.session_factory.begin() as session:
            if session.get(MessageRecord, message_id) is None:
                raise MessageNotFoundError(message_id)
            feedback = session.scalar(
                select(FeedbackRecord).where(FeedbackRecord.message_id == message_id)
            )
            if feedback is None:
                feedback = FeedbackRecord(
                    id=uuid4().hex,
                    message_id=message_id,
                    score=score,
                    comment=comment,
                )
                session.add(feedback)
            else:
                feedback.score = score
                feedback.comment = comment
            session.flush()
            return feedback
