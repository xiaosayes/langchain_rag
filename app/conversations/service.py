"""Persistence operations for the single administrator's conversations."""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.conversations.schemas import MAX_TITLE_LENGTH, ConversationOut
from app.web.records import ConversationRecord, MessageRecord

DEFAULT_TITLE = "新会话"


class ConversationNotFoundError(LookupError):
    """Raised when an operation references an unknown conversation."""


class ConversationService:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    @staticmethod
    def _title(title: str | None) -> str:
        value = DEFAULT_TITLE if title is None else title
        if len(value) > MAX_TITLE_LENGTH:
            raise ValueError(f"Title must be at most {MAX_TITLE_LENGTH} characters")
        return value

    @staticmethod
    def _out(conversation: ConversationRecord) -> ConversationOut:
        return ConversationOut(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )

    @staticmethod
    def _get(session: Session, conversation_id: str) -> ConversationRecord:
        conversation = session.get(ConversationRecord, conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)
        return conversation

    def list(self) -> list[ConversationOut]:
        """List every conversation; the sole administrator has no tenant scope."""
        with self.session_factory() as session:
            conversations = session.scalars(
                select(ConversationRecord).order_by(
                    ConversationRecord.created_at,
                    ConversationRecord.id,
                )
            ).all()
            return [self._out(conversation) for conversation in conversations]

    def create(self, title: str | None = None) -> ConversationOut:
        with self.session_factory.begin() as session:
            conversation = ConversationRecord(
                id=uuid4().hex,
                title=self._title(title),
            )
            session.add(conversation)
            session.flush()
            return self._out(conversation)

    def rename(self, conversation_id: str, title: str) -> ConversationOut:
        with self.session_factory.begin() as session:
            conversation = self._get(session, conversation_id)
            conversation.title = self._title(title)
            session.flush()
            return self._out(conversation)

    def delete(self, conversation_id: str) -> None:
        """Delete through ORM relationships so messages and feedback cascade."""
        with self.session_factory.begin() as session:
            conversation = self._get(session, conversation_id)
            session.delete(conversation)

    def messages(self, conversation_id: str, limit: int = 6) -> list[MessageRecord]:
        """Return the latest N records in chronological order for chat history."""
        with self.session_factory() as session:
            self._get(session, conversation_id)
            latest_first = session.scalars(
                select(MessageRecord)
                .where(MessageRecord.conversation_id == conversation_id)
                .order_by(MessageRecord.created_at.desc(), MessageRecord.id.desc())
                .limit(limit)
            ).all()
            return list(reversed(latest_first))

    def list_messages(self, conversation_id: str) -> list[MessageRecord]:
        """Return the complete chronological history for display."""
        with self.session_factory() as session:
            self._get(session, conversation_id)
            return list(
                session.scalars(
                    select(MessageRecord)
                    .where(MessageRecord.conversation_id == conversation_id)
                    .order_by(MessageRecord.created_at, MessageRecord.id)
                ).all()
            )
