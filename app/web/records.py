"""Single-administrator web persistence records."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.models import _utc_now, _UTCDateTime


class AdminRecord(Base):
    __tablename__ = "admins"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    username: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        _UTCDateTime(), default=_utc_now, nullable=False
    )


class AuthSessionRecord(Base):
    __tablename__ = "auth_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    csrf_token: Mapped[str] = mapped_column(String, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(_UTCDateTime(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        _UTCDateTime(), default=_utc_now, nullable=False
    )


class DocumentVersionRecord(Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint(
            "document_id", "version", name="uq_document_versions_document_version"
        ),
        CheckConstraint(
            "status IN ('uploaded', 'parsing', 'chunking', 'embedding', "
            "'indexing', 'ready', 'failed')",
            name="ck_document_versions_status",
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default="uploaded", server_default=text("'uploaded'"), nullable=False
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        _UTCDateTime(), default=_utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        _UTCDateTime(), default=_utc_now, onupdate=_utc_now, nullable=False
    )


class TaskRecord(Base):
    __tablename__ = "tasks"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed')",
            name="ck_tasks_status",
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id"), nullable=False
    )
    version_id: Mapped[str | None] = mapped_column(
        ForeignKey("document_versions.id"), nullable=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default="queued", server_default=text("'queued'"), nullable=False
    )
    attempts: Mapped[int] = mapped_column(
        Integer, default=0, server_default=text("0"), nullable=False
    )
    max_attempts: Mapped[int] = mapped_column(
        Integer, default=3, server_default=text("3"), nullable=False
    )
    payload_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, server_default=text("'{}'"), nullable=False
    )
    lease_owner: Mapped[str | None] = mapped_column(String, nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        _UTCDateTime(), nullable=True
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        _UTCDateTime(), default=_utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        _UTCDateTime(), default=_utc_now, onupdate=_utc_now, nullable=False
    )


class ConversationRecord(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(
        String(200), default="新会话", server_default=text("'新会话'"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        _UTCDateTime(), default=_utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        _UTCDateTime(), default=_utc_now, onupdate=_utc_now, nullable=False
    )

    messages: Mapped[list[MessageRecord]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class MessageRecord(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list, server_default=text("'[]'"), nullable=False
    )
    degraded: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("0"), nullable=False
    )
    generation_id: Mapped[str | None] = mapped_column(
        ForeignKey("index_generations.id"), nullable=True
    )
    cancelled: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("0"), nullable=False
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        _UTCDateTime(), default=_utc_now, nullable=False
    )

    conversation: Mapped[ConversationRecord] = relationship(
        back_populates="messages"
    )
    feedback: Mapped[FeedbackRecord | None] = relationship(
        back_populates="message", cascade="all, delete-orphan"
    )


class ChatRunRecord(Base):
    __tablename__ = "chat_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'succeeded', 'failed', 'cancelled')",
            name="ck_chat_runs_status",
        ),
        Index("ix_chat_runs_created_at", "created_at"),
        Index("ix_chat_runs_status_created_at", "status", "created_at"),
        Index("ix_chat_runs_conversation_id", "conversation_id"),
        Index("ix_chat_runs_message_id", "message_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    # These identifiers deliberately have no foreign keys: purging business data
    # must not silently remove diagnostic runs, and monitoring purge must not
    # delete conversations or messages.
    conversation_id: Mapped[str | None] = mapped_column(String, nullable=True)
    message_id: Mapped[str | None] = mapped_column(String, nullable=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_messages_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list, server_default=text("'[]'"), nullable=False
    )
    retrieved_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    generation_id: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), default="running", server_default=text("'running'"), nullable=False
    )
    degraded: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("0"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        _UTCDateTime(), default=_utc_now, nullable=False
    )
    ended_at: Mapped[datetime | None] = mapped_column(_UTCDateTime(), nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    events: Mapped[list[ChatRunEventRecord]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class ChatRunEventRecord(Base):
    __tablename__ = "chat_run_events"
    __table_args__ = (
        Index("ix_chat_run_events_run_started", "run_id", "started_at", "id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("chat_runs.id", ondelete="CASCADE"), nullable=False
    )
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    started_at: Mapped[datetime] = mapped_column(_UTCDateTime(), nullable=False)
    ended_at: Mapped[datetime] = mapped_column(_UTCDateTime(), nullable=False)
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False)
    details_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, server_default=text("'{}'"), nullable=False
    )

    run: Mapped[ChatRunRecord] = relationship(back_populates="events")


class FeedbackRecord(Base):
    __tablename__ = "feedback"
    __table_args__ = (
        CheckConstraint("score IN (-1, 1)", name="ck_feedback_score"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    comment: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        _UTCDateTime(), default=_utc_now, nullable=False
    )

    message: Mapped[MessageRecord] = relationship(back_populates="feedback")
