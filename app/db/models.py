"""Relational records for documents, parent chunks, and index generations."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    Text,
    TypeDecorator,
    text,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _UTCDateTime(TypeDecorator[datetime]):
    """Store UTC timestamps and restore awareness when SQLite returns text."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(
        self, value: datetime | None, _dialect: Dialect
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(
        self, value: datetime | None, _dialect: Dialect
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    filename: Mapped[str] = mapped_column(
        String, default="", server_default=text("''"), nullable=False
    )
    media_type: Mapped[str] = mapped_column(
        String, default="", server_default=text("''"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(16), default="ready", server_default=text("'ready'"), nullable=False
    )
    current_version_id: Mapped[str | None] = mapped_column(
        String, server_default=text("NULL"), nullable=True
    )
    error: Mapped[str | None] = mapped_column(
        Text, server_default=text("NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        _UTCDateTime(),
        default=_utc_now,
        server_default=text("'1970-01-01 00:00:00'"),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        _UTCDateTime(),
        default=_utc_now,
        onupdate=_utc_now,
        server_default=text("'1970-01-01 00:00:00'"),
        nullable=False,
    )
    deleted: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("0"), nullable=False
    )


class IndexGenerationRecord(Base):
    __tablename__ = "index_generations"
    __table_args__ = (
        CheckConstraint(
            "status IN ('building', 'active', 'retired', 'failed')",
            name="ck_index_generations_status",
        ),
        Index(
            "uq_index_generations_one_active",
            "status",
            unique=True,
            sqlite_where=text("status = 'active'"),
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    # NULL deliberately denotes an unknown legacy vector space, never compatibility.
    vector_space_fingerprint: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )


class GenerationDocumentRecord(Base):
    """A document's immutable identity and hash within one index generation."""

    __tablename__ = "generation_documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    generation_id: Mapped[str] = mapped_column(
        ForeignKey("index_generations.id"), primary_key=True
    )
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class ParentChunkRecord(Base):
    __tablename__ = "parent_chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    generation_id: Mapped[str] = mapped_column(
        ForeignKey("index_generations.id"), primary_key=True
    )
    document_id: Mapped[str] = mapped_column(String, nullable=False)
    ordinal: Mapped[int] = mapped_column(nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class ChildChunkRecord(Base):
    """Content hashes needed to reuse dense vectors from the active generation."""

    __tablename__ = "child_chunks"
    __table_args__ = (
        ForeignKeyConstraint(
            ["parent_id", "generation_id"],
            ["parent_chunks.id", "parent_chunks.generation_id"],
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    generation_id: Mapped[str] = mapped_column(
        ForeignKey("index_generations.id"), primary_key=True
    )
    parent_id: Mapped[str] = mapped_column(String, nullable=False)
    document_id: Mapped[str] = mapped_column(String, nullable=False)
    ordinal: Mapped[int] = mapped_column(nullable=False)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
