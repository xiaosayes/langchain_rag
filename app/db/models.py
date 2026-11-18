"""Relational records for documents, parent chunks, and index generations."""

from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


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
