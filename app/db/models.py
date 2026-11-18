"""Relational records for documents, parent chunks, and index generations."""

from typing import Any

from sqlalchemy import JSON, CheckConstraint, ForeignKey, String, Text
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
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)


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
