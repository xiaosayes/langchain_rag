"""Typed values shared by document storage and ingestion services."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

DocumentKind = Literal["document", "qa"]


class DocumentOut(BaseModel):
    """Public document state without storage paths or queue internals."""

    id: str
    filename: str
    media_type: str
    status: str
    current_version_id: str | None
    latest_version_id: str | None
    parent_count: int
    child_count: int
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None
    created_at: datetime
    updated_at: datetime


class TaskOut(BaseModel):
    """Public task state with an already-sanitized terminal error."""

    id: str
    kind: str
    status: str
    attempts: int
    error: str | None


class UploadAccepted(BaseModel):
    """Identifiers for an immutable candidate and its queued ingestion task."""

    document_id: str
    version_id: str
    task_id: str


class DocumentTaskAccepted(BaseModel):
    """Identifiers for a queued operation on an existing document."""

    document_id: str
    version_id: str | None
    task_id: str


@dataclass(frozen=True, slots=True)
class StoredUpload:
    """A validated raw upload staged under immutable document/version IDs."""

    document_id: str
    version_id: str
    path: Path
    sha256: str
    media_type: str
    kind: DocumentKind
