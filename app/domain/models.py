"""Immutable values shared by ingestion, indexing, and retrieval."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class SourceLocation:
    page: int | None = None
    sheet: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    heading_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedBlock:
    document_id: str
    sequence: int
    text: str
    location: SourceLocation
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParentChunk:
    id: str
    document_id: str
    ordinal: int
    text: str
    content_hash: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ChildChunk:
    id: str
    parent_id: str
    document_id: str
    ordinal: int
    text: str
    content_hash: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchHit:
    child_id: str
    parent_id: str
    document_id: str
    text: str
    score: float
    source: Literal["dense", "bm25", "fused"]
    metadata: dict[str, Any]
