"""Typed values shared by document storage and ingestion services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DocumentKind = Literal["document", "qa"]


@dataclass(frozen=True, slots=True)
class StoredUpload:
    """A validated raw upload staged under immutable document/version IDs."""

    document_id: str
    version_id: str
    path: Path
    sha256: str
    media_type: str
    kind: DocumentKind
