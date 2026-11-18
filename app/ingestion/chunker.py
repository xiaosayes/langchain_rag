"""Deterministic parent-child chunking that respects source structure."""

import hashlib
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import ChunkSettings
from app.domain.ids import stable_id
from app.domain.models import ChildChunk, ParentChunk, ParsedBlock

_SEPARATORS = ["\n\n", "\n", "。", "；", "，", ""]


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _same_structure(first: ParsedBlock, second: ParsedBlock) -> bool:
    return (
        first.location.heading_path == second.location.heading_path
        and first.location.sheet == second.location.sheet
        and first.location.page == second.location.page
        and first.metadata == second.metadata
    )


def _source_metadata(blocks: list[tuple[ParsedBlock, str]]) -> dict[str, Any]:
    first = blocks[0][0]
    metadata = dict(first.metadata)
    location = first.location
    if location.heading_path:
        metadata["heading_path"] = location.heading_path
    if location.page is not None:
        metadata["page"] = location.page
    if location.sheet is not None:
        metadata["sheet"] = location.sheet

    row_starts = [
        block.location.row_start
        for block, _ in blocks
        if block.location.row_start is not None
    ]
    row_ends = [
        block.location.row_end
        for block, _ in blocks
        if block.location.row_end is not None
    ]
    if row_starts:
        metadata["row_start"] = min(row_starts)
    if row_ends:
        metadata["row_end"] = max(row_ends)
    return metadata


def _splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=_SEPARATORS,
    )


def make_child(parent: ParentChunk, ordinal: int, text: str) -> ChildChunk:
    """Create one non-empty child with normalized text and copied lineage."""
    normalized = _normalize(text)
    if not normalized:
        raise ValueError(f"Child {ordinal} of parent {parent.id} is empty")
    content_hash = _content_hash(normalized)
    return ChildChunk(
        id=stable_id(
            "child", parent.document_id, parent.id, str(ordinal), content_hash
        ),
        parent_id=parent.id,
        document_id=parent.document_id,
        ordinal=ordinal,
        text=normalized,
        content_hash=content_hash,
        metadata=dict(parent.metadata),
    )


def chunk_blocks(
    document_id: str, blocks: list[ParsedBlock], settings: ChunkSettings
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Convert ordered source blocks into stable, structure-aware chunks."""
    if settings.parent_size <= 0 or settings.child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")
    if settings.child_overlap < 0 or settings.child_overlap >= settings.child_size:
        raise ValueError(
            "child_overlap must be non-negative and smaller than child_size"
        )

    parent_splitter = _splitter(settings.parent_size, 0)
    pieces: list[tuple[str, dict[str, Any]]] = []
    current: list[tuple[ParsedBlock, str]] = []
    current_length = 0

    def flush_current() -> None:
        nonlocal current, current_length
        if not current:
            return
        text = _normalize(" ".join(text for _, text in current))
        if text:
            pieces.append((text, _source_metadata(current)))
        current = []
        current_length = 0

    for block in blocks:
        text = _normalize(block.text)
        if not text:
            continue

        if block.metadata.get("record_type") == "qa":
            flush_current()
            if len(text) > settings.parent_size:
                raise ValueError(
                    f"Q&A block in document {document_id!r} at sequence "
                    f"{block.sequence} has {len(text)} characters, exceeding "
                    f"parent_size={settings.parent_size}; question and answer "
                    "must remain atomic"
                )
            pieces.append((text, _source_metadata([(block, text)])))
            continue

        if current and not _same_structure(current[0][0], block):
            flush_current()

        if len(text) > settings.parent_size:
            flush_current()
            metadata = _source_metadata([(block, text)])
            for split_text in parent_splitter.split_text(text):
                normalized = _normalize(split_text)
                if normalized:
                    pieces.append((normalized, dict(metadata)))
            continue

        added_length = len(text) + (1 if current else 0)
        if current and current_length + added_length > settings.parent_size:
            flush_current()
            added_length = len(text)

        current.append((block, text))
        current_length += added_length

    flush_current()

    parents: list[ParentChunk] = []
    for ordinal, (text, metadata) in enumerate(pieces):
        content_hash = _content_hash(text)
        parents.append(
            ParentChunk(
                id=stable_id("parent", document_id, str(ordinal), content_hash),
                document_id=document_id,
                ordinal=ordinal,
                text=text,
                content_hash=content_hash,
                metadata=metadata,
            )
        )

    child_splitter = _splitter(settings.child_size, settings.child_overlap)
    children: list[ChildChunk] = []
    for parent in parents:
        child_texts = (
            [parent.text]
            if parent.metadata.get("record_type") == "qa"
            else child_splitter.split_text(parent.text)
        )
        children.extend(
            make_child(parent, ordinal, text)
            for ordinal, text in enumerate(child_texts)
        )

    return parents, children
