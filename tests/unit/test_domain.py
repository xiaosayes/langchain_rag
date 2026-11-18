import hashlib
import re
from dataclasses import FrozenInstanceError

import pytest

from app.domain.ids import stable_id
from app.domain.models import (
    ChildChunk,
    ParentChunk,
    ParsedBlock,
    SearchHit,
    SourceLocation,
)


def test_stable_id_is_namespaced_and_repeatable():
    first = stable_id("child", "doc-1", "parent-1", "0", "报销")
    assert first == stable_id("child", "doc-1", "parent-1", "0", "报销")
    assert first != stable_id("parent", "doc-1", "parent-1", "0", "报销")
    assert first.startswith("child_")


def test_stable_id_uses_utf8_unit_separators_and_24_hex_characters():
    parts = ("doc-1", "parent-1", "0", "报销")
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    result = stable_id("child", *parts)

    assert result == f"child_{digest[:24]}"
    assert re.fullmatch(r"child_[0-9a-f]{24}", result)
    assert stable_id("child", "ab", "c") != stable_id("child", "a", "bc")


def test_source_location_preserves_page_sheet_rows_and_headings():
    location = SourceLocation(
        page=2,
        sheet="facts",
        row_start=3,
        row_end=8,
        heading_path=("财务", "报销"),
    )

    assert location.page == 2
    assert location.sheet == "facts"
    assert location.row_start == 3
    assert location.row_end == 8
    assert location.heading_path == ("财务", "报销")
    assert SourceLocation().heading_path == ()


def test_domain_values_are_frozen_and_preserve_metadata():
    location = SourceLocation(heading_path=("财务",))
    block = ParsedBlock(
        document_id="doc-1",
        sequence=0,
        text="原始文本",
        location=location,
    )
    parent = ParentChunk(
        id="parent-1",
        document_id="doc-1",
        ordinal=0,
        text="父块文本",
        content_hash="parent-hash",
        metadata={"heading_path": ("财务",)},
    )
    child = ChildChunk(
        id="child-1",
        parent_id="parent-1",
        document_id="doc-1",
        ordinal=0,
        text="子块文本",
        content_hash="child-hash",
        metadata={},
    )
    hit = SearchHit(
        child_id="child-1",
        parent_id="parent-1",
        document_id="doc-1",
        text="子块文本",
        score=0.5,
        source="dense",
        metadata={},
    )

    assert block.metadata == {}
    assert block.metadata is not ParsedBlock("doc-2", 1, "其他", location).metadata
    assert parent.document_id == child.document_id == hit.document_id
    with pytest.raises(FrozenInstanceError):
        block.sequence = 1
    with pytest.raises(FrozenInstanceError):
        parent.ordinal = 1
    with pytest.raises(FrozenInstanceError):
        child.ordinal = 1
    with pytest.raises(FrozenInstanceError):
        hit.score = 1.0
