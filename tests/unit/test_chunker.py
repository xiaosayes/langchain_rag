import hashlib
from pathlib import Path

import pytest

from app.config import ChunkSettings
from app.domain.models import ParsedBlock, SourceLocation
from app.ingestion.chunker import chunk_blocks
from app.ingestion.registry import ParserRegistry
from app.ingestion.text_office import QAParser
from app.mockgen.catalog import DocumentSpec, build_catalog

CORPUS = Path("mock_data/corpus")
KNOWLEDGE_QA = Path("mock_data/qa/knowledge_qa.xlsx")


def _block(
    sequence: int,
    text: str,
    location: SourceLocation | None = None,
    metadata: dict[str, object] | None = None,
) -> ParsedBlock:
    return ParsedBlock(
        document_id="doc-1",
        sequence=sequence,
        text=text,
        location=SourceLocation() if location is None else location,
        metadata={} if metadata is None else metadata,
    )


def _assert_chunk_invariants(parents, children, document_id: str) -> None:
    parent_ids = {parent.id for parent in parents}
    assert len(parent_ids) == len(parents)
    assert len({child.id for child in children}) == len(children)
    assert {child.parent_id for child in children} == parent_ids
    assert all(parent.document_id == document_id for parent in parents)
    assert all(child.document_id == document_id for child in children)
    assert all(parent.text.strip() for parent in parents)
    assert all(child.text.strip() for child in children)
    for parent in parents:
        assert (
            parent.content_hash
            == hashlib.sha256(parent.text.encode("utf-8")).hexdigest()
        )
    for child in children:
        assert (
            child.content_hash == hashlib.sha256(child.text.encode("utf-8")).hexdigest()
        )


def test_chunking_is_repeatable_and_references_real_parents():
    blocks = [
        _block(
            0,
            "报销制度。" * 400,
            SourceLocation(heading_path=("财务", "报销")),
        )
    ]
    settings = ChunkSettings(parent_size=1000, child_size=250, child_overlap=50)

    parents1, children1 = chunk_blocks("doc-1", blocks, settings)
    parents2, children2 = chunk_blocks("doc-1", blocks, settings)

    assert (parents1, children1) == (parents2, children2)
    _assert_chunk_invariants(parents1, children1, "doc-1")
    assert len(parents1) >= 2
    assert max(len(parent.text) for parent in parents1) >= 900
    assert all(len(parent.text) <= 1000 for parent in parents1)
    assert max(len(child.text) for child in children1) >= 200
    assert all(len(child.text) <= 250 for child in children1)
    assert all(
        parent.metadata["heading_path"] == ("财务", "报销") for parent in parents1
    )
    assert all(
        child.metadata["heading_path"] == ("财务", "报销") for child in children1
    )


def test_plain_text_children_use_the_configured_overlap():
    parents, children = chunk_blocks(
        "doc-1",
        [_block(0, "界" * 1000)],
        ChunkSettings(parent_size=1000, child_size=250, child_overlap=50),
    )

    assert len(parents) == 1
    assert [len(child.text) for child in children] == [250, 250, 250, 250, 200]
    assert children[0].text[-50:] == children[1].text[:50]
    assert all(child.parent_id == parents[0].id for child in children)


def test_structural_boundaries_and_source_metadata_survive():
    docx_metadata = {"parser": "docx", "custom": "kept"}
    blocks = [
        _block(
            0,
            "报销第一段",
            SourceLocation(heading_path=("财务", "报销")),
            docx_metadata,
        ),
        _block(
            1,
            "报销第二段",
            SourceLocation(heading_path=("财务", "报销")),
            docx_metadata,
        ),
        _block(
            2,
            "差旅第一段",
            SourceLocation(heading_path=("财务", "差旅")),
            docx_metadata,
        ),
        _block(3, "第二页", SourceLocation(page=2), {"parser": "pdf"}),
        _block(4, "第三页", SourceLocation(page=3), {"parser": "pdf"}),
        _block(
            5,
            "第一个行组",
            SourceLocation(sheet=" facts ", row_start=2, row_end=3),
            {"parser": "xlsx"},
        ),
        _block(
            6,
            "第二个行组",
            SourceLocation(sheet=" facts ", row_start=4, row_end=5),
            {"parser": "xlsx"},
        ),
        _block(
            7,
            "另一个工作表",
            SourceLocation(sheet="other", row_start=1, row_end=1),
            {"parser": "xlsx"},
        ),
    ]

    parents, children = chunk_blocks("doc-1", blocks, ChunkSettings())

    _assert_chunk_invariants(parents, children, "doc-1")
    assert len(parents) == 6
    assert parents[0].text == "报销第一段 报销第二段"
    assert parents[0].metadata == {
        "parser": "docx",
        "custom": "kept",
        "heading_path": ("财务", "报销"),
    }
    assert parents[1].metadata["heading_path"] == ("财务", "差旅")
    assert parents[2].metadata["page"] == 2
    assert parents[3].metadata["page"] == 3
    assert parents[4].text == "第一个行组 第二个行组"
    assert parents[4].metadata["sheet"] == " facts "
    assert (parents[4].metadata["row_start"], parents[4].metadata["row_end"]) == (2, 5)
    assert parents[5].metadata["sheet"] == "other"
    assert all(
        child.metadata
        == next(parent.metadata for parent in parents if parent.id == child.parent_id)
        for child in children
    )


def test_long_qa_is_one_atomic_parent_and_child():
    question = "Question: 报销制度的具体要求是什么？"
    answer = "Answer: " + "必须附完整凭证并经过审批。" * 20
    text = question + "\n" + answer
    assert len(text) > 250
    location = SourceLocation(sheet="knowledge_qa", row_start=2, row_end=2)
    metadata = {
        "record_type": "qa",
        "qa_id": "DOC-000-Q01",
        "source_document_id": "DOC-000",
        "domain": "财务管理",
    }

    parents, children = chunk_blocks(
        "doc-1", [_block(0, text, location, metadata)], ChunkSettings()
    )

    assert len(parents) == len(children) == 1
    assert parents[0].text == children[0].text
    assert question in parents[0].text
    assert answer in parents[0].text
    assert children[0].parent_id == parents[0].id
    assert children[0].metadata["record_type"] == "qa"
    assert children[0].metadata["qa_id"] == "DOC-000-Q01"
    assert children[0].metadata["row_start"] == 2


def test_over_limit_qa_raises_instead_of_splitting_question_and_answer():
    text = "Question: 要求？\nAnswer: " + "完整答案。" * 300

    with pytest.raises(ValueError, match=r"Q&A.*doc-1.*sequence 0.*parent_size=1000"):
        chunk_blocks(
            "doc-1", [_block(0, text, metadata={"record_type": "qa"})], ChunkSettings()
        )


def test_empty_blocks_are_skipped_and_hashes_use_normalized_text():
    parents, children = chunk_blocks(
        "doc-1",
        [_block(0, " \n\t "), _block(1, "  hello\n\tworld  ")],
        ChunkSettings(),
    )

    assert len(parents) == len(children) == 1
    assert parents[0].text == children[0].text == "hello world"
    _assert_chunk_invariants(parents, children, "doc-1")
    assert chunk_blocks("doc-1", [_block(0, " \n ")], ChunkSettings()) == ([], [])


@pytest.mark.parametrize(
    "spec",
    [item for item in build_catalog() if not item.scanned],
    ids=lambda spec: spec.id,
)
def test_non_scanned_corpus_chunks_have_complete_references(spec: DocumentSpec):
    blocks = ParserRegistry.default().parse(CORPUS / f"{spec.id}.{spec.fmt}", spec.id)

    parents, children = chunk_blocks(spec.id, blocks, ChunkSettings())

    _assert_chunk_invariants(parents, children, spec.id)
    assert all(len(parent.text) <= 1000 for parent in parents)
    assert all(len(child.text) <= 250 for child in children)
    for block in blocks:
        location = block.location
        matching_parents = [
            parent
            for parent in parents
            if all(
                parent.metadata.get(key) == value
                for key, value in (
                    ("page", location.page),
                    ("sheet", location.sheet),
                    ("heading_path", location.heading_path),
                )
                if value is not None and value != ()
            )
        ]
        assert matching_parents


def test_knowledge_qa_corpus_remains_atomic():
    blocks = QAParser().parse(KNOWLEDGE_QA)

    parents, children = chunk_blocks("knowledge_qa", blocks, ChunkSettings())

    _assert_chunk_invariants(parents, children, "knowledge_qa")
    assert len(parents) == len(children) == len(blocks) == 1000
    assert all(
        len(parent.text) == len(child.text)
        for parent, child in zip(parents, children, strict=True)
    )
    assert {parent.metadata["qa_id"] for parent in parents} == {
        block.metadata["qa_id"] for block in blocks
    }
