from pathlib import Path

import pytest

from app.config import RetrievalSettings
from app.db.session import create_session_factory
from app.domain.models import ParentChunk, SearchHit
from app.indexing.repository import IndexRepository
from app.models.bailian import RerankResult
from app.retrieval.service import RetrievalService, RetrievalUnavailable


class FakeEmbeddings:
    async def aembed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


class FakeStore:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.error: Exception | None = None
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        self.calls.append((query, top_k))
        if self.error is not None:
            raise self.error
        return self.hits[:top_k]


class FakeQdrant(FakeStore):
    def search(
        self, generation_id: str, vector: list[float], top_k: int
    ) -> list[SearchHit]:
        return super().search(generation_id, top_k)


class FakeReranker:
    def __init__(self, *, reverse: bool = False) -> None:
        self.reverse = reverse
        self.error: Exception | None = None
        self.calls: list[tuple[str, list[str], int]] = []

    async def arerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[RerankResult]:
        self.calls.append((query, documents, top_n))
        if self.error is not None:
            raise self.error
        indices = list(range(len(documents)))
        if self.reverse:
            indices.reverse()
        return [
            RerankResult(index=index, relevance_score=1.0 - index / 100)
            for index in indices[:top_n]
        ]


def make_hit(
    child_id: str,
    parent_id: str,
    *,
    document_id: str = "doc-1",
    source: str = "dense",
) -> SearchHit:
    return SearchHit(
        child_id=child_id,
        parent_id=parent_id,
        document_id=document_id,
        text=f"text for {child_id}",
        score=1.0,
        source=source,  # type: ignore[arg-type]
        metadata={"source": f"{parent_id}.txt"},
    )


def make_parent(
    parent_id: str,
    *,
    document_id: str = "doc-1",
    text: str | None = None,
) -> ParentChunk:
    return ParentChunk(
        id=parent_id,
        document_id=document_id,
        ordinal=0,
        text=parent_id if text is None else text,
        content_hash=f"hash-{parent_id}",
        metadata={"source": f"{parent_id}.txt", "page": 3},
    )


def activate(
    repository: IndexRepository, generation_id: str, parents: list[ParentChunk]
) -> None:
    repository.prepare_generation(generation_id)
    repository.activate_generation(generation_id, parents)


def build_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    dense_hits: list[SearchHit],
    bm25_hits: list[SearchHit],
    parents: list[ParentChunk],
    reranker: FakeReranker | None = None,
    settings: RetrievalSettings | None = None,
) -> RetrievalService:
    repository = IndexRepository(create_session_factory(tmp_path / "rag.sqlite3"))
    activate(repository, "gen-test", parents)
    dense = FakeQdrant(dense_hits)
    bm25 = FakeStore(bm25_hits)
    service = RetrievalService(
        repository,
        dense,  # type: ignore[arg-type]
        tmp_path / "bm25",
        FakeEmbeddings(),  # type: ignore[arg-type]
        reranker or FakeReranker(reverse=True),  # type: ignore[arg-type]
        settings or RetrievalSettings(),
    )
    monkeypatch.setattr(service, "load_bm25", lambda generation_id: bm25)
    return service


@pytest.fixture
def retrieval_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> RetrievalService:
    return build_service(
        tmp_path,
        monkeypatch,
        dense_hits=[
            make_hit("child-from-first-store", "parent-first"),
            make_hit("child-shared", "parent-shared"),
        ],
        bm25_hits=[
            make_hit("child-shared", "parent-shared", source="bm25"),
            make_hit(
                "child-from-second-store",
                "parent-second",
                document_id="doc-2",
                source="bm25",
            ),
        ],
        parents=[
            make_parent("parent-first"),
            make_parent("parent-shared"),
            make_parent("parent-second", document_id="doc-2"),
        ],
    )


@pytest.mark.asyncio
async def test_rerank_order_controls_parent_selection(
    retrieval_service: RetrievalService,
):
    result = await retrieval_service.asearch("报销标准", "gen-test")

    assert [hit.child_id for hit in result.hits] == [
        "child-from-second-store",
        "child-from-first-store",
        "child-shared",
    ]
    assert [document.metadata["parent_id"] for document in result.documents] == [
        "parent-second",
        "parent-first",
        "parent-shared",
    ]
    assert len(result.documents) <= 6
    assert all(
        document.metadata["generation_id"] == "gen-test"
        for document in result.documents
    )
    assert result.documents[0].metadata == {
        "generation_id": "gen-test",
        "document_id": "doc-2",
        "parent_id": "parent-second",
        "source": "parent-second.txt",
        "page": 3,
    }
    assert sum(len(document.page_content) for document in result.documents) <= 12000
    assert result.degraded is False


@pytest.mark.asyncio
async def test_parent_deduplication_document_limit_and_total_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    parents = [
        make_parent(f"parent-{index}", document_id=f"doc-{(index - 1) // 3 + 1}")
        for index in range(1, 10)
    ]
    hits = [make_hit("child-duplicate", "parent-1")] + [
        make_hit(f"child-{index}", f"parent-{index}") for index in range(1, 10)
    ]
    service = build_service(
        tmp_path,
        monkeypatch,
        dense_hits=hits,
        bm25_hits=[],
        parents=parents,
        reranker=FakeReranker(),
    )

    result = await service.asearch("policy", "gen-test")

    assert [document.metadata["parent_id"] for document in result.documents] == [
        "parent-1",
        "parent-2",
        "parent-4",
        "parent-5",
        "parent-7",
        "parent-8",
    ]
    assert len({document.metadata["parent_id"] for document in result.documents}) == 6


@pytest.mark.asyncio
async def test_context_budget_skips_oversized_parents_without_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service = build_service(
        tmp_path,
        monkeypatch,
        dense_hits=[
            make_hit("child-long", "parent-long"),
            make_hit("child-skip", "parent-skip", document_id="doc-2"),
            make_hit("child-fit", "parent-fit", document_id="doc-3"),
        ],
        bm25_hits=[],
        parents=[
            make_parent("parent-long", text="a" * 11900),
            make_parent("parent-skip", document_id="doc-2", text="b" * 200),
            make_parent("parent-fit", document_id="doc-3", text="c" * 100),
        ],
        reranker=FakeReranker(),
    )

    result = await service.asearch("policy", "gen-test")

    assert [document.metadata["parent_id"] for document in result.documents] == [
        "parent-long",
        "parent-fit",
    ]
    assert sum(len(document.page_content) for document in result.documents) == 12000


@pytest.mark.asyncio
async def test_rerank_receives_at_most_thirty_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    reranker = FakeReranker()
    service = build_service(
        tmp_path,
        monkeypatch,
        dense_hits=[
            make_hit(f"child-{index}", f"parent-{index}") for index in range(35)
        ],
        bm25_hits=[],
        parents=[make_parent(f"parent-{index}") for index in range(35)],
        reranker=reranker,
    )

    result = await service.asearch("policy", "gen-test")

    assert len(result.hits) == 30
    assert len(reranker.calls[0][1]) == 30
    assert reranker.calls[0][2] == 30


@pytest.mark.asyncio
async def test_dense_failure_degrades_to_bm25(
    retrieval_service: RetrievalService,
):
    retrieval_service.qdrant.error = RuntimeError("dense unavailable")

    result = await retrieval_service.asearch("policy", "gen-test")

    assert result.degraded is True
    assert {hit.child_id for hit in result.hits} == {
        "child-shared",
        "child-from-second-store",
    }
    assert result.documents


@pytest.mark.asyncio
async def test_bm25_failure_degrades_to_dense(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service = build_service(
        tmp_path,
        monkeypatch,
        dense_hits=[make_hit("child-dense", "parent-dense")],
        bm25_hits=[],
        parents=[make_parent("parent-dense")],
    )
    service.load_bm25("gen-test").error = LookupError("BM25 unavailable")

    result = await service.asearch("policy", "gen-test")

    assert result.degraded is True
    assert [hit.child_id for hit in result.hits] == ["child-dense"]
    assert [document.page_content for document in result.documents] == ["parent-dense"]


@pytest.mark.asyncio
async def test_rerank_failure_preserves_fused_order_and_degrades(
    retrieval_service: RetrievalService,
):
    retrieval_service.reranker.error = RuntimeError("rerank unavailable")

    result = await retrieval_service.asearch("policy", "gen-test")

    assert result.degraded is True
    assert [hit.child_id for hit in result.hits] == [
        "child-shared",
        "child-from-first-store",
        "child-from-second-store",
    ]
    assert result.documents


@pytest.mark.asyncio
async def test_both_store_failures_raise_retrieval_unavailable(
    retrieval_service: RetrievalService,
):
    retrieval_service.qdrant.error = RuntimeError("dense unavailable")
    retrieval_service.load_bm25("gen-test").error = LookupError("BM25 unavailable")

    with pytest.raises(RetrievalUnavailable) as caught:
        await retrieval_service.asearch("policy", "gen-test")

    assert "RuntimeError" in str(caught.value)
    assert "LookupError" in str(caught.value)


@pytest.mark.asyncio
async def test_parent_resolution_never_reads_another_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service = build_service(
        tmp_path,
        monkeypatch,
        dense_hits=[make_hit("child", "shared-parent")],
        bm25_hits=[],
        parents=[make_parent("shared-parent", text="requested generation")],
        reranker=FakeReranker(),
    )
    activate(
        service.repository,
        "gen-other",
        [make_parent("shared-parent", text="other generation")],
    )

    result = await service.asearch("policy", "gen-test")

    assert [document.page_content for document in result.documents] == [
        "requested generation"
    ]
    assert result.documents[0].metadata["generation_id"] == "gen-test"
