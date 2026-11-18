import logging

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage

from app.domain.models import SearchHit
from app.rag.service import INSUFFICIENT_CONTEXT_REFUSAL, RAGService
from app.retrieval.service import RetrievalResult


class FakeStreamingChatModel:
    def __init__(self, answer_chunks: list[str], rewrite: str = "改写后的查询") -> None:
        self.answer_chunks = answer_chunks
        self.rewrite = rewrite
        self.rewrite_calls = 0
        self.answer_calls = 0
        self.rewrite_messages: list[BaseMessage] = []
        self.answer_messages: list[BaseMessage] = []

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.rewrite_calls += 1
        self.rewrite_messages = messages
        return AIMessage(content=self.rewrite)

    async def astream(self, messages: list[BaseMessage]):
        self.answer_calls += 1
        self.answer_messages = messages
        for chunk in self.answer_chunks:
            yield AIMessageChunk(content=chunk)


class FakeRetrievalService:
    def __init__(self, documents: list[Document], *, degraded: bool = False) -> None:
        self.documents = documents
        self.degraded = degraded
        self.calls: list[tuple[str, str]] = []

    async def asearch(self, query: str, generation_id: str) -> RetrievalResult:
        self.calls.append((query, generation_id))
        return RetrievalResult(
            hits=[
                SearchHit(
                    child_id=f"child-{index}",
                    parent_id=document.metadata["parent_id"],
                    document_id=document.metadata["document_id"],
                    text=document.page_content,
                    score=1.0,
                    source="dense",
                    metadata={},
                )
                for index, document in enumerate(self.documents, start=1)
            ],
            documents=self.documents,
            degraded=self.degraded,
        )


def make_documents() -> list[Document]:
    return [
        Document(
            page_content="现行制度规定，住宿标准为每天500元。",
            metadata={
                "document_id": "doc-current",
                "parent_id": "parent-current",
                "generation_id": "gen-test",
                "version": "current",
            },
        ),
        Document(
            page_content="旧版制度规定，住宿标准为每天300元。",
            metadata={
                "document_id": "doc-old",
                "parent_id": "parent-old",
                "generation_id": "gen-test",
                "version": "superseded",
            },
        ),
    ]


@pytest.fixture
def fake_chat() -> FakeStreamingChatModel:
    return FakeStreamingChatModel(["住宿标准为每天", "500元[S1]。"])


@pytest.fixture
def retrieval_service() -> FakeRetrievalService:
    return FakeRetrievalService(make_documents(), degraded=True)


@pytest.fixture
def rag_service(
    fake_chat: FakeStreamingChatModel, retrieval_service: FakeRetrievalService
) -> RAGService:
    return RAGService(fake_chat, retrieval_service, object())  # type: ignore[arg-type]


def message_text(messages: list[BaseMessage]) -> str:
    return "\n".join(str(message.content) for message in messages)


@pytest.mark.asyncio
async def test_first_turn_skips_rewrite_and_streams_grounded_answer(
    rag_service: RAGService,
    fake_chat: FakeStreamingChatModel,
    retrieval_service: FakeRetrievalService,
):
    prepared = await rag_service.prepare("住宿标准？", [], "gen-test")

    assert prepared.original_question == "住宿标准？"
    assert prepared.retrieval_query == "住宿标准？"
    assert fake_chat.rewrite_calls == 0
    assert retrieval_service.calls == [("住宿标准？", "gen-test")]
    assert prepared.degraded is True
    assert prepared.generation_id == "gen-test"
    assert prepared.insufficient_context is False

    chunks = [chunk async for chunk in rag_service.stream_answer(prepared)]

    assert chunks == ["住宿标准为每天", "500元[S1]。"]
    assert "".join(chunks) == "住宿标准为每天500元[S1]。"
    assert fake_chat.answer_calls == 1
    prompt = message_text(fake_chat.answer_messages)
    assert "住宿标准？" in prompt
    assert "[S1]" in prompt
    assert "现行制度规定，住宿标准为每天500元。" in prompt
    assert "[S2]" in prompt
    assert "旧版制度规定，住宿标准为每天300元。" in prompt
    assert "中文" in prompt
    assert "当前版本" in prompt
    assert "不足" in prompt


@pytest.mark.asyncio
async def test_follow_up_rewrites_exactly_once_and_preserves_original_question(
    rag_service: RAGService,
    fake_chat: FakeStreamingChatModel,
    retrieval_service: FakeRetrievalService,
):
    history = [HumanMessage(content="差旅制度"), AIMessage(content="请继续提问")]

    prepared = await rag_service.prepare("住宿呢？", history, "gen-follow-up")

    assert fake_chat.rewrite_calls == 1
    assert prepared.original_question == "住宿呢？"
    assert prepared.retrieval_query == "改写后的查询"
    assert retrieval_service.calls == [("改写后的查询", "gen-follow-up")]
    assert history[0].content in message_text(fake_chat.rewrite_messages)
    assert "住宿呢？" in message_text(fake_chat.rewrite_messages)

    _ = [chunk async for chunk in rag_service.stream_answer(prepared)]
    assert fake_chat.rewrite_calls == 1
    assert "住宿呢？" in message_text(fake_chat.answer_messages)


@pytest.mark.asyncio
async def test_citations_label_each_supplied_parent_exactly_once(
    rag_service: RAGService,
):
    prepared = await rag_service.prepare("住宿标准？", [], "gen-test")

    assert [citation.label for citation in prepared.citations] == ["S1", "S2"]
    assert [citation.parent_id for citation in prepared.citations] == [
        "parent-current",
        "parent-old",
    ]
    assert [citation.document_id for citation in prepared.citations] == [
        "doc-current",
        "doc-old",
    ]
    assert [citation.text for citation in prepared.citations] == [
        document.page_content for document in prepared.documents
    ]
    assert all(
        citation.metadata == document.metadata
        for citation, document in zip(
            prepared.citations, prepared.documents, strict=True
        )
    )


@pytest.mark.asyncio
async def test_empty_context_returns_exact_refusal_without_answer_model_call(
    fake_chat: FakeStreamingChatModel,
):
    retrieval = FakeRetrievalService([])
    service = RAGService(fake_chat, retrieval, object())  # type: ignore[arg-type]

    prepared = await service.prepare("没有资料的问题", [], "gen-empty")
    chunks = [chunk async for chunk in service.stream_answer(prepared)]

    assert chunks == [INSUFFICIENT_CONTEXT_REFUSAL]
    assert chunks == ["根据当前知识库，无法可靠回答该问题。"]
    assert prepared.documents == []
    assert prepared.citations == []
    assert prepared.insufficient_context is True
    assert prepared.generation_id == "gen-empty"
    assert fake_chat.answer_calls == 0
    assert fake_chat.rewrite_calls == 0


@pytest.mark.asyncio
async def test_unknown_split_citation_is_removed_and_known_citations_remain(
    caplog: pytest.LogCaptureFixture,
):
    chat = FakeStreamingChatModel(
        ["现行标准[S", "1]，未知说法[S", "99]。旧版说明[S2]。"]
    )
    service = RAGService(
        chat,
        FakeRetrievalService(make_documents()),
        object(),  # type: ignore[arg-type]
    )
    prepared = await service.prepare("住宿标准？", [], "gen-test")

    with caplog.at_level(logging.WARNING, logger="app.rag.service"):
        chunks = [chunk async for chunk in service.stream_answer(prepared)]

    assert "".join(chunks) == "现行标准[S1]，未知说法。旧版说明[S2]。"
    assert "S99" in caplog.text
    assert "citation" in caplog.text.lower()
    assert chat.answer_calls == 1
