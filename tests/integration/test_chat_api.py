from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from sqlalchemy import func, select

from app.api.chat import _encode_event, _encode_stream, stream_chat
from app.chat.schemas import ChatRequest
from app.config import AppSettings
from app.db.models import IndexGenerationRecord
from app.main import create_app
from app.rag.service import Citation, RAGPreparation, RAGService
from app.web.records import FeedbackRecord, MessageRecord


class FakeRAGService:
    def __init__(self) -> None:
        self.chunks = ["grounded ", "answer[S1]"]
        self.prepare_error: Exception | None = None
        self.stream_error: Exception | None = None
        self.cleanup_error: Exception | None = None
        self.histories = []
        self.generation_ids = []
        self.prepare_run_ids = []
        self.stream_run_ids = []
        self.prepare_calls = 0
        self.block_preparation = False
        self.block_stream = False
        self.preparation_started = asyncio.Event()
        self.release_preparation = asyncio.Event()
        self.stream_waiting = asyncio.Event()
        self.stream_closed = asyncio.Event()

    async def prepare(self, question, history, generation_id, *, run_id=None):
        self.prepare_calls += 1
        self.histories.append(history)
        self.generation_ids.append(generation_id)
        self.prepare_run_ids.append(run_id)
        if self.prepare_error is not None:
            raise self.prepare_error
        if self.block_preparation:
            if self.prepare_calls == 5:
                self.preparation_started.set()
            await self.release_preparation.wait()
        documents = [
            Document(
                page_content=f"source-{index}",
                metadata={"document_id": f"doc-{index}", "parent_id": f"parent-{index}"},
            )
            for index in range(1, 3)
        ]
        return RAGPreparation(
            original_question=question,
            retrieval_query=question,
            documents=documents,
            citations=[
                Citation(
                    label=f"S{index}",
                    document_id=document.metadata["document_id"],
                    parent_id=document.metadata["parent_id"],
                    text=document.page_content,
                    metadata=dict(document.metadata),
                )
                for index, document in enumerate(documents, start=1)
            ],
            degraded=True,
            generation_id=generation_id,
            insufficient_context=False,
        )

    async def stream_answer(self, prepared, *, run_id=None):
        self.stream_run_ids.append(run_id)
        try:
            if self.chunks:
                yield self.chunks[0]
            if self.stream_error is not None:
                raise self.stream_error
            for chunk in self.chunks[1:]:
                yield chunk
            if self.block_stream:
                self.stream_waiting.set()
                await asyncio.Event().wait()
        finally:
            self.stream_closed.set()
            if self.cleanup_error is not None:
                raise self.cleanup_error


def make_app(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAG_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "correct-password")
    settings = AppSettings(app_env="test", model_provider="fake")
    app = create_app(settings=settings, qdrant_probe=lambda: {"status": "ok"})
    fake = FakeRAGService()
    app.state.rag_service = fake
    if hasattr(app.state, "chat_service"):
        app.state.chat_service.rag_service = fake
    return app, fake


def activate_generation(app, generation_id: str = "gen-active") -> None:
    factory = app.state.conversation_service.session_factory
    with factory.begin() as session:
        session.add(IndexGenerationRecord(id=generation_id, status="active"))


def login(client: TestClient) -> str:
    response = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "correct-password"},
    )
    assert response.status_code == 200
    return response.json()["csrf_token"]


def chat(client: TestClient, csrf: str, payload: dict):
    return client.post(
        "/api/chat/stream", json=payload, headers={"X-CSRF-Token": csrf}
    )


def assert_one_terminal_run(app, status: str, answer: str) -> dict:
    runs = app.state.monitoring_service.list_runs()
    assert len(runs) == 1
    detail = app.state.monitoring_service.get_run(runs[0]["id"])
    assert detail is not None
    assert detail["status"] == status
    assert detail["answer"] == answer
    trace_path = app.state.monitoring_service.trace_path
    assert len(trace_path.read_text(encoding="utf-8").splitlines()) == 1
    return detail


def sse_events(response) -> list[tuple[str, dict]]:
    events = []
    for block in response.text.replace("\r\n", "\n").strip().split("\n\n"):
        lines = block.splitlines()
        event = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
        data = "\n".join(
            line.removeprefix("data: ") for line in lines if line.startswith("data: ")
        )
        events.append((event, json.loads(data)))
    return events


def test_sse_order_json_citations_and_persisted_metadata(tmp_path, monkeypatch):
    app, fake = make_app(tmp_path, monkeypatch)
    activate_generation(app)
    with TestClient(app) as client:
        csrf = login(client)
        response = chat(client, csrf, {"question": "What is grounded?"})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = sse_events(response)
        assert [event for event, _ in events] == [
            "start",
            "token",
            "token",
            "citations",
            "done",
        ]
        start = events[0][1]
        assert start["conversation_id"]
        assert start["message_id"]
        assert start["generation_id"] == "gen-active"
        assert start["degraded"] is True
        assert events[1][1] == {"content": "grounded "}
        assert events[2][1] == {"content": "answer[S1]"}
        citations = events[3][1]["citations"]
        assert [citation["label"] for citation in citations] == ["S1", "S2"]
        assert [citation["text"] for citation in citations] == ["source-1", "source-2"]
        assert events[4][1]["message_id"] == start["message_id"]
        assert fake.histories == [[]]

        messages = app.state.conversation_service.list_messages(start["conversation_id"])
        assert [(message.role, message.content) for message in messages] == [
            ("user", "What is grounded?"),
            ("assistant", "grounded answer[S1]"),
        ]
        assistant = messages[1]
        assert assistant.id == start["message_id"]
        assert assistant.citations_json == citations
        assert assistant.degraded is True
        assert assistant.generation_id == "gen-active"
        assert assistant.cancelled is False
        assert assistant.error is None

        detail = assert_one_terminal_run(app, "succeeded", "grounded answer[S1]")
        assert detail["message_id"] == start["message_id"]
        assert detail["conversation_id"] == start["conversation_id"]
        assert detail["error"] is None
        assert fake.prepare_run_ids == [detail["id"]]
        assert fake.stream_run_ids == [detail["id"]]


def test_history_is_last_six_prior_messages_with_langchain_roles(tmp_path, monkeypatch):
    app, fake = make_app(tmp_path, monkeypatch)
    activate_generation(app)
    factory = app.state.conversation_service.session_factory
    with TestClient(app) as client:
        csrf = login(client)
        conversation_id = client.post(
            "/api/sessions", json={}, headers={"X-CSRF-Token": csrf}
        ).json()["id"]
        base = datetime(2026, 1, 1, tzinfo=UTC)
        with factory.begin() as session:
            for index in range(8):
                session.add(
                    MessageRecord(
                        id=f"message-{index}",
                        conversation_id=conversation_id,
                        role="user" if index % 2 == 0 else "assistant",
                        content=f"content-{index}",
                        created_at=base + timedelta(minutes=index),
                    )
                )

        response = chat(
            client,
            csrf,
            {"conversation_id": conversation_id, "question": "follow up"},
        )

        assert response.status_code == 200
        history = fake.histories[0]
        assert [message.content for message in history] == [
            f"content-{index}" for index in range(2, 8)
        ]
        assert all(
            isinstance(message, HumanMessage if index % 2 == 0 else AIMessage)
            for index, message in zip(range(2, 8), history, strict=True)
        )
        assert "follow up" not in [message.content for message in history]
        assert fake.generation_ids == ["gen-active"]


def test_missing_generation_is_a_sanitized_sse_error(tmp_path, monkeypatch):
    app, fake = make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        csrf = login(client)
        response = chat(client, csrf, {"question": "question"})

        assert response.status_code == 200
        events = sse_events(response)
        assert [event for event, _ in events] == ["error"]
        assert "active" in events[0][1]["message"].lower()
        assert fake.prepare_calls == 0
        conversation = app.state.conversation_service.list()[0]
        messages = app.state.conversation_service.list_messages(conversation.id)
        assert [(message.role, message.content) for message in messages] == [
            ("user", "question")
        ]
        detail = assert_one_terminal_run(app, "failed", "")
        assert detail["message_id"] is None
        assert detail["error"] == events[0][1]["message"]


def test_provider_failure_is_sanitized_and_partial_is_not_done(tmp_path, monkeypatch):
    app, fake = make_app(tmp_path, monkeypatch)
    activate_generation(app)
    secret = "provider-secret-token"
    fake.stream_error = RuntimeError(f"provider failed with {secret}")
    with TestClient(app) as client:
        csrf = login(client)
        response = chat(client, csrf, {"question": "question"})

        assert response.status_code == 200
        events = sse_events(response)
        assert [event for event, _ in events] == ["start", "token", "error"]
        assert secret not in response.text
        assert "provider failed with" not in response.text
        conversation_id = events[0][1]["conversation_id"]
        factory = app.state.conversation_service.session_factory
        with factory() as session:
            assistant = session.scalar(
                select(MessageRecord).where(MessageRecord.role == "assistant")
            )
            assert assistant is not None
            assert assistant.content == "grounded "
            assert assistant.cancelled is False
            assert assistant.error == events[-1][1]["message"]
            assert secret not in (assistant.error or "")
        assert app.state.conversation_service.list_messages(conversation_id)
        detail = assert_one_terminal_run(app, "failed", "grounded ")
        assert detail["message_id"] == events[0][1]["message_id"]
        assert detail["error"] == events[-1][1]["message"]
        assert secret not in (detail["error"] or "")


def test_cancellation_closes_upstream_and_persists_partial_cancelled(
    tmp_path, monkeypatch
):
    async def exercise():
        app, fake = make_app(tmp_path, monkeypatch)
        activate_generation(app)
        fake.block_stream = True
        fake.chunks = ["grounded "]
        service = app.state.chat_service
        conversation_id = app.state.conversation_service.create().id
        run_id = app.state.monitoring_service.start_run("question", conversation_id)
        prepared = await service.prepare("question", conversation_id, run_id=run_id)
        events = service.stream(conversation_id, prepared, run_id=run_id)
        start = await anext(events)
        token = await anext(events)
        assert start.event == "start"
        assert token.event == "token"
        pending = asyncio.create_task(anext(events))
        await asyncio.wait_for(fake.stream_waiting.wait(), timeout=1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await asyncio.wait_for(fake.stream_closed.wait(), timeout=1)
        await events.aclose()

        factory = app.state.conversation_service.session_factory
        with factory() as session:
            assistant = session.scalar(
                select(MessageRecord).where(MessageRecord.role == "assistant")
            )
            assert assistant is not None
            assert assistant.content == "grounded "
            assert assistant.cancelled is True
            assert assistant.error is None
        detail = assert_one_terminal_run(app, "cancelled", "grounded ")
        assert detail["message_id"] == start.data["message_id"]
        assert detail["error"] is None

    asyncio.run(exercise())


@pytest.mark.asyncio
async def test_close_after_token_finalizes_real_rag_before_terminal_snapshot(
    tmp_path, monkeypatch
):
    app, _ = make_app(tmp_path, monkeypatch)
    activate_generation(app)

    class ChatModel:
        async def astream(self, messages):
            yield AIMessageChunk(content="partial ")
            yield AIMessageChunk(content="unconsumed")

    monitoring = app.state.monitoring_service
    rag = RAGService(
        ChatModel(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        monitoring_service=monitoring,
    )
    service = app.state.chat_service
    service.rag_service = rag
    conversation_id = app.state.conversation_service.create().id
    run_id = monitoring.start_run("question", conversation_id)
    document = Document(
        page_content="source",
        metadata={"document_id": "doc-1", "parent_id": "parent-1"},
    )
    prepared = RAGPreparation(
        original_question="question",
        retrieval_query="question",
        documents=[document],
        citations=[
            Citation(
                label="S1",
                document_id="doc-1",
                parent_id="parent-1",
                text="source",
                metadata=dict(document.metadata),
            )
        ],
        degraded=False,
        generation_id="gen-active",
        insufficient_context=False,
    )
    events = service.stream(conversation_id, prepared, run_id=run_id)
    assert (await anext(events)).event == "start"
    assert (await anext(events)).data == {"content": "partial "}

    await events.aclose()

    detail = assert_one_terminal_run(app, "cancelled", "partial ")
    snapshot = json.loads(
        monitoring.trace_path.read_text(encoding="utf-8").splitlines()[0]
    )
    for traced in (detail, snapshot):
        chat_events = [event for event in traced["events"] if event["stage"] == "chat"]
        assert len(chat_events) == 1
        assert chat_events[0]["status"] == "cancelled"
    factory = app.state.conversation_service.session_factory
    with factory() as session:
        assistants = session.scalars(
            select(MessageRecord).where(MessageRecord.role == "assistant")
        ).all()
        assert len(assistants) == 1
        assert assistants[0].content == "partial "
        assert assistants[0].cancelled is True


@pytest.mark.asyncio
async def test_api_stream_close_before_first_advance_finishes_cancelled_run(
    tmp_path, monkeypatch
):
    app, _ = make_app(tmp_path, monkeypatch)
    activate_generation(app)
    request = Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "method": "POST",
            "scheme": "http",
            "path": "/api/chat/stream",
            "raw_path": b"/api/chat/stream",
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("testclient", 123),
            "server": ("testserver", 80),
            "app": app,
        }
    )
    response = await stream_chat(ChatRequest(question="early close"), request)

    await response.body_iterator.aclose()

    detail = assert_one_terminal_run(app, "cancelled", "")
    assert detail["message_id"] is None
    assert not app.state.chat_semaphore.locked()
    factory = app.state.conversation_service.session_factory
    with factory() as session:
        assert (
            session.scalars(
                select(MessageRecord).where(MessageRecord.role == "assistant")
            ).all()
            == []
        )


@pytest.mark.asyncio
async def test_encoding_failure_closes_upstream_releases_slot_and_cancels_run(
    tmp_path, monkeypatch
):
    app, fake = make_app(tmp_path, monkeypatch)
    activate_generation(app)
    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    service = app.state.chat_service
    conversation_id = app.state.conversation_service.create().id
    run_id = app.state.monitoring_service.start_run("question", conversation_id)
    prepared = await service.prepare("question", conversation_id, run_id=run_id)
    encoded = _encode_stream(
        service.stream(conversation_id, prepared, run_id=run_id), semaphore.release
    )

    def fail_token_encoding(event):
        if event.event == "token":
            raise RuntimeError("encoding failed")
        return _encode_event(event)

    monkeypatch.setattr("app.api.chat._encode_event", fail_token_encoding)
    assert (await anext(encoded)).startswith(b"event: start")
    with pytest.raises(RuntimeError, match="encoding failed"):
        await anext(encoded)

    try:
        assert fake.stream_closed.is_set()
        assert not semaphore.locked()
        detail = assert_one_terminal_run(app, "cancelled", "grounded ")
        assert detail["error"] is None
        factory = app.state.conversation_service.session_factory
        with factory() as session:
            assistant = session.scalar(
                select(MessageRecord).where(MessageRecord.role == "assistant")
            )
            assert assistant is not None
            assert assistant.content == "grounded "
            assert assistant.cancelled is True
    finally:
        await encoded.aclose()


@pytest.mark.asyncio
async def test_generator_cleanup_failure_releases_slot_for_next_request(
    tmp_path, monkeypatch
):
    app, fake = make_app(tmp_path, monkeypatch)
    activate_generation(app)
    app.state.chat_semaphore = asyncio.Semaphore(1)
    fake.chunks = ["partial"]
    fake.cleanup_error = RuntimeError("upstream cleanup interrupted")
    semaphore = app.state.chat_semaphore
    await semaphore.acquire()
    service = app.state.chat_service
    prepared = await service.prepare("first", None)
    conversation_id = app.state.conversation_service.list()[0].id
    encoded = _encode_stream(
        service.stream(conversation_id, prepared), semaphore.release
    )
    assert (await anext(encoded)).startswith(b"event: start")
    assert (await anext(encoded)).startswith(b"event: token")

    with pytest.raises(RuntimeError, match="upstream cleanup interrupted"):
        await encoded.aclose()

    assert not semaphore.locked()
    fake.cleanup_error = None
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        login_response = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "correct-password"},
        )
        csrf = login_response.json()["csrf_token"]
        response = await client.post(
            "/api/chat/stream",
            json={"question": "after cleanup"},
            headers={"X-CSRF-Token": csrf},
        )
        assert response.status_code == 200
        assert sse_events(response)[-1][0] == "done"


@pytest.mark.asyncio
async def test_asgi_disconnect_persists_partial_and_releases_route_slot(
    tmp_path, monkeypatch
):
    app, fake = make_app(tmp_path, monkeypatch)
    activate_generation(app)
    app.state.chat_semaphore = asyncio.Semaphore(1)
    fake.chunks = ["partial"]
    fake.block_stream = True
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        login_response = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "correct-password"},
        )
        csrf = login_response.json()["csrf_token"]
        cookie = "; ".join(f"{name}={value}" for name, value in client.cookies.items())
        body = json.dumps({"question": "disconnect"}).encode()
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/chat/stream",
            "raw_path": b"/api/chat/stream",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"testserver"),
                (b"cookie", cookie.encode()),
                (b"x-csrf-token", csrf.encode()),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
            "client": ("testclient", 123),
            "server": ("testserver", 80),
        }
        token_sent = asyncio.Event()
        request_sent = False
        sent_messages = []

        async def receive():
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.wait_for(token_sent.wait(), timeout=1)
            return {"type": "http.disconnect"}

        async def send(message):
            sent_messages.append(message)
            if message["type"] == "http.response.body" and b"event: token" in message.get(
                "body", b""
            ):
                token_sent.set()

        await asyncio.wait_for(app(scope, receive, send), timeout=2)
        await asyncio.wait_for(fake.stream_closed.wait(), timeout=1)

        factory = app.state.conversation_service.session_factory
        with factory() as session:
            assistant = session.scalar(
                select(MessageRecord).where(MessageRecord.role == "assistant")
            )
            assert assistant is not None
            assert assistant.content == "partial"
            assert assistant.cancelled is True
            assert assistant.error is None
        assert not app.state.chat_semaphore.locked()
        detail = assert_one_terminal_run(app, "cancelled", "partial")
        assert detail["message_id"] == assistant.id
        assert detail["error"] is None
        assert any(
            message["type"] == "http.response.body"
            and b"event: token" in message.get("body", b"")
            for message in sent_messages
        )

        fake.block_stream = False
        response = await client.post(
            "/api/chat/stream",
            json={"question": "after disconnect"},
            headers={"X-CSRF-Token": csrf},
        )
        assert response.status_code == 200
        assert sse_events(response)[-1][0] == "done"


@pytest.mark.asyncio
async def test_sixth_concurrent_chat_gets_immediate_429(tmp_path, monkeypatch):
    app, fake = make_app(tmp_path, monkeypatch)
    activate_generation(app)
    fake.block_preparation = True
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "correct-password"},
        )
        csrf = response.json()["csrf_token"]
        headers = {"X-CSRF-Token": csrf}
        requests = [
            asyncio.create_task(
                client.post(
                    "/api/chat/stream", json={"question": f"question-{index}"}, headers=headers
                )
            )
            for index in range(5)
        ]
        await asyncio.wait_for(fake.preparation_started.wait(), timeout=2)

        sixth = await client.post(
            "/api/chat/stream", json={"question": "sixth"}, headers=headers
        )
        assert sixth.status_code == 429
        assert fake.prepare_calls == 5

        fake.release_preparation.set()
        responses = await asyncio.gather(*requests)
        assert [response.status_code for response in responses] == [200] * 5


@pytest.mark.parametrize("provider_failure", [False, True])
def test_monitoring_failures_never_change_sse_or_partial_answers(
    tmp_path, monkeypatch, provider_failure: bool
):
    app, fake = make_app(tmp_path, monkeypatch)
    activate_generation(app)
    if provider_failure:
        fake.stream_error = RuntimeError("provider-secret")

    def fail_monitoring(*args, **kwargs):
        raise RuntimeError("monitoring unavailable")

    monkeypatch.setattr(app.state.monitoring_service, "set_message_id", fail_monitoring)
    monkeypatch.setattr(app.state.monitoring_service, "finish_run", fail_monitoring)
    with TestClient(app) as client:
        response = chat(client, login(client), {"question": "question"})

    assert response.status_code == 200
    events = sse_events(response)
    if provider_failure:
        assert [event for event, _ in events] == ["start", "token", "error"]
        assert "provider-secret" not in response.text
    else:
        assert [event for event, _ in events] == [
            "start",
            "token",
            "token",
            "citations",
            "done",
        ]
        assert events[-2][1]["citations"][0]["text"] == "source-1"


def test_monitoring_start_failure_does_not_change_chat(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    activate_generation(app)

    def fail_start(*args, **kwargs):
        raise RuntimeError("monitoring unavailable")

    monkeypatch.setattr(app.state.monitoring_service, "start_run", fail_start)
    with TestClient(app) as client:
        response = chat(client, login(client), {"question": "question"})

    assert response.status_code == 200
    assert [event for event, _ in sse_events(response)] == [
        "start",
        "token",
        "token",
        "citations",
        "done",
    ]


def test_feedback_is_local_upsert_with_validation(tmp_path, monkeypatch):
    app, _ = make_app(tmp_path, monkeypatch)
    factory = app.state.conversation_service.session_factory
    with TestClient(app) as client:
        csrf = login(client)
        conversation_id = client.post(
            "/api/sessions", json={}, headers={"X-CSRF-Token": csrf}
        ).json()["id"]
        with factory.begin() as session:
            session.add(
                MessageRecord(
                    id="assistant-message",
                    conversation_id=conversation_id,
                    role="assistant",
                    content="answer",
                )
            )

        maximum_comment = "x" * 2000
        first = client.post(
            "/api/feedback",
            json={
                "message_id": "assistant-message",
                "score": 1,
                "comment": maximum_comment,
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert first.status_code == 200
        assert first.json()["score"] == 1
        assert first.json()["comment"] == maximum_comment
        assert "run_id" not in first.json()
        assert "langsmith" not in first.text.lower()

        second = client.post(
            "/api/feedback",
            json={"message_id": "assistant-message", "score": -1, "comment": "updated"},
            headers={"X-CSRF-Token": csrf},
        )
        assert second.status_code == 200
        assert second.json()["id"] == first.json()["id"]
        assert second.json()["score"] == -1
        assert second.json()["comment"] == "updated"
        with factory() as session:
            assert session.scalar(select(func.count()).select_from(FeedbackRecord)) == 1
            feedback = session.scalar(select(FeedbackRecord))
            assert feedback is not None
            assert not hasattr(feedback, "run_id")

        assert (
            client.post(
                "/api/feedback",
                json={"message_id": "missing", "score": 1},
                headers={"X-CSRF-Token": csrf},
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/api/feedback",
                json={"message_id": "assistant-message", "score": 0},
                headers={"X-CSRF-Token": csrf},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/feedback",
                json={
                    "message_id": "assistant-message",
                    "score": 1,
                    "comment": "x" * 2001,
                },
                headers={"X-CSRF-Token": csrf},
            ).status_code
            == 422
        )


@pytest.mark.parametrize("path", ["/api/chat/stream", "/api/feedback"])
def test_chat_and_feedback_require_authentication(tmp_path, monkeypatch, path):
    app, _ = make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        assert client.post(path, json={}).status_code == 401


@pytest.mark.parametrize("path", ["/api/chat/stream", "/api/feedback"])
def test_chat_and_feedback_require_csrf(tmp_path, monkeypatch, path):
    app, _ = make_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        login(client)
        assert client.post(path, json={}).status_code == 403
