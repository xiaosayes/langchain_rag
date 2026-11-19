from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.config import AppSettings
from app.main import create_app
from app.web.records import FeedbackRecord, MessageRecord


def make_app(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAG_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "correct-password")
    settings = AppSettings(app_env="test", model_provider="fake")
    return create_app(settings=settings, qdrant_probe=lambda: {"status": "ok"})


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    app = make_app(tmp_path, monkeypatch)
    with TestClient(app) as test_client:
        yield test_client


def login(client: TestClient) -> str:
    response = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "correct-password"},
    )
    assert response.status_code == 200
    return response.json()["csrf_token"]


def create_conversation(client: TestClient, csrf: str, title: str | None = None):
    payload = {} if title is None else {"title": title}
    return client.post("/api/sessions", json=payload, headers={"X-CSRF-Token": csrf})


def test_create_list_rename_delete_and_shared_admin_history(client: TestClient):
    csrf = login(client)

    created = create_conversation(client, csrf)
    assert created.status_code == 200
    conversation = created.json()
    assert set(conversation) == {"id", "title", "created_at", "updated_at"}
    assert conversation["id"]
    assert conversation["title"] == "新会话"
    assert datetime.fromisoformat(conversation["created_at"]).tzinfo == UTC
    assert datetime.fromisoformat(conversation["updated_at"]).tzinfo == UTC

    other = create_conversation(client, csrf, "Project notes")
    assert other.status_code == 200

    renamed = client.patch(
        f"/api/sessions/{conversation['id']}",
        json={"title": "Renamed conversation"},
        headers={"X-CSRF-Token": csrf},
    )
    assert renamed.status_code == 200
    assert renamed.json()["id"] == conversation["id"]
    assert renamed.json()["title"] == "Renamed conversation"
    assert renamed.json()["created_at"] == conversation["created_at"]

    # A separate login for the same sole administrator sees the same records.
    second_csrf = login(client)
    assert second_csrf != csrf
    listed = client.get("/api/sessions")
    assert listed.status_code == 200
    assert {item["id"] for item in listed.json()} == {
        conversation["id"],
        other.json()["id"],
    }
    assert {item["title"] for item in listed.json()} == {
        "Renamed conversation",
        "Project notes",
    }

    deleted = client.delete(
        f"/api/sessions/{conversation['id']}",
        headers={"X-CSRF-Token": second_csrf},
    )
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert [item["id"] for item in client.get("/api/sessions").json()] == [
        other.json()["id"]
    ]
    assert client.get(f"/api/sessions/{conversation['id']}/messages").status_code == 404


def test_titles_default_for_null_and_are_limited_to_200_characters(
    client: TestClient,
):
    csrf = login(client)

    null_title = client.post(
        "/api/sessions",
        json={"title": None},
        headers={"X-CSRF-Token": csrf},
    )
    assert null_title.status_code == 200
    assert null_title.json()["title"] == "新会话"

    maximum_title = "题" * 200
    created = create_conversation(client, csrf, maximum_title)
    assert created.status_code == 200
    assert created.json()["title"] == maximum_title

    assert create_conversation(client, csrf, "x" * 201).status_code == 422
    assert (
        client.patch(
            f"/api/sessions/{created.json()['id']}",
            json={"title": "x" * 201},
            headers={"X-CSRF-Token": csrf},
        ).status_code
        == 422
    )


def test_display_messages_and_latest_history_are_chronological(client: TestClient):
    csrf = login(client)
    conversation_id = create_conversation(client, csrf).json()["id"]
    factory = client.app.state.conversation_service.session_factory
    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    with factory.begin() as session:
        for index in range(8):
            session.add(
                MessageRecord(
                    id=f"message-{index}",
                    conversation_id=conversation_id,
                    role="user" if index % 2 == 0 else "assistant",
                    content=f"content-{index}",
                    citations_json=(
                        [{"source": "notes.txt", "chunk_id": "chunk-0"}]
                        if index == 0
                        else []
                    ),
                    degraded=index == 1,
                    generation_id=None,
                    created_at=base_time + timedelta(minutes=index),
                )
            )

    response = client.get(f"/api/sessions/{conversation_id}/messages")

    assert response.status_code == 200
    displayed = response.json()
    assert [message["content"] for message in displayed] == [
        f"content-{index}" for index in range(8)
    ]
    assert set(displayed[0]) == {
        "id",
        "role",
        "content",
        "citations",
        "degraded",
        "generation_id",
        "created_at",
    }
    assert displayed[0]["citations"] == [{"source": "notes.txt", "chunk_id": "chunk-0"}]
    assert displayed[0]["degraded"] is False
    assert displayed[1]["degraded"] is True
    assert displayed[0]["generation_id"] is None
    assert "cancelled" not in displayed[0]
    assert "error" not in displayed[0]

    service = client.app.state.conversation_service
    assert [message.content for message in service.messages(conversation_id)] == [
        f"content-{index}" for index in range(2, 8)
    ]
    assert [
        message.content for message in service.messages(conversation_id, limit=3)
    ] == [f"content-{index}" for index in range(5, 8)]
    assert [message.content for message in service.list_messages(conversation_id)] == [
        f"content-{index}" for index in range(8)
    ]


def test_delete_cascades_messages_and_feedback(client: TestClient):
    csrf = login(client)
    conversation_id = create_conversation(client, csrf).json()["id"]
    factory = client.app.state.conversation_service.session_factory
    with factory.begin() as session:
        for index in range(2):
            message_id = f"message-{index}"
            session.add(
                MessageRecord(
                    id=message_id,
                    conversation_id=conversation_id,
                    role="assistant",
                    content=f"answer-{index}",
                )
            )
            session.add(
                FeedbackRecord(
                    id=f"feedback-{index}",
                    message_id=message_id,
                    score=1 if index == 0 else -1,
                    comment="useful" if index == 0 else "not useful",
                )
            )

    response = client.delete(
        f"/api/sessions/{conversation_id}", headers={"X-CSRF-Token": csrf}
    )

    assert response.status_code == 204
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(MessageRecord)) == 0
        assert session.scalar(select(func.count()).select_from(FeedbackRecord)) == 0


def test_unknown_conversations_return_404(client: TestClient):
    csrf = login(client)

    assert (
        client.patch(
            "/api/sessions/missing",
            json={"title": "new title"},
            headers={"X-CSRF-Token": csrf},
        ).status_code
        == 404
    )
    assert (
        client.delete(
            "/api/sessions/missing", headers={"X-CSRF-Token": csrf}
        ).status_code
        == 404
    )
    assert client.get("/api/sessions/missing/messages").status_code == 404


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/sessions"),
        ("POST", "/api/sessions"),
        ("PATCH", "/api/sessions/missing"),
        ("DELETE", "/api/sessions/missing"),
        ("GET", "/api/sessions/missing/messages"),
    ],
)
def test_all_conversation_routes_require_authentication(
    client: TestClient, method: str, path: str
):
    response = client.request(method, path, json={"title": "title"})

    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required"}


@pytest.mark.parametrize("headers", [{}, {"X-CSRF-Token": "wrong-token"}])
def test_conversation_mutations_require_csrf(client: TestClient, headers: dict):
    csrf = login(client)

    rejected_create = client.post("/api/sessions", json={}, headers=headers)
    assert rejected_create.status_code == 403
    assert rejected_create.json() == {"detail": "Invalid CSRF token"}

    conversation_id = create_conversation(client, csrf).json()["id"]
    rejected_rename = client.patch(
        f"/api/sessions/{conversation_id}", json={"title": "new"}, headers=headers
    )
    rejected_delete = client.delete(f"/api/sessions/{conversation_id}", headers=headers)
    assert rejected_rename.status_code == rejected_delete.status_code == 403
    assert client.get(f"/api/sessions/{conversation_id}/messages").status_code == 200
