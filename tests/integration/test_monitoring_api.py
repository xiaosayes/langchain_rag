from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import AppSettings
from app.main import create_app
from app.web.records import ConversationRecord, FeedbackRecord, MessageRecord

_SUMMARY_FIELDS = {
    "hours",
    "total",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "degraded",
    "average_duration_ms",
    "p95_duration_ms",
}
_RUN_FIELDS = {
    "id",
    "conversation_id",
    "message_id",
    "question",
    "status",
    "degraded",
    "created_at",
    "ended_at",
    "duration_ms",
}
_CREDENTIAL_FIELDS = {
    "api_key",
    "authorization",
    "cookie",
    "cookies",
    "csrf",
    "csrf_token",
    "dashscope_api_key",
    "password",
    "password_hash",
    "session",
    "session_id",
    "token",
    "token_hash",
}


def make_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RAG_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "correct-password")
    settings = AppSettings(app_env="test", model_provider="fake")
    return create_app(settings=settings, qdrant_probe=lambda: {"status": "ok"})


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    app = make_app(tmp_path, monkeypatch)
    with TestClient(app) as test_client:
        yield test_client


def login(client: TestClient) -> None:
    response = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "correct-password"},
    )
    assert response.status_code == 200


def seed_runs(client: TestClient) -> tuple[str, str]:
    service = client.app.state.monitoring_service
    succeeded_id = service.start_run("original question", "conversation-1")
    service.set_message_id(succeeded_id, "message-1")
    started_at = datetime(2026, 11, 20, 10, 0, tzinfo=UTC)
    service.record_event(
        succeeded_id,
        "dense",
        "succeeded",
        started_at,
        12.5,
        {"candidates": 2},
    )
    service.update_preparation(
        succeeded_id,
        SimpleNamespace(
            retrieval_query="rewritten question",
            documents=[Document(page_content="retrieved text")],
            degraded=False,
            generation_id="generation-1",
        ),
        [SystemMessage(content="system prompt"), HumanMessage(content="question")],
    )
    service.finish_run(succeeded_id, "succeeded", "final answer")

    failed_id = service.start_run("failed question", "conversation-2")
    service.set_message_id(failed_id, "message-2")
    service.finish_run(failed_id, "failed", "partial answer", "sanitized error")
    return succeeded_id, failed_id


def assert_no_credential_fields(value: Any) -> None:
    if isinstance(value, dict):
        assert _CREDENTIAL_FIELDS.isdisjoint(value)
        for nested in value.values():
            assert_no_credential_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            assert_no_credential_fields(nested)


@pytest.mark.parametrize(
    "path",
    [
        "/api/monitoring/summary?hours=24",
        "/api/monitoring/runs?limit=50&status=&query=",
        "/api/monitoring/runs/run-1",
    ],
)
def test_monitoring_routes_require_authentication(client: TestClient, path: str):
    response = client.get(path)

    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required"}


def test_authenticated_summary_list_filters_and_detail_without_csrf(
    client: TestClient,
):
    succeeded_id, failed_id = seed_runs(client)
    factory = client.app.state.monitoring_service.session_factory
    with factory.begin() as session:
        session.add(ConversationRecord(id="feedback-conversation", title="Feedback"))
        session.add(
            MessageRecord(
                id="feedback-message",
                conversation_id="feedback-conversation",
                role="assistant",
                content="answer",
            )
        )
        session.add(
            FeedbackRecord(id="feedback-1", message_id="feedback-message", score=1)
        )
    login(client)

    # No X-CSRF-Token header is needed for any read-only monitoring request.
    summary_response = client.get("/api/monitoring/summary?hours=24")
    assert summary_response.status_code == 200
    summary = summary_response.json()
    assert set(summary) == _SUMMARY_FIELDS
    assert summary["hours"] == 24
    assert summary["total"] == 2
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert summary["running"] == summary["cancelled"] == summary["degraded"] == 0
    assert summary["average_duration_ms"] >= 0
    assert summary["p95_duration_ms"] >= 0
    assert not {"likes", "dislikes", "like_count", "dislike_count"} & set(summary)

    listed_response = client.get("/api/monitoring/runs?limit=50&status=&query=")
    assert listed_response.status_code == 200
    listed = listed_response.json()
    assert {run["id"] for run in listed} == {succeeded_id, failed_id}
    assert all(set(run) == _RUN_FIELDS for run in listed)

    failed = client.get("/api/monitoring/runs?status=failed").json()
    assert [run["id"] for run in failed] == [failed_id]
    for query in (succeeded_id, "conversation-1", "message-1"):
        filtered = client.get(f"/api/monitoring/runs?query={query}")
        assert filtered.status_code == 200
        assert [run["id"] for run in filtered.json()] == [succeeded_id]

    detail_response = client.get(f"/api/monitoring/runs/{succeeded_id}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert set(detail) == _RUN_FIELDS | {
        "retrieval_query",
        "prompt_messages",
        "retrieved_text",
        "answer",
        "generation_id",
        "error",
        "events",
    }
    assert detail["question"] == "original question"
    assert detail["retrieval_query"] == "rewritten question"
    assert detail["retrieved_text"] == "retrieved text"
    assert detail["prompt_messages"] == [
        {"type": "system", "content": "system prompt"},
        {"type": "human", "content": "question"},
    ]
    assert detail["answer"] == "final answer"
    assert [event["stage"] for event in detail["events"]] == ["dense"]
    assert detail["events"][0]["details"] == {"candidates": 2}

    for payload in (summary, listed, detail):
        assert_no_credential_fields(payload)
        serialized = str(payload).lower()
        assert "like_count" not in serialized
        assert "dislike_count" not in serialized


@pytest.mark.parametrize("hours", [0, -1, 8761, "not-an-integer", "1.5"])
def test_summary_rejects_invalid_hours(client: TestClient, hours: object):
    login(client)

    response = client.get("/api/monitoring/summary", params={"hours": hours})

    assert response.status_code == 422


@pytest.mark.parametrize("hours", [1, 8760])
def test_summary_accepts_hour_bounds(client: TestClient, hours: int):
    login(client)

    response = client.get("/api/monitoring/summary", params={"hours": hours})

    assert response.status_code == 200
    assert response.json()["hours"] == hours


@pytest.mark.parametrize("limit", [0, -1, 101, "not-an-integer", "1.5"])
def test_runs_reject_invalid_limit(client: TestClient, limit: object):
    login(client)

    response = client.get("/api/monitoring/runs", params={"limit": limit})

    assert response.status_code == 422


@pytest.mark.parametrize("status", ["unknown", "RUNNING", " succeeded"])
def test_runs_reject_invalid_status(client: TestClient, status: str):
    login(client)

    response = client.get("/api/monitoring/runs", params={"status": status})

    assert response.status_code == 422


@pytest.mark.parametrize("limit", [1, 100])
def test_runs_accept_limit_bounds_and_default_query_parameters(
    client: TestClient, limit: int
):
    login(client)

    response = client.get("/api/monitoring/runs", params={"limit": limit})

    assert response.status_code == 200
    assert response.json() == []


def test_missing_run_returns_404_after_authentication(client: TestClient):
    login(client)

    response = client.get("/api/monitoring/runs/missing")

    assert response.status_code == 404
    assert response.json() == {"detail": "Monitoring run not found"}
