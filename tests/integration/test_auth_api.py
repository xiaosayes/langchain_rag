from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api import deps as auth_deps
from app.config import AppSettings, WebSettings
from app.db.session import create_session_factory
from app.main import create_app
from app.web.records import AuthSessionRecord


def make_app(tmp_path: Path, monkeypatch, *, secure_cookie: bool = False):
    monkeypatch.setenv("RAG_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "correct-password")
    settings = AppSettings(
        app_env="test",
        model_provider="fake",
        web=WebSettings(session_ttl_hours=12, secure_cookie=secure_cookie),
    )
    return create_app(settings=settings, qdrant_probe=lambda: {"status": "ok"})


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    app = make_app(tmp_path, monkeypatch)
    with TestClient(app) as test_client:
        yield test_client


def login(client: TestClient):
    return client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "correct-password"},
    )


def cookie_attributes(response) -> set[str]:
    return {
        part.strip().split("=", 1)[0].lower()
        for part in response.headers["set-cookie"].split(";")
    }


def test_login_cookie_flags_static_csrf_me_and_logout_revocation(client: TestClient):
    response = login(client)

    assert response.status_code == 200
    raw_session = response.cookies["rag_session"]
    assert raw_session
    attributes = cookie_attributes(response)
    assert {"httponly", "samesite", "path", "max-age"} <= attributes
    assert "secure" not in attributes
    set_cookie = response.headers["set-cookie"].lower()
    assert "samesite=lax" in set_cookie
    assert "path=/" in set_cookie
    assert "max-age=43200" in set_cookie
    csrf = response.json()["csrf_token"]
    assert csrf
    assert "session" not in response.json()
    assert raw_session not in response.text
    assert "correct-password" not in response.text

    first_me = client.get("/api/auth/me")
    second_me = client.get("/api/auth/me")
    assert first_me.status_code == 200
    assert second_me.status_code == 200
    assert first_me.json()["csrf_token"] == csrf
    assert second_me.json()["csrf_token"] == csrf
    assert first_me.json()["expires_at"] == second_me.json()["expires_at"]
    assert "session" not in first_me.json()
    assert raw_session not in first_me.text

    logout = client.post("/api/auth/logout", headers={"X-CSRF-Token": csrf})
    assert logout.status_code == 204
    assert logout.content == b""
    assert client.get("/api/auth/me").status_code == 401


def test_secure_cookie_is_enabled_only_by_configuration(tmp_path: Path, monkeypatch):
    app = make_app(tmp_path / "secure", monkeypatch, secure_cookie=True)
    with TestClient(app) as secure_client:
        response = login(secure_client)
        assert response.status_code == 200
        assert "secure" in cookie_attributes(response)
        assert "httponly" in cookie_attributes(response)
        assert "samesite=lax" in response.headers["set-cookie"].lower()


def test_me_rejects_missing_and_unknown_session(client: TestClient):
    assert client.get("/api/auth/me").status_code == 401

    client.cookies.set("rag_session", "not-a-real-session", path="/")
    response = client.get("/api/auth/me")
    assert response.status_code == 401
    assert "not-a-real-session" not in response.text


@pytest.mark.parametrize("headers", [{}, {"X-CSRF-Token": "wrong-token"}])
def test_logout_requires_a_valid_csrf_token(
    client: TestClient, headers: dict, monkeypatch
):
    compare_calls = []
    original_compare_digest = auth_deps.secrets.compare_digest

    def recording_compare_digest(left, right):
        compare_calls.append((left, right))
        return original_compare_digest(left, right)

    monkeypatch.setattr(auth_deps.secrets, "compare_digest", recording_compare_digest)
    response = login(client)
    assert response.status_code == 200

    rejected = client.post("/api/auth/logout", headers=headers)

    assert rejected.status_code == 403
    assert len(compare_calls) == 1
    assert rejected.json() == {"detail": "Invalid CSRF token"}
    assert "wrong-token" not in rejected.text
    assert client.get("/api/auth/me").status_code == 200


def test_repeated_me_and_two_tabs_share_stable_csrf(tmp_path: Path, monkeypatch):
    app = make_app(tmp_path, monkeypatch)
    with TestClient(app) as first_tab, TestClient(app) as second_tab:
        response = login(first_tab)
        assert response.status_code == 200
        csrf = response.json()["csrf_token"]
        second_tab.cookies.set(
            "rag_session", response.cookies["rag_session"], path="/"
        )

        first = first_tab.get("/api/auth/me")
        repeated = first_tab.get("/api/auth/me")
        second = second_tab.get("/api/auth/me")

        assert first.status_code == repeated.status_code == second.status_code == 200
        assert first.json()["csrf_token"] == csrf
        assert repeated.json()["csrf_token"] == csrf
        assert second.json()["csrf_token"] == csrf
        logout = second_tab.post(
            "/api/auth/logout", headers={"X-CSRF-Token": csrf}
        )
        assert logout.status_code == 204
        assert first_tab.get("/api/auth/me").status_code == 401


def test_expired_session_is_unauthenticated(client: TestClient, tmp_path: Path):
    response = login(client)
    assert response.status_code == 200
    factory = create_session_factory(tmp_path / "index.sqlite3")
    with factory() as session:
        record = session.scalar(select(AuthSessionRecord))
        assert record is not None
        record.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        session.commit()

    assert client.get("/api/auth/me").status_code == 401


def test_logout_requires_authentication_before_csrf(client: TestClient):
    response = client.post(
        "/api/auth/logout", headers={"X-CSRF-Token": "anything"}
    )
    assert response.status_code == 401
