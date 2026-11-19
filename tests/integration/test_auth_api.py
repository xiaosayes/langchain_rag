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
    settings = AppSettings(
        app_env="test",
        model_provider="fake",
        web=WebSettings(session_ttl_hours=12, secure_cookie=secure_cookie),
    )
    return create_app(settings=settings, qdrant_probe=lambda: {"status": "ok"})


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    app = make_app(tmp_path, monkeypatch)
    app.state.auth_service.create_admin("admin", "correct-password")
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


def test_login_cookie_flags_me_and_logout_revocation(client: TestClient):
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
    assert response.json()["csrf_token"]
    assert "session" not in response.json()
    assert raw_session not in response.text
    assert "correct-password" not in response.text

    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["csrf_token"]
    assert me.json()["expires_at"]
    assert "session" not in me.json()
    assert raw_session not in me.text

    csrf = me.json()["csrf_token"]
    logout = client.post("/api/auth/logout", headers={"X-CSRF-Token": csrf})
    assert logout.status_code == 204
    assert logout.content == b""
    assert client.get("/api/auth/me").status_code == 401


def test_secure_cookie_is_enabled_only_by_configuration(tmp_path: Path, monkeypatch):
    app = make_app(tmp_path / "secure", monkeypatch, secure_cookie=True)
    app.state.auth_service.create_admin("admin", "correct-password")
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


def test_me_rotates_csrf_and_old_token_cannot_authorize_mutation(client: TestClient):
    original = login(client)
    assert original.status_code == 200
    original_csrf = original.json()["csrf_token"]

    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["csrf_token"] != original_csrf
    stale = client.post(
        "/api/auth/logout", headers={"X-CSRF-Token": original_csrf}
    )
    assert stale.status_code == 403
    fresh = client.post(
        "/api/auth/logout", headers={"X-CSRF-Token": me.json()["csrf_token"]}
    )
    assert fresh.status_code == 204


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


@pytest.mark.parametrize("username", ["admin", "unknown-user"])
def test_five_failed_logins_are_rate_limited_without_account_oracle(
    client: TestClient, username: str
):
    responses = [
        client.post(
            "/api/auth/login",
            json={"username": username, "password": "wrong-password"},
        )
        for _ in range(5)
    ]

    assert [response.status_code for response in responses] == [401, 401, 401, 401, 429]
    assert responses[0].json() == {"detail": "Invalid username or password"}
    assert responses[-1].json() == {"detail": "Too many login attempts"}
    for response in responses:
        assert "wrong-password" not in response.text
        assert username not in response.text

    valid_but_limited = login(client)
    assert valid_but_limited.status_code == 429
    assert valid_but_limited.json() == responses[-1].json()
    assert "correct-password" not in valid_but_limited.text
