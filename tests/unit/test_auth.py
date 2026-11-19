import hashlib
import inspect
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from argon2 import PasswordHasher
from sqlalchemy import func, select

from app.db.session import create_session_factory
from app.web.records import AdminRecord, AuthSessionRecord


@pytest.fixture
def session_factory(tmp_path: Path):
    return create_session_factory(tmp_path / "auth.sqlite3")


@pytest.fixture
def service(session_factory):
    from app.web.auth import AuthService

    return AuthService(session_factory, session_ttl_hours=12)


def test_create_admin_uses_argon2id_and_rejects_every_second_admin(service):
    from app.web.auth import DuplicateAdminError

    admin = service.create_admin("admin", "correct-password")

    assert admin.password_hash.startswith("$argon2id$")
    assert "correct-password" not in admin.password_hash
    assert PasswordHasher().verify(admin.password_hash, "correct-password") is True
    with pytest.raises(DuplicateAdminError):
        service.create_admin("admin", "another-password")
    with pytest.raises(DuplicateAdminError):
        service.create_admin("different-name", "another-password")

    with service.session_factory() as session:
        assert session.scalar(select(func.count(AdminRecord.id))) == 1


def test_login_stores_only_sha256_hashes_and_logout_revokes(service):
    service.create_admin("admin", "correct-password")

    raw_session, csrf, expires_at = service.login(
        "admin", "correct-password", "192.0.2.10"
    )

    assert raw_session
    assert csrf
    assert raw_session != csrf
    assert expires_at.tzinfo is not None
    assert expires_at.utcoffset() == timedelta(0)
    with service.session_factory() as session:
        record = session.scalar(select(AuthSessionRecord))
        assert record is not None
        assert record.token_hash == hashlib.sha256(raw_session.encode()).hexdigest()
        assert record.csrf_token == hashlib.sha256(csrf.encode()).hexdigest()
        assert raw_session not in (record.token_hash, record.csrf_token)
        assert csrf not in (record.token_hash, record.csrf_token)

    assert service.get_session(raw_session) is not None
    service.logout(raw_session)
    assert service.get_session(raw_session) is None


def test_expired_session_is_rejected_and_removed(session_factory):
    from app.web.auth import AuthService

    now = datetime(2026, 1, 1, tzinfo=UTC)
    service = AuthService(
        session_factory,
        session_ttl_hours=1,
        clock=lambda: now,
    )
    service.create_admin("admin", "correct-password")
    raw_session, _, _ = service.login("admin", "correct-password", "192.0.2.10")

    now += timedelta(hours=1, seconds=1)

    assert service.get_session(raw_session) is None
    with service.session_factory() as session:
        assert session.scalar(select(AuthSessionRecord)) is None


def test_invalid_credentials_do_not_create_a_session(service):
    from app.web.auth import InvalidCredentialsError

    service.create_admin("admin", "correct-password")

    for username, password in (
        ("admin", "wrong-password"),
        ("unknown", "correct-password"),
    ):
        with pytest.raises(InvalidCredentialsError):
            service.login(username, password, "192.0.2.10")

    with service.session_factory() as session:
        assert session.scalar(select(AuthSessionRecord)) is None


def test_rate_limiter_expires_failures_after_five_minutes(session_factory):
    from app.web.auth import AuthService, InvalidCredentialsError, LoginRateLimitError

    current_time = [1000.0]
    service = AuthService(
        session_factory,
        monotonic=lambda: current_time[0],
    )
    service.create_admin("admin", "correct-password")

    for _ in range(4):
        with pytest.raises(InvalidCredentialsError):
            service.login("admin", "wrong-password", "192.0.2.10")
    with pytest.raises(LoginRateLimitError):
        service.login("admin", "wrong-password", "192.0.2.10")

    current_time[0] += 301
    raw_session, _, _ = service.login(
        "admin", "correct-password", "192.0.2.10"
    )
    assert service.get_session(raw_session) is not None


def test_rate_limit_is_scoped_to_the_source_ip(service):
    from app.web.auth import InvalidCredentialsError, LoginRateLimitError

    service.create_admin("admin", "correct-password")
    for _ in range(5):
        with pytest.raises((InvalidCredentialsError, LoginRateLimitError)):
            service.login("admin", "wrong-password", "192.0.2.10")

    with pytest.raises(LoginRateLimitError):
        service.login("admin", "correct-password", "192.0.2.10")
    raw_session, _, _ = service.login(
        "admin", "correct-password", "192.0.2.11"
    )
    assert service.get_session(raw_session) is not None


def test_rotating_csrf_stores_only_the_new_hash(service):
    service.create_admin("admin", "correct-password")
    raw_session, original_csrf, _ = service.login(
        "admin", "correct-password", "192.0.2.10"
    )
    auth_session = service.get_session(raw_session)
    assert auth_session is not None

    rotated = service.rotate_csrf(auth_session)

    assert rotated.csrf_token != original_csrf
    assert rotated.expires_at == auth_session.expires_at
    with service.session_factory() as session:
        record = session.scalar(select(AuthSessionRecord))
        assert record.csrf_token == hashlib.sha256(
            rotated.csrf_token.encode()
        ).hexdigest()
        assert original_csrf not in record.csrf_token


def test_interactive_bootstrap_uses_getpass_without_password_arguments(
    tmp_path: Path, monkeypatch, capsys
):
    from app import web_admin
    from app.config import AppSettings, WebSettings
    from app.web.records import AdminRecord

    monkeypatch.setenv("RAG_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(
        web_admin,
        "load_settings",
        lambda: AppSettings(app_env="test", model_provider="fake", web=WebSettings()),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "admin")
    prompts = []

    def secret_prompt(prompt: str) -> str:
        prompts.append(prompt)
        return "correct-password"

    monkeypatch.setattr(web_admin.getpass, "getpass", secret_prompt)
    web_admin.main()

    assert len(prompts) == 2
    assert list(inspect.signature(web_admin.main).parameters) == []
    output = capsys.readouterr()
    assert "correct-password" not in output.out
    assert "correct-password" not in output.err
    factory = create_session_factory(tmp_path / "index.sqlite3")
    with factory() as session:
        admin = session.scalar(select(AdminRecord))
        assert admin is not None
        assert admin.username == "admin"
        assert admin.password_hash.startswith("$argon2id$")
        assert "correct-password" not in admin.password_hash
