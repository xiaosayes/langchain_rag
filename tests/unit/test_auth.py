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


def test_ensure_admin_uses_argon2id_and_rejects_every_second_admin(service):
    from app.web.auth import DuplicateAdminError

    admin = service.ensure_admin("admin", "correct-password")
    same_admin = service.ensure_admin("admin", "correct-password")

    assert same_admin.id == admin.id
    assert admin.password_hash.startswith("$argon2id$")
    assert "correct-password" not in admin.password_hash
    assert PasswordHasher().verify(admin.password_hash, "correct-password") is True
    with pytest.raises(DuplicateAdminError):
        service.ensure_admin("admin", "another-password")
    with pytest.raises(DuplicateAdminError):
        service.ensure_admin("different-name", "another-password")

    with service.session_factory() as session:
        assert session.scalar(select(func.count(AdminRecord.id))) == 1


def test_login_stores_only_sha256_hashes_and_logout_revokes(service):
    service.ensure_admin("admin", "correct-password")

    raw_session, csrf, expires_at = service.login("admin", "correct-password")

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

    auth_session = service.get_session(raw_session)
    assert auth_session is not None
    assert auth_session.csrf_token == csrf
    service.logout(raw_session)
    assert service.get_session(raw_session) is None


def test_csrf_is_stable_across_service_restarts(session_factory, service):
    from app.web.auth import AuthService

    service.ensure_admin("admin", "correct-password")
    raw_session, csrf, _ = service.login("admin", "correct-password")

    restarted_service = AuthService(session_factory)
    auth_session = restarted_service.get_session(raw_session)

    assert auth_session is not None
    assert auth_session.csrf_token == csrf


def test_expired_session_is_rejected_and_removed(session_factory):
    from app.web.auth import AuthService

    now = datetime(2026, 1, 1, tzinfo=UTC)
    service = AuthService(
        session_factory,
        session_ttl_hours=1,
        clock=lambda: now,
    )
    service.ensure_admin("admin", "correct-password")
    raw_session, _, _ = service.login("admin", "correct-password")

    now += timedelta(hours=1, seconds=1)

    assert service.get_session(raw_session) is None
    with service.session_factory() as session:
        assert session.scalar(select(AuthSessionRecord)) is None


def test_invalid_credentials_do_not_create_a_session(service):
    from app.web.auth import InvalidCredentialsError

    service.ensure_admin("admin", "correct-password")

    for username, password in (
        ("admin", "wrong-password"),
        ("unknown", "correct-password"),
    ):
        with pytest.raises(InvalidCredentialsError):
            service.login(username, password)

    with service.session_factory() as session:
        assert session.scalar(select(AuthSessionRecord)) is None


def test_excluded_login_features_are_absent(service):
    from app.web import auth

    assert list(inspect.signature(service.login).parameters) == [
        "username",
        "password"
    ]
    assert not hasattr(auth, "LoginRateLimitError")
    assert not hasattr(service, "_failures")
    assert not hasattr(service, "rotate_csrf")
    assert not hasattr(service, "create_admin")
    assert not (Path(auth.__file__).parent.parent / "web_admin.py").exists()
