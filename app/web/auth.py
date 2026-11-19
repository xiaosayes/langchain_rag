"""Minimal single-account password and opaque-session services."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.web.records import AdminRecord, AuthSessionRecord


class DuplicateAdminError(ValueError):
    """Raised when the configured account does not match the sole admin."""


class InvalidCredentialsError(ValueError):
    """Raised for either an unknown username or an incorrect password."""


@dataclass(frozen=True, slots=True)
class AuthSession:
    """A live session with its stable raw CSRF token and expiry."""

    csrf_token: str
    expires_at: datetime


class AuthService:
    """Persist one environment-provisioned account and hashed login sessions."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        session_ttl_hours: int = 12,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.session_ttl = timedelta(hours=session_ttl_hours)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._password_hasher = PasswordHasher()
        self._dummy_password_hash = self._password_hasher.hash(
            "dummy-password-for-unknown-users"
        )

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def ensure_admin(self, username: str, password: str) -> AdminRecord:
        """Create or verify the sole account without offering account management."""
        username = username.strip()
        if not username:
            raise ValueError("Username must not be empty")
        if not password:
            raise ValueError("Password must not be empty")

        with self.session_factory() as session:
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            existing = session.scalar(select(AdminRecord).limit(1))
            if existing is not None:
                if existing.username != username:
                    raise DuplicateAdminError("The administrator already exists")
                try:
                    self._password_hasher.verify(existing.password_hash, password)
                except (VerificationError, InvalidHashError) as error:
                    raise DuplicateAdminError(
                        "The configured administrator does not match"
                    ) from error
                return existing

            admin = AdminRecord(
                id=uuid.uuid4().hex,
                username=username,
                password_hash=self._password_hasher.hash(password),
            )
            session.add(admin)
            try:
                session.commit()
            except IntegrityError as error:
                session.rollback()
                raise DuplicateAdminError(
                    "The administrator already exists"
                ) from error
            return admin

    def login(self, username: str, password: str) -> tuple[str, str, datetime]:
        """Verify credentials and return raw session/CSRF tokens and expiry."""
        with self.session_factory() as session:
            admin = session.scalar(
                select(AdminRecord).where(AdminRecord.username == username)
            )
            password_hash = (
                admin.password_hash if admin is not None else self._dummy_password_hash
            )
            try:
                self._password_hasher.verify(password_hash, password)
            except (VerificationError, InvalidHashError) as error:
                raise InvalidCredentialsError("Invalid username or password") from error

            session_secret = secrets.token_urlsafe(32)
            csrf_token = secrets.token_urlsafe(32)
            # The opaque cookie payload carries both random values so the stable
            # CSRF token can be recovered after restart without storing it raw.
            raw_session_token = f"{session_secret}.{csrf_token}"
            expires_at = self._clock() + self.session_ttl
            session.add(
                AuthSessionRecord(
                    id=uuid.uuid4().hex,
                    token_hash=self._token_hash(raw_session_token),
                    csrf_token=self._token_hash(csrf_token),
                    expires_at=expires_at,
                )
            )
            session.commit()
            return raw_session_token, csrf_token, expires_at

    def get_session(self, raw_token: str) -> AuthSession | None:
        """Load a live session and recover its stable CSRF value from the cookie."""
        with self.session_factory() as session:
            record = session.scalar(
                select(AuthSessionRecord).where(
                    AuthSessionRecord.token_hash == self._token_hash(raw_token)
                )
            )
            if record is None:
                return None
            if record.expires_at <= self._clock():
                session.delete(record)
                session.commit()
                return None
            _, separator, csrf_token = raw_token.partition(".")
            if not separator or self._token_hash(csrf_token) != record.csrf_token:
                return None
            return AuthSession(
                csrf_token=csrf_token,
                expires_at=record.expires_at,
            )

    def logout(self, raw_token: str) -> None:
        """Revoke a session; unknown or already revoked tokens are harmless."""
        with self.session_factory() as session:
            record = session.scalar(
                select(AuthSessionRecord).where(
                    AuthSessionRecord.token_hash == self._token_hash(raw_token)
                )
            )
            if record is not None:
                session.delete(record)
                session.commit()
