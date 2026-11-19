"""Single-administrator password, session, CSRF, and login-rate services."""

from __future__ import annotations

import hashlib
import secrets
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from threading import Lock

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.web.records import AdminRecord, AuthSessionRecord

_MAX_LOGIN_FAILURES = 5
_LOGIN_WINDOW_SECONDS = 5 * 60


class DuplicateAdminError(ValueError):
    """Raised when bootstrap would create a second administrator."""


class InvalidCredentialsError(ValueError):
    """Raised for either an unknown username or an incorrect password."""


class LoginRateLimitError(ValueError):
    """Raised when an IP has exhausted its login attempts."""


class InvalidSessionError(ValueError):
    """Raised when a previously loaded session can no longer be rotated."""


@dataclass(frozen=True, slots=True)
class AuthSession:
    """An authenticated session and its current CSRF value or verifier.

    ``get_session`` populates ``csrf_token`` with the stored SHA-256 verifier so
    request dependencies never recover a raw CSRF token from the database.
    ``rotate_csrf`` returns a new raw token exactly once for the `/me` response.
    """

    csrf_token: str
    expires_at: datetime
    _token_hash: str = field(default="", repr=False, compare=False)


class AuthService:
    """Persist one admin and opaque, hashed server-side login sessions."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        session_ttl_hours: int = 12,
        *,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.session_factory = session_factory
        self.session_ttl = timedelta(hours=session_ttl_hours)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic
        self._password_hasher = PasswordHasher()
        # Unknown usernames still perform Argon2id verification to avoid a timing
        # oracle. This dummy verifier is never associated with an admin record.
        self._dummy_password_hash = self._password_hasher.hash(
            "dummy-password-for-unknown-users"
        )
        self._failures: dict[str, deque[float]] = {}
        self._rate_lock = Lock()

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create_admin(self, username: str, password: str) -> AdminRecord:
        """Create the sole administrator, rejecting same-name and new-name duplicates."""
        username = username.strip()
        if not username:
            raise ValueError("Username must not be empty")
        if not password:
            raise ValueError("Password must not be empty")

        with self.session_factory() as session:
            # Serialize SQLite bootstraps so concurrent commands cannot both pass
            # the singleton check before either one commits.
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            if session.scalar(select(func.count(AdminRecord.id))):
                raise DuplicateAdminError("The administrator already exists")
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

    def _prune_failures(self, ip_address: str, now: float) -> deque[float]:
        failures = self._failures.setdefault(ip_address, deque())
        cutoff = now - _LOGIN_WINDOW_SECONDS
        while failures and failures[0] <= cutoff:
            failures.popleft()
        return failures

    def _is_rate_limited(self, ip_address: str) -> bool:
        with self._rate_lock:
            failures = self._prune_failures(ip_address, self._monotonic())
            return len(failures) >= _MAX_LOGIN_FAILURES

    def _record_failure(self, ip_address: str) -> bool:
        with self._rate_lock:
            failures = self._prune_failures(ip_address, self._monotonic())
            failures.append(self._monotonic())
            return len(failures) >= _MAX_LOGIN_FAILURES

    def _clear_failures(self, ip_address: str) -> None:
        with self._rate_lock:
            self._failures.pop(ip_address, None)

    def login(
        self, username: str, password: str, ip_address: str
    ) -> tuple[str, str, datetime]:
        """Verify credentials and return raw session/CSRF tokens and expiry."""
        if self._is_rate_limited(ip_address):
            raise LoginRateLimitError("Too many login attempts")

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
                limited = self._record_failure(ip_address)
                if limited:
                    raise LoginRateLimitError("Too many login attempts") from error
                raise InvalidCredentialsError(
                    "Invalid username or password"
                ) from error

            self._clear_failures(ip_address)
            raw_session_token = secrets.token_urlsafe(32)
            csrf_token = secrets.token_urlsafe(32)
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
        """Load a live session by hashed token and remove it if expired."""
        token_hash = self._token_hash(raw_token)
        with self.session_factory() as session:
            record = session.scalar(
                select(AuthSessionRecord).where(
                    AuthSessionRecord.token_hash == token_hash
                )
            )
            if record is None:
                return None
            if record.expires_at <= self._clock():
                session.delete(record)
                session.commit()
                return None
            return AuthSession(
                csrf_token=record.csrf_token,
                expires_at=record.expires_at,
                _token_hash=record.token_hash,
            )

    def rotate_csrf(self, auth_session: AuthSession) -> AuthSession:
        """Issue a raw CSRF token for `/me` while storing only its new hash."""
        csrf_token = secrets.token_urlsafe(32)
        csrf_hash = self._token_hash(csrf_token)
        with self.session_factory() as session:
            record = session.scalar(
                select(AuthSessionRecord).where(
                    AuthSessionRecord.token_hash == auth_session._token_hash
                )
            )
            if record is None or record.expires_at <= self._clock():
                if record is not None:
                    session.delete(record)
                    session.commit()
                raise InvalidSessionError("Session is no longer valid")
            record.csrf_token = csrf_hash
            session.commit()
            return AuthSession(
                csrf_token=csrf_token,
                expires_at=record.expires_at,
                _token_hash=record.token_hash,
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
