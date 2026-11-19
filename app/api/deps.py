"""Authentication and CSRF dependencies for cookie-based API requests."""

import hashlib
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from app.web.auth import AuthService, AuthSession


def get_auth_session(request: Request) -> AuthSession:
    """Resolve the configured HttpOnly cookie to a live persisted session."""
    settings = request.app.state.settings
    raw_token = request.cookies.get(settings.web.session_cookie_name)
    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    service: AuthService = request.app.state.auth_service
    auth_session = service.get_session(raw_token)
    if auth_session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return auth_session


def require_csrf(
    request: Request,
    session: Annotated[AuthSession, Depends(get_auth_session)],
) -> None:
    """Require a header matching the session's stable CSRF token."""
    supplied_token = request.headers.get("X-CSRF-Token", "")
    supplied_hash = hashlib.sha256(supplied_token.encode("utf-8")).hexdigest()
    expected_hash = hashlib.sha256(session.csrf_token.encode("utf-8")).hexdigest()
    if not secrets.compare_digest(supplied_hash, expected_hash):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid CSRF token",
        )
