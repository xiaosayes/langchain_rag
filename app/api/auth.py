"""Login, current-session, and logout endpoints."""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, SecretStr

from app.api.deps import get_auth_session, require_csrf
from app.web.auth import AuthService, AuthSession, InvalidCredentialsError

auth_router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: SecretStr


def _session_payload(csrf_token: str, expires_at: datetime) -> dict[str, str]:
    return {"csrf_token": csrf_token, "expires_at": expires_at.isoformat()}


@auth_router.post("/login")
def login(credentials: LoginRequest, request: Request, response: Response):
    """Authenticate without revealing which credential failed."""
    service: AuthService = request.app.state.auth_service
    try:
        raw_session, csrf_token, expires_at = service.login(
            credentials.username,
            credentials.password.get_secret_value(),
        )
    except InvalidCredentialsError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        ) from error

    settings = request.app.state.settings.web
    response.set_cookie(
        key=settings.session_cookie_name,
        value=raw_session,
        max_age=int(service.session_ttl.total_seconds()),
        path="/",
        secure=settings.secure_cookie,
        httponly=True,
        samesite="lax",
    )
    return _session_payload(csrf_token, expires_at)


@auth_router.get("/me")
def me(session: Annotated[AuthSession, Depends(get_auth_session)]):
    """Return the stable CSRF token and expiry, never the session token."""
    return _session_payload(session.csrf_token, session.expires_at)


@auth_router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_csrf)],
)
def logout(request: Request) -> Response:
    """Revoke the server-side session and expire only its browser cookie."""
    service: AuthService = request.app.state.auth_service
    settings = request.app.state.settings.web
    raw_session = request.cookies.get(settings.session_cookie_name, "")
    service.logout(raw_session)
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    response.delete_cookie(
        key=settings.session_cookie_name,
        path="/",
        secure=settings.secure_cookie,
        httponly=True,
        samesite="lax",
    )
    return response
