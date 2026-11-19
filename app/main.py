"""FastAPI application composition root."""

import os
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

from fastapi import FastAPI
from qdrant_client import QdrantClient

from app.api.auth import auth_router
from app.api.documents import documents_router
from app.api.tasks import tasks_router
from app.bootstrap import state_directory
from app.config import AppSettings, load_settings
from app.db.session import create_session_factory
from app.documents.service import DocumentService
from app.web.auth import AuthService
from app.web.health import health_router


def default_qdrant_probe(settings: AppSettings) -> Mapping[str, Any]:
    """Check Qdrant connectivity without requiring an active index generation."""
    client = QdrantClient(url=settings.qdrant.url)
    try:
        client.get_collections()
    finally:
        client.close()
    return {"status": "ok"}


def create_app(
    settings: AppSettings | None = None,
    qdrant_probe: Callable[[], Mapping[str, Any]] | None = None,
    auth_service: AuthService | None = None,
) -> FastAPI:
    """Compose the web application with injectable settings and Qdrant health."""
    settings = settings or load_settings()
    app = FastAPI(title="Enterprise RAG", version="0.2.0")
    app.state.settings = settings
    app.state.qdrant_probe = qdrant_probe or partial(default_qdrant_probe, settings)
    if auth_service is None:
        root = state_directory()
        root.mkdir(parents=True, exist_ok=True)
        auth_service = AuthService(
            create_session_factory(root / "index.sqlite3"),
            session_ttl_hours=settings.web.session_ttl_hours,
        )
    admin_username = os.environ.get("ADMIN_USERNAME")
    admin_password = os.environ.get("ADMIN_PASSWORD")
    if (admin_username is None) != (admin_password is None):
        raise ValueError("ADMIN_USERNAME and ADMIN_PASSWORD must be set together")
    if admin_username is not None and admin_password is not None:
        auth_service.ensure_admin(admin_username, admin_password)
    app.state.auth_service = auth_service
    app.state.document_service = DocumentService(auth_service.session_factory, settings)
    app.include_router(auth_router)
    app.include_router(documents_router)
    app.include_router(tasks_router)
    app.include_router(health_router)
    return app
