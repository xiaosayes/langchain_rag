"""FastAPI application composition root."""

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

from fastapi import FastAPI
from qdrant_client import QdrantClient

from app.config import AppSettings, load_settings
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
) -> FastAPI:
    """Compose the web application with injectable settings and Qdrant health."""
    settings = settings or load_settings()
    app = FastAPI(title="Enterprise RAG", version="0.2.0")
    app.state.settings = settings
    app.state.qdrant_probe = qdrant_probe or partial(default_qdrant_probe, settings)
    app.include_router(health_router)
    return app
