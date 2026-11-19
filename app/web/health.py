"""Process liveness and dependency readiness endpoints."""

from collections.abc import Callable, Mapping
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.bootstrap import state_directory
from app.db.session import create_session_factory

health_router = APIRouter()


@health_router.get("/health/live")
def liveness() -> dict[str, str]:
    """Report process liveness without touching external dependencies."""
    return {"status": "ok"}


@health_router.get("/health/ready")
def readiness(request: Request) -> JSONResponse:
    """Check SQLite and Qdrant while returning only sanitized status values."""
    database_status = "ok"
    qdrant_status = "ok"

    try:
        root = state_directory()
        root.mkdir(parents=True, exist_ok=True)
        session_factory = create_session_factory(root / "index.sqlite3")
        with session_factory() as session:
            session.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 - readiness sanitizes dependency failures.
        database_status = "error"

    try:
        probe: Callable[[], Mapping[str, Any]] = request.app.state.qdrant_probe
        if probe().get("status") != "ok":
            qdrant_status = "error"
    except Exception:  # noqa: BLE001 - readiness sanitizes dependency failures.
        qdrant_status = "error"

    ready = database_status == "ok" and qdrant_status == "ok"
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ok" if ready else "error",
            "database": database_status,
            "qdrant": qdrant_status,
        },
    )
