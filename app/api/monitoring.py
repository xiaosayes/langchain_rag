"""Authenticated read-only routes for local chat monitoring."""

import logging
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.deps import get_auth_session
from app.monitoring.service import MonitoringService

RunStatus = Literal["running", "succeeded", "failed", "cancelled"]

logger = logging.getLogger(__name__)

monitoring_router = APIRouter(
    prefix="/api/monitoring",
    tags=["monitoring"],
    dependencies=[Depends(get_auth_session)],
)


def _service(request: Request) -> MonitoringService:
    return request.app.state.monitoring_service


def _storage_unavailable(error: Exception) -> HTTPException:
    logger.warning("Monitoring query failed (%s)", type(error).__name__)
    return HTTPException(status_code=503, detail="Monitoring storage unavailable")


@monitoring_router.get("/summary")
def summary(
    request: Request,
    hours: Annotated[int, Query(ge=1, le=8760)] = 24,
) -> dict[str, int | float]:
    """Return request counts and latency aggregates for the requested window."""
    try:
        return _service(request).summary_strict(hours)
    except Exception as error:
        raise _storage_unavailable(error) from error


@monitoring_router.get("/runs")
def list_runs(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    status: Annotated[RunStatus | Literal[""] | None, Query()] = None,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """Return newest run summaries with optional exact status/ID filters."""
    try:
        return _service(request).list_runs_strict(
            limit=limit, status=status or None, query=query
        )
    except Exception as error:
        raise _storage_unavailable(error) from error


@monitoring_router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict[str, Any]:
    """Return one complete local trace, or 404 when its ID is unknown."""
    try:
        detail = _service(request).get_run_strict(run_id)
    except Exception as error:
        raise _storage_unavailable(error) from error
    if detail is None:
        raise HTTPException(status_code=404, detail="Monitoring run not found")
    return detail
