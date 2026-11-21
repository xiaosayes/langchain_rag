"""Authenticated read-only routes for local chat monitoring."""

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.deps import get_auth_session
from app.monitoring.service import MonitoringService

RunStatus = Literal["running", "succeeded", "failed", "cancelled"]

monitoring_router = APIRouter(
    prefix="/api/monitoring",
    tags=["monitoring"],
    dependencies=[Depends(get_auth_session)],
)


def _service(request: Request) -> MonitoringService:
    return request.app.state.monitoring_service


@monitoring_router.get("/summary")
def summary(
    request: Request,
    hours: Annotated[int, Query(ge=1, le=8760)] = 24,
) -> dict[str, int | float]:
    """Return request counts and latency aggregates for the requested window."""
    return _service(request).summary(hours)


@monitoring_router.get("/runs")
def list_runs(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    status: Annotated[RunStatus | Literal[""] | None, Query()] = None,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """Return newest run summaries with optional exact status/ID filters."""
    return _service(request).list_runs(
        limit=limit, status=status or None, query=query
    )


@monitoring_router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict[str, Any]:
    """Return one complete local trace, or 404 when its ID is unknown."""
    detail = _service(request).get_run(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Monitoring run not found")
    return detail
