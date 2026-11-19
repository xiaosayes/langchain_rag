"""Authenticated read-only task status routes."""

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.deps import get_auth_session
from app.documents.schemas import TaskOut
from app.error_sanitization import sanitized_error_message
from app.web.records import TaskRecord

tasks_router = APIRouter(
    prefix="/api/tasks",
    tags=["tasks"],
    dependencies=[Depends(get_auth_session)],
)


@tasks_router.get("/{task_id}", response_model=TaskOut)
def get_task(request: Request, task_id: str) -> TaskOut:
    """Return queue progress without leases, payloads, or storage details."""
    service = request.app.state.document_service
    with service.session_factory() as session:
        task = session.get(TaskRecord, task_id)
        if task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found",
            )
        error = (
            sanitized_error_message(RuntimeError(task.error))
            if task.error is not None
            else None
        )
        return TaskOut(
            id=task.id,
            kind=task.kind,
            status=task.status,
            attempts=task.attempts,
            error=error,
        )
