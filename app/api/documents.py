"""Authenticated document upload, inspection, download, and mutation routes."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_auth_session, require_csrf
from app.db.models import (
    ChildChunkRecord,
    DocumentRecord,
    IndexGenerationRecord,
    ParentChunkRecord,
)
from app.documents.schemas import (
    DocumentKind,
    DocumentOut,
    DocumentTaskAccepted,
    UploadAccepted,
)
from app.error_sanitization import sanitized_error_message
from app.web.records import DocumentVersionRecord

documents_router = APIRouter(
    prefix="/api/documents",
    tags=["documents"],
    dependencies=[Depends(get_auth_session)],
)


def _sanitized_stored_error(error: str | None) -> str | None:
    if error is None:
        return None
    return sanitized_error_message(RuntimeError(error))


def _document_or_404(session: Session, document_id: str) -> DocumentRecord:
    document = session.get(DocumentRecord, document_id)
    if document is None or document.deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    return document


def _document_out(
    document: DocumentRecord,
    latest_version_id: str | None,
    parent_count: int,
    child_count: int,
) -> DocumentOut:
    return DocumentOut(
        id=document.id,
        filename=document.filename,
        media_type=document.media_type,
        status=document.status,
        current_version_id=document.current_version_id,
        latest_version_id=latest_version_id,
        parent_count=parent_count,
        child_count=child_count,
        source_metadata=dict(document.metadata_json),
        error=_sanitized_stored_error(document.error),
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def _active_counts(
    session: Session, document_ids: list[str]
) -> tuple[dict[str, int], dict[str, int]]:
    if not document_ids:
        return {}, {}
    generation_id = session.scalar(
        select(IndexGenerationRecord.id).where(IndexGenerationRecord.status == "active")
    )
    if generation_id is None:
        return {}, {}
    parent_rows = session.execute(
        select(ParentChunkRecord.document_id, func.count())
        .where(
            ParentChunkRecord.generation_id == generation_id,
            ParentChunkRecord.document_id.in_(document_ids),
        )
        .group_by(ParentChunkRecord.document_id)
    ).all()
    child_rows = session.execute(
        select(ChildChunkRecord.document_id, func.count())
        .where(
            ChildChunkRecord.generation_id == generation_id,
            ChildChunkRecord.document_id.in_(document_ids),
        )
        .group_by(ChildChunkRecord.document_id)
    ).all()
    return dict(parent_rows), dict(child_rows)


def _latest_version_ids(session: Session, document_ids: list[str]) -> dict[str, str]:
    if not document_ids:
        return {}
    versions = session.scalars(
        select(DocumentVersionRecord)
        .where(DocumentVersionRecord.document_id.in_(document_ids))
        .order_by(
            DocumentVersionRecord.document_id,
            DocumentVersionRecord.version.desc(),
            DocumentVersionRecord.id.desc(),
        )
    ).all()
    latest: dict[str, str] = {}
    for version in versions:
        latest.setdefault(version.document_id, version.id)
    return latest


@documents_router.get("", response_model=list[DocumentOut])
def list_documents(request: Request) -> list[DocumentOut]:
    """List public state, deriving chunk counts only from the active generation."""
    service = request.app.state.document_service
    with service.session_factory() as session:
        documents = list(
            session.scalars(
                select(DocumentRecord)
                .where(DocumentRecord.deleted.is_(False))
                .order_by(DocumentRecord.created_at, DocumentRecord.id)
            )
        )
        document_ids = [document.id for document in documents]
        latest_versions = _latest_version_ids(session, document_ids)
        parent_counts, child_counts = _active_counts(session, document_ids)
        return [
            _document_out(
                document,
                latest_versions.get(document.id),
                parent_counts.get(document.id, 0),
                child_counts.get(document.id, 0),
            )
            for document in documents
        ]


@documents_router.post(
    "/upload",
    response_model=UploadAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_csrf)],
)
async def upload_document(
    request: Request,
    file: Annotated[UploadFile, File()],
    kind: Annotated[DocumentKind, Form()] = "document",
) -> UploadAccepted:
    """Stream a validated candidate to immutable storage and enqueue ingestion."""
    try:
        version, task = request.app.state.document_service.upload(
            file.filename or "", file.file, kind
        )
    except ValueError as error:
        message = str(error)
        response_status = (
            status.HTTP_413_CONTENT_TOO_LARGE
            if message.startswith("Upload exceeds")
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=response_status, detail=message) from error
    finally:
        await file.close()
    return UploadAccepted(
        document_id=version.document_id,
        version_id=version.id,
        task_id=task.id,
    )


@documents_router.get("/{document_id}", response_model=DocumentOut)
def get_document(request: Request, document_id: str) -> DocumentOut:
    service = request.app.state.document_service
    with service.session_factory() as session:
        document = _document_or_404(session, document_id)
        latest_versions = _latest_version_ids(session, [document_id])
        parent_counts, child_counts = _active_counts(session, [document_id])
        return _document_out(
            document,
            latest_versions.get(document_id),
            parent_counts.get(document_id, 0),
            child_counts.get(document_id, 0),
        )


@documents_router.get("/{document_id}/content", response_class=FileResponse)
def download_document(request: Request, document_id: str) -> FileResponse:
    """Serve only the accepted ready version, never a pending candidate."""
    service = request.app.state.document_service
    with service.session_factory() as session:
        document = _document_or_404(session, document_id)
        version = (
            session.get(DocumentVersionRecord, document.current_version_id)
            if document.current_version_id is not None
            else None
        )
        if (
            version is None
            or version.document_id != document.id
            or version.status != "ready"
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Accepted document content is not available",
            )
        path = Path(version.path)
        filename = document.filename
        media_type = document.media_type or "application/octet-stream"
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Accepted document content is not available",
        )
    return FileResponse(
        path=path,
        media_type=media_type,
        filename=filename or "document",
        content_disposition_type="attachment",
    )


def _enqueue_document_task(
    request: Request, document_id: str, kind: str
) -> DocumentTaskAccepted:
    service = request.app.state.document_service
    with service.session_factory() as session:
        document = _document_or_404(session, document_id)
        version_id = document.current_version_id
    task = service.queue.enqueue(document_id, version_id, kind)
    return DocumentTaskAccepted(
        document_id=document_id,
        version_id=version_id,
        task_id=task.id,
    )


@documents_router.post(
    "/{document_id}/reindex",
    response_model=DocumentTaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_csrf)],
)
def reindex_document(request: Request, document_id: str) -> DocumentTaskAccepted:
    """Enqueue reindexing without changing the accepted or active generation."""
    return _enqueue_document_task(request, document_id, "reindex")


@documents_router.delete(
    "/{document_id}",
    response_model=DocumentTaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_csrf)],
)
def delete_document(request: Request, document_id: str) -> DocumentTaskAccepted:
    """Enqueue deletion without deleting active content in the request thread."""
    return _enqueue_document_task(request, document_id, "delete")
