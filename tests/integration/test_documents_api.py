from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import AppSettings, WebSettings
from app.db.models import DocumentRecord, IndexGenerationRecord
from app.domain.models import ChildChunk, ParentChunk
from app.indexing.repository import IndexRepository
from app.main import create_app
from app.web.records import DocumentVersionRecord, TaskRecord


def make_app(tmp_path: Path, monkeypatch, *, max_upload_mb: int = 1):
    monkeypatch.setenv("RAG_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "correct-password")
    settings = AppSettings(
        app_env="test",
        model_provider="fake",
        web=WebSettings(
            raw_file_dir=tmp_path / "raw",
            max_upload_mb=max_upload_mb,
        ),
    )
    return create_app(settings=settings, qdrant_probe=lambda: {"status": "ok"})


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    app = make_app(tmp_path, monkeypatch)
    with TestClient(app) as test_client:
        yield test_client


def login(client: TestClient) -> str:
    response = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "correct-password"},
    )
    assert response.status_code == 200
    return response.json()["csrf_token"]


def upload(client: TestClient, csrf: str, content: bytes = b"accepted text"):
    return client.post(
        "/api/documents/upload",
        headers={"X-CSRF-Token": csrf},
        data={"kind": "document"},
        files={"file": ("notes.txt", content, "text/plain")},
    )


def activate_upload(client: TestClient, upload_payload: dict) -> str:
    """Simulate worker publication with real generation-scoped SQLite records."""
    factory = client.app.state.document_service.session_factory
    document_id = upload_payload["document_id"]
    version_id = upload_payload["version_id"]
    task_id = upload_payload["task_id"]
    generation_id = f"generation-{document_id}"
    parents = [
        ParentChunk(
            id=f"parent-{ordinal}",
            document_id=document_id,
            ordinal=ordinal,
            text=f"parent text {ordinal}",
            content_hash=f"parent-hash-{ordinal}",
            metadata={"source": "notes.txt"},
        )
        for ordinal in range(2)
    ]
    children = [
        ChildChunk(
            id=f"child-{ordinal}",
            parent_id=f"parent-{ordinal // 2}",
            document_id=document_id,
            ordinal=ordinal,
            text=f"child text {ordinal}",
            content_hash=f"child-hash-{ordinal}",
            metadata={},
        )
        for ordinal in range(4)
    ]
    repository = IndexRepository(factory)
    repository.prepare_generation(generation_id)
    repository.activate_generation(generation_id, parents, children)
    with factory.begin() as session:
        document = session.get(DocumentRecord, document_id)
        version = session.get(DocumentVersionRecord, version_id)
        task = session.get(TaskRecord, task_id)
        assert document is not None
        assert version is not None
        assert task is not None
        document.current_version_id = version_id
        document.status = "ready"
        version.status = "ready"
        task.status = "succeeded"
    return generation_id


def test_upload_list_detail_download_reindex_delete_and_task_status(
    client: TestClient, tmp_path: Path
):
    csrf = login(client)

    accepted = upload(client, csrf)

    assert accepted.status_code == 202
    uploaded = accepted.json()
    assert set(uploaded) == {"document_id", "version_id", "task_id"}
    assert all(uploaded.values())
    assert str(tmp_path) not in accepted.text

    pending_list = client.get("/api/documents")
    pending_detail = client.get(f"/api/documents/{uploaded['document_id']}")
    assert pending_list.status_code == pending_detail.status_code == 200
    assert len(pending_list.json()) == 1
    pending = pending_detail.json()
    assert pending_list.json()[0] == pending
    assert pending == {
        "id": uploaded["document_id"],
        "filename": "notes.txt",
        "media_type": "text/plain",
        "status": "uploaded",
        "current_version_id": None,
        "latest_version_id": uploaded["version_id"],
        "parent_count": 0,
        "child_count": 0,
        "source_metadata": {"kind": "document"},
        "error": None,
        "created_at": pending["created_at"],
        "updated_at": pending["updated_at"],
    }
    created_at = datetime.fromisoformat(pending["created_at"])
    assert created_at.tzinfo == UTC

    pending_download = client.get(f"/api/documents/{uploaded['document_id']}/content")
    assert pending_download.status_code == 409
    assert str(tmp_path) not in pending_download.text

    queued_task = client.get(f"/api/tasks/{uploaded['task_id']}")
    assert queued_task.status_code == 200
    assert queued_task.json() == {
        "id": uploaded["task_id"],
        "kind": "document",
        "status": "queued",
        "attempts": 0,
        "error": None,
    }

    generation_id = activate_upload(client, uploaded)
    detail = client.get(f"/api/documents/{uploaded['document_id']}")
    assert detail.status_code == 200
    assert detail.json()["current_version_id"] == uploaded["version_id"]
    assert detail.json()["latest_version_id"] == uploaded["version_id"]
    assert detail.json()["parent_count"] == 2
    assert detail.json()["child_count"] == 4
    assert client.get("/api/documents").json()[0]["parent_count"] == 2

    download = client.get(f"/api/documents/{uploaded['document_id']}/content")
    assert download.status_code == 200
    assert download.content == b"accepted text"
    assert download.headers["content-type"].startswith("text/plain")
    assert download.headers["content-disposition"] == 'attachment; filename="notes.txt"'
    assert str(tmp_path) not in download.headers["content-disposition"]

    reindexed = client.post(
        f"/api/documents/{uploaded['document_id']}/reindex",
        headers={"X-CSRF-Token": csrf},
    )
    assert reindexed.status_code == 202
    assert reindexed.json() == {
        "document_id": uploaded["document_id"],
        "version_id": uploaded["version_id"],
        "task_id": reindexed.json()["task_id"],
    }
    reindex_task = client.get(f"/api/tasks/{reindexed.json()['task_id']}")
    assert reindex_task.status_code == 200
    assert reindex_task.json()["kind"] == "reindex"
    assert reindex_task.json()["status"] == "queued"

    deleted = client.delete(
        f"/api/documents/{uploaded['document_id']}",
        headers={"X-CSRF-Token": csrf},
    )
    assert deleted.status_code == 202
    assert deleted.json() == {
        "document_id": uploaded["document_id"],
        "version_id": uploaded["version_id"],
        "task_id": deleted.json()["task_id"],
    }
    delete_task = client.get(f"/api/tasks/{deleted.json()['task_id']}")
    assert delete_task.status_code == 200
    assert delete_task.json()["kind"] == "delete"
    assert delete_task.json()["status"] == "queued"

    # Both operations only enqueue work: accepted content and active counts remain.
    after_delete = client.get(f"/api/documents/{uploaded['document_id']}")
    assert after_delete.status_code == 200
    assert after_delete.json()["current_version_id"] == uploaded["version_id"]
    assert after_delete.json()["parent_count"] == 2
    assert (
        client.get(f"/api/documents/{uploaded['document_id']}/content").content
        == b"accepted text"
    )
    factory = client.app.state.document_service.session_factory
    with factory() as session:
        generation = session.get(IndexGenerationRecord, generation_id)
        document = session.get(DocumentRecord, uploaded["document_id"])
        assert generation is not None
        assert generation.status == "active"
        assert document is not None
        assert document.deleted is False


def test_pending_replacement_keeps_active_generation_counts_and_download(
    client: TestClient,
):
    csrf = login(client)
    first = upload(client, csrf, b"accepted old").json()
    generation_id = activate_upload(client, first)

    replacement = upload(client, csrf, b"pending replacement")
    assert replacement.status_code == 202
    replacement_payload = replacement.json()
    assert replacement_payload["document_id"] == first["document_id"]
    assert replacement_payload["version_id"] != first["version_id"]

    detail = client.get(f"/api/documents/{first['document_id']}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "uploaded"
    assert detail.json()["current_version_id"] == first["version_id"]
    assert detail.json()["latest_version_id"] == replacement_payload["version_id"]
    assert detail.json()["parent_count"] == 2
    assert detail.json()["child_count"] == 4
    download = client.get(f"/api/documents/{first['document_id']}/content")
    assert download.status_code == 200
    assert download.content == b"accepted old"
    with client.app.state.document_service.session_factory() as session:
        assert session.get(IndexGenerationRecord, generation_id).status == "active"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/documents"),
        ("POST", "/api/documents/upload"),
        ("GET", "/api/documents/unknown"),
        ("GET", "/api/documents/unknown/content"),
        ("POST", "/api/documents/unknown/reindex"),
        ("DELETE", "/api/documents/unknown"),
        ("GET", "/api/tasks/unknown"),
    ],
)
def test_all_document_and_task_routes_require_authentication(
    client: TestClient, method: str, path: str
):
    response = client.request(method, path)

    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required"}


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/documents/upload"),
        ("POST", "/api/documents/unknown/reindex"),
        ("DELETE", "/api/documents/unknown"),
    ],
)
def test_mutations_require_a_valid_csrf_token(
    client: TestClient, method: str, path: str
):
    login(client)

    response = client.request(method, path, headers={"X-CSRF-Token": "wrong-token"})

    assert response.status_code == 403
    assert response.json() == {"detail": "Invalid CSRF token"}
    assert "wrong-token" not in response.text


@pytest.mark.parametrize(
    ("filename", "content"),
    [("notes.md", b"unsupported"), ("broken.txt", b"\xff")],
)
def test_invalid_upload_returns_400_without_creating_a_document(
    client: TestClient, filename: str, content: bytes
):
    csrf = login(client)

    response = client.post(
        "/api/documents/upload",
        headers={"X-CSRF-Token": csrf},
        files={"file": (filename, content, "application/octet-stream")},
    )

    assert response.status_code == 400
    assert client.get("/api/documents").json() == []


def test_oversize_upload_returns_413_without_creating_a_document(
    tmp_path: Path, monkeypatch
):
    app = make_app(tmp_path, monkeypatch, max_upload_mb=1)
    with TestClient(app) as client:
        csrf = login(client)
        response = client.post(
            "/api/documents/upload",
            headers={"X-CSRF-Token": csrf},
            files={
                "file": (
                    "large.txt",
                    b"x" * (1024 * 1024 + 1),
                    "text/plain",
                )
            },
        )

        assert response.status_code == 413
        assert "1 MB" in response.json()["detail"]
        assert str(tmp_path) not in response.text
        assert client.get("/api/documents").json() == []
        assert not list((tmp_path / "raw").rglob("*.txt"))


def test_unknown_resources_return_404(client: TestClient):
    csrf = login(client)

    assert client.get("/api/documents/missing").status_code == 404
    assert client.get("/api/documents/missing/content").status_code == 404
    assert (
        client.post(
            "/api/documents/missing/reindex", headers={"X-CSRF-Token": csrf}
        ).status_code
        == 404
    )
    assert (
        client.delete(
            "/api/documents/missing", headers={"X-CSRF-Token": csrf}
        ).status_code
        == 404
    )
    assert client.get("/api/tasks/missing").status_code == 404


def test_task_and_document_errors_are_sanitized(client: TestClient):
    csrf = login(client)
    uploaded = upload(client, csrf).json()
    factory = client.app.state.document_service.session_factory
    secret_error = (
        r"C:\private\SECRET_PATH\index.sqlite3 "
        "Authorization: Bearer SECRET_TOKEN https://secret.internal"
    )
    with factory.begin() as session:
        document = session.get(DocumentRecord, uploaded["document_id"])
        task = session.get(TaskRecord, uploaded["task_id"])
        document.error = secret_error
        task.error = secret_error
        task.status = "failed"
        task.attempts = 2

    detail = client.get(f"/api/documents/{uploaded['document_id']}")
    task_response = client.get(f"/api/tasks/{uploaded['task_id']}")

    assert detail.status_code == task_response.status_code == 200
    assert task_response.json()["status"] == "failed"
    assert task_response.json()["attempts"] == 2
    for response in (detail, task_response):
        assert "Operation failed" in response.text
        assert "SECRET_PATH" not in response.text
        assert "SECRET_TOKEN" not in response.text
        assert "secret.internal" not in response.text
        assert str(client.app.state.settings.web.raw_file_dir) not in response.text


def test_latest_version_is_selected_by_version_number_not_identifier(
    client: TestClient,
):
    csrf = login(client)
    uploaded = upload(client, csrf).json()
    factory = client.app.state.document_service.session_factory
    with factory.begin() as session:
        session.add(
            DocumentVersionRecord(
                id="aaa-lexically-first-but-newer",
                document_id=uploaded["document_id"],
                version=2,
                path="do-not-expose-this-path",
                sha256="a" * 64,
                status="uploaded",
            )
        )

    response = client.get(f"/api/documents/{uploaded['document_id']}")

    assert response.status_code == 200
    assert response.json()["latest_version_id"] == "aaa-lexically-first-but-newer"
    assert "do-not-expose-this-path" not in response.text
    assert response.json()["parent_count"] == 0
    assert response.json()["child_count"] == 0


def test_active_generation_counts_do_not_leak_from_retired_generations(
    client: TestClient,
):
    csrf = login(client)
    uploaded = upload(client, csrf).json()
    factory = client.app.state.document_service.session_factory
    document_id = uploaded["document_id"]
    repository = IndexRepository(factory)
    retired = ParentChunk(
        id="retired-parent",
        document_id=document_id,
        ordinal=0,
        text="retired",
        content_hash="retired-hash",
        metadata={},
    )
    repository.prepare_generation("retired-generation")
    repository.activate_generation("retired-generation", [retired])
    repository.prepare_generation("active-generation")
    repository.activate_generation("active-generation", [])

    response = client.get(f"/api/documents/{document_id}")

    assert response.status_code == 200
    assert response.json()["parent_count"] == 0
    assert response.json()["child_count"] == 0
    with factory() as session:
        assert (
            session.scalar(
                select(IndexGenerationRecord).where(
                    IndexGenerationRecord.status == "active"
                )
            ).id
            == "active-generation"
        )
