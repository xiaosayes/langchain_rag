"""Real Qdrant/fake-provider API smoke flow and browser process entry points."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.bootstrap import build_index_builder
from app.config import (
    AppSettings,
    ChunkSettings,
    EmbeddingSettings,
    QdrantSettings,
    RetrievalSettings,
    WebSettings,
    load_settings,
)
from app.db.models import ChildChunkRecord, DocumentRecord, IndexGenerationRecord
from app.db.session import create_session_factory
from app.main import create_app
from app.web.queue import TaskQueue
from app.web.records import AuthSessionRecord
from app.worker import IngestionWorker

pytest_plugins = ("tests.integration.conftest",)

ADMIN_USERNAME = "e2e-admin"
ADMIN_PASSWORD = "e2e-correct-password"
ALPHA_TEXT = (
    "Conference travel policy: the hotel limit is 750 CNY per night. "
    "Keep the itemized invoice for reimbursement."
)
BETA_TEXT = (
    "Equipment repair policy: call the internal hotline 4000 before "
    "authorizing any external repair."
)


def _settings(qdrant_url: str, root: Path) -> AppSettings:
    return AppSettings(
        app_env="test",
        model_provider="fake",
        embedding=EmbeddingSettings(dimension=64),
        qdrant=QdrantSettings(url=qdrant_url),
        chunk=ChunkSettings(parent_size=180, child_size=60, child_overlap=10),
        retrieval=RetrievalSettings(
            dense_top_k=10,
            bm25_top_k=10,
            final_parent_count=4,
        ),
        web=WebSettings(raw_file_dir=root / "raw"),
    )


def _active_generation(client: TestClient) -> str | None:
    factory = client.app.state.document_service.session_factory
    with factory() as session:
        return session.scalar(
            select(IndexGenerationRecord.id).where(
                IndexGenerationRecord.status == "active"
            )
        )


def _upload(client: TestClient, csrf: str, path: Path) -> dict[str, str]:
    response = client.post(
        "/api/documents/upload",
        headers={"X-CSRF-Token": csrf},
        data={"kind": "document"},
        files={"file": (path.name, path.read_bytes(), "text/plain")},
    )
    assert response.status_code == 202, response.text
    return response.json()


def _run_task(client: TestClient, worker: IngestionWorker, task_id: str) -> dict:
    assert worker.run_once() == 1
    response = client.get(f"/api/tasks/{task_id}")
    assert response.status_code == 200
    return response.json()


def _sse_events(response) -> list[tuple[str, dict]]:
    events = []
    for block in response.text.replace("\r\n", "\n").strip().split("\n\n"):
        lines = block.splitlines()
        event = next(
            line.removeprefix("event: ")
            for line in lines
            if line.startswith("event: ")
        )
        data = "\n".join(
            line.removeprefix("data: ")
            for line in lines
            if line.startswith("data: ")
        )
        events.append((event, json.loads(data)))
    return events


@pytest.mark.integration
def test_single_admin_web_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qdrant_url: str,
):
    """Exercise the public API and real worker against isolated persistent state."""
    state = tmp_path / "state"
    monkeypatch.setenv("MODEL_PROVIDER", "fake")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("RAG_STATE_DIR", str(state))
    monkeypatch.setenv("ADMIN_USERNAME", ADMIN_USERNAME)
    monkeypatch.setenv("ADMIN_PASSWORD", ADMIN_PASSWORD)
    alpha = tmp_path / "alpha.txt"
    beta = tmp_path / "beta.txt"
    alpha.write_text(ALPHA_TEXT, encoding="utf-8")
    beta.write_text(BETA_TEXT, encoding="utf-8")

    settings = _settings(qdrant_url, state)
    app = create_app(settings=settings)
    builder = build_index_builder(settings)
    worker = IngestionWorker(
        TaskQueue(app.state.document_service.session_factory),
        builder,
        owner="e2e-worker",
        lease_seconds=60,
    )
    try:
        with TestClient(app) as client:
            ready = client.get("/health/ready")
            assert ready.status_code == 200
            assert ready.json() == {
                "status": "ok",
                "database": "ok",
                "qdrant": "ok",
            }
            assert _active_generation(client) is None
            assert client.get("/api/documents").status_code == 401

            # Login intentionally has no rate limiter in this single-admin phase.
            for _ in range(12):
                rejected = client.post(
                    "/api/auth/login",
                    json={"username": ADMIN_USERNAME, "password": "wrong"},
                )
                assert rejected.status_code == 401
            login = client.post(
                "/api/auth/login",
                json={"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD},
            )
            assert login.status_code == 200
            assert login.cookies["rag_session"]
            cookie = login.headers["set-cookie"].lower()
            assert "httponly" in cookie
            assert "samesite=lax" in cookie
            assert "max-age=43200" in cookie
            csrf = login.json()["csrf_token"]
            first_me = client.get("/api/auth/me")
            second_me = client.get("/api/auth/me")
            assert first_me.json() == second_me.json()
            assert first_me.json()["csrf_token"] == csrf

            alpha_upload = _upload(client, csrf, alpha)
            beta_upload = _upload(client, csrf, beta)
            assert alpha_upload["document_id"] != beta_upload["document_id"]
            assert worker.run_once() == 1
            for task_id in (alpha_upload["task_id"], beta_upload["task_id"]):
                task = client.get(f"/api/tasks/{task_id}").json()
                assert task["status"] == "succeeded", task
                assert task["attempts"] == 1
            documents = client.get("/api/documents").json()
            assert {document["filename"] for document in documents} == {
                "alpha.txt",
                "beta.txt",
            }
            assert all(document["status"] == "ready" for document in documents)
            assert all(document["child_count"] > 0 for document in documents)
            initial_generation = _active_generation(client)
            assert initial_generation is not None

            bad_upload = client.post(
                "/api/documents/upload",
                headers={"X-CSRF-Token": csrf},
                files={"file": ("broken.txt", b"\xff", "text/plain")},
            )
            assert bad_upload.status_code == 400
            assert _active_generation(client) == initial_generation
            assert len(client.get("/api/documents").json()) == 2

            chat = client.post(
                "/api/chat/stream",
                headers={"X-CSRF-Token": csrf},
                json={
                    "question": "Conference travel policy hotel limit 750 CNY "
                    "per night itemized invoice reimbursement"
                },
            )
            assert chat.status_code == 200
            events = _sse_events(chat)
            assert events[0][0] == "start"
            assert events[-1][0] == "done"
            assert events[-2][0] == "citations"
            assert all(
                event == "token" for event, _ in events[1:-2]
            ), events
            answer = "".join(
                data["content"] for event, data in events if event == "token"
            )
            assert "750 CNY" in answer
            assert "[S1]" in answer
            start = events[0][1]
            citations = events[-2][1]["citations"]
            assert citations
            assert any("750 CNY" in citation["text"] for citation in citations)
            messages = client.get(
                f"/api/sessions/{start['conversation_id']}/messages"
            ).json()
            assert messages[-1]["id"] == start["message_id"]
            assert messages[-1]["content"] == answer
            assert messages[-1]["citations"] == citations

            feedback = client.post(
                "/api/feedback",
                headers={"X-CSRF-Token": csrf},
                json={
                    "message_id": start["message_id"],
                    "score": 1,
                    "comment": "Grounded and useful",
                },
            )
            assert feedback.status_code == 200
            assert feedback.json()["score"] == 1
            assert "run_id" not in feedback.json()

            reindex = client.post(
                f"/api/documents/{alpha_upload['document_id']}/reindex",
                headers={"X-CSRF-Token": csrf},
            )
            assert reindex.status_code == 202
            reindex_task = _run_task(client, worker, reindex.json()["task_id"])
            assert reindex_task["status"] == "succeeded"
            reindexed_generation = _active_generation(client)
            assert reindexed_generation != initial_generation

            # A failed delete leaves both active content and public download intact.
            failed_delete = client.delete(
                f"/api/documents/{beta_upload['document_id']}",
                headers={"X-CSRF-Token": csrf},
            )
            assert failed_delete.status_code == 202
            real_build = builder.build

            def fail_build(*args, **kwargs):
                raise RuntimeError("forced delete validation failure")

            monkeypatch.setattr(builder, "build", fail_build)
            failed_task = _run_task(
                client, worker, failed_delete.json()["task_id"]
            )
            assert failed_task["status"] == "failed"
            assert _active_generation(client) == reindexed_generation
            assert client.get(
                f"/api/documents/{beta_upload['document_id']}/content"
            ).content == beta.read_bytes()
            factory = app.state.document_service.session_factory
            with factory() as session:
                beta_record = session.get(
                    DocumentRecord, beta_upload["document_id"]
                )
                assert beta_record is not None
                assert beta_record.deleted is False
                assert beta_record.current_version_id == beta_upload["version_id"]
            monkeypatch.setattr(builder, "build", real_build)

            deleted = client.delete(
                f"/api/documents/{beta_upload['document_id']}",
                headers={"X-CSRF-Token": csrf},
            )
            assert deleted.status_code == 202
            delete_task = _run_task(client, worker, deleted.json()["task_id"])
            assert delete_task["status"] == "succeeded"
            delete_generation = _active_generation(client)
            assert delete_generation != reindexed_generation
            remaining = client.get("/api/documents").json()
            assert [document["filename"] for document in remaining] == ["alpha.txt"]
            assert client.get(
                f"/api/documents/{beta_upload['document_id']}"
            ).status_code == 404
            assert client.get(
                f"/api/documents/{beta_upload['document_id']}/content"
            ).status_code == 404
            with factory() as session:
                beta_record = session.get(
                    DocumentRecord, beta_upload["document_id"]
                )
                assert beta_record is not None
                assert beta_record.deleted is True
                child_document_ids = set(
                    session.scalars(
                        select(ChildChunkRecord.document_id).where(
                            ChildChunkRecord.generation_id == delete_generation
                        )
                    )
                )
                assert alpha_upload["document_id"] in child_document_ids
                assert beta_upload["document_id"] not in child_document_ids

            with factory.begin() as session:
                auth_session = session.scalar(select(AuthSessionRecord))
                assert auth_session is not None
                auth_session.expires_at = datetime.now(UTC) - timedelta(seconds=1)
            assert client.get("/api/auth/me").status_code == 401
    finally:
        builder.qdrant.client.close()
        app.state.rag_service.retrieval_service.qdrant.client.close()


def _browser_settings() -> AppSettings:
    return load_settings(Path(os.environ["RAG_E2E_CONFIG"]))


def _serve_api(port: int) -> None:
    import uvicorn

    uvicorn.run(
        create_app(settings=_browser_settings()),
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )


def _serve_worker() -> None:
    settings = _browser_settings()
    root = Path(os.environ["RAG_STATE_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    builder = build_index_builder(settings)
    worker = IngestionWorker(
        TaskQueue(create_session_factory(root / "index.sqlite3")),
        builder,
        lease_seconds=settings.web.task_lease_seconds,
    )
    try:
        while True:
            if worker.run_once() == 0:
                time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        builder.qdrant.client.close()


def main() -> None:
    """Provide explicit API/worker processes for the Playwright harness."""
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--serve-api", action="store_true")
    mode.add_argument("--serve-worker", action="store_true")
    parser.add_argument("--port", type=int, default=18000)
    arguments = parser.parse_args()
    if arguments.serve_api:
        _serve_api(arguments.port)
    else:
        _serve_worker()


if __name__ == "__main__":
    main()
