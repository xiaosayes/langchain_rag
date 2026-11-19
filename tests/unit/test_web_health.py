from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from app.config import AppSettings, WebSettings
from app.main import create_app


def test_settings() -> AppSettings:
    return AppSettings(
        app_env="test",
        model_provider="fake",
        dashscope_api_key=SecretStr("health-test-secret"),
        web=WebSettings(
            host="127.0.0.1",
            port=18000,
            session_cookie_name="rag_session",
            session_ttl_hours=12,
            secure_cookie=False,
            max_upload_mb=50,
            max_chat_concurrency=5,
            task_lease_seconds=300,
            raw_file_dir=Path("var/core/raw_files"),
            frontend_dist=Path("frontend/dist"),
        ),
    )


test_settings.__test__ = False


def test_health_endpoints_expose_no_secrets(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAG_STATE_DIR", str(tmp_path))
    client = TestClient(
        create_app(settings=test_settings(), qdrant_probe=lambda: {"status": "ok"})
    )

    assert client.get("/health/live").json() == {"status": "ok"}
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["database"] == "ok"
    assert ready.json()["qdrant"] == "ok"
    assert "dashscope_api_key" not in ready.text
    assert "health-test-secret" not in ready.text


def test_failed_qdrant_probe_is_sanitized(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAG_STATE_DIR", str(tmp_path))

    def failing_probe():
        raise RuntimeError("raw Qdrant failure containing health-test-secret")

    client = TestClient(
        create_app(settings=test_settings(), qdrant_probe=failing_probe)
    )
    ready = client.get("/health/ready")

    assert ready.status_code == 503
    assert ready.json() == {
        "status": "error",
        "database": "ok",
        "qdrant": "error",
    }
    assert "raw Qdrant failure" not in ready.text
    assert "health-test-secret" not in ready.text


def test_web_settings_have_expected_defaults():
    assert WebSettings() == WebSettings(
        host="127.0.0.1",
        port=18000,
        session_cookie_name="rag_session",
        session_ttl_hours=12,
        secure_cookie=False,
        max_upload_mb=50,
        max_chat_concurrency=5,
        task_lease_seconds=300,
        raw_file_dir=Path("var/core/raw_files"),
        frontend_dist=Path("frontend/dist"),
    )


@pytest.mark.parametrize(
    "field",
    ["port", "max_upload_mb", "task_lease_seconds", "session_ttl_hours"],
)
@pytest.mark.parametrize("value", [0, -1])
def test_web_settings_require_positive_values(field: str, value: int):
    with pytest.raises(ValidationError):
        WebSettings(**{field: value})
