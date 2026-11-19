from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.config import AppSettings, WebSettings
from app.main import create_app


def _settings(frontend_dist: Path) -> AppSettings:
    return AppSettings(
        app_env="test",
        model_provider="fake",
        dashscope_api_key=SecretStr("static-test-secret"),
        web=WebSettings(frontend_dist=frontend_dist),
    )


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setenv("RAG_STATE_DIR", str(tmp_path / "state"))
    return TestClient(
        create_app(
            settings=_settings(tmp_path / "dist"),
            qdrant_probe=lambda: {"status": "ok"},
        )
    )


def test_non_api_gets_serve_index_and_existing_assets(
    tmp_path: Path, monkeypatch
):
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text(
        "<!doctype html><html><body><div id=root>SPA</div></body></html>",
        encoding="utf-8",
    )
    (assets / "app.js").write_text("console.log('app')", encoding="utf-8")
    client = _client(tmp_path, monkeypatch)

    for path in ("/", "/chat", "/chat/conversation-1", "/documents"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert '<div id=root>SPA</div>' in response.text

    asset = client.get("/assets/app.js")
    assert asset.status_code == 200
    assert asset.text == "console.log('app')"
    assert "javascript" in asset.headers["content-type"]


def test_api_health_and_missing_assets_never_fall_through_to_html(
    tmp_path: Path, monkeypatch
):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<div>PRIVATE SPA MARKER</div>", encoding="utf-8")
    client = _client(tmp_path, monkeypatch)

    unknown_api = client.get("/api/unknown")
    assert unknown_api.status_code == 404
    assert unknown_api.json() == {"detail": "Not Found"}

    api_error = client.get("/api/documents")
    assert api_error.status_code == 401
    assert api_error.json() == {"detail": "Authentication required"}

    unknown_health = client.get("/health/unknown")
    assert unknown_health.status_code == 404
    assert unknown_health.json() == {"detail": "Not Found"}

    missing_asset = client.get("/assets/missing.js")
    assert missing_asset.status_code == 404
    assert not missing_asset.headers["content-type"].startswith("text/html")

    for response in (unknown_api, api_error, unknown_health, missing_asset):
        assert "PRIVATE SPA MARKER" not in response.text
        assert not response.headers["content-type"].startswith("text/html")


def test_missing_frontend_build_leaves_non_api_routes_as_404(
    tmp_path: Path, monkeypatch
):
    client = _client(tmp_path, monkeypatch)

    response = client.get("/chat")

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}
