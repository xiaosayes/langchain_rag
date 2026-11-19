"""Print the explicit API and worker commands for two development terminals."""

from app.config import load_settings


def main() -> None:
    settings = load_settings()
    print("Run these commands separately from the repository root in two terminals.")
    print("Both processes need the same MODEL_PROVIDER, APP_ENV, RAG_STATE_DIR, and admin settings.")
    print("Start Qdrant first; this helper does not start or hide any process.")
    print(
        "API:    uv run uvicorn app.main:create_app --factory "
        f"--host {settings.web.host} --port {settings.web.port}"
    )
    print("Worker: uv run python -m app.worker")


if __name__ == "__main__":
    main()
