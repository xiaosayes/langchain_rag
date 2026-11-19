import json
from pathlib import Path
from types import SimpleNamespace

from openai import OpenAIError
from qdrant_client.http.exceptions import ResponseHandlingException
from sqlalchemy.exc import DatabaseError
from typer.testing import CliRunner

from app.cli import cli_app
from app.config import AppSettings, load_settings
from app.models.bailian import BailianRequestError
from app.verification import CoreVerifier

runner = CliRunner()


def assert_excludes_sensitive_text(output: str, *sentinels: str) -> None:
    for sentinel in sentinels:
        assert sentinel not in output
    assert "Authorization" not in output
    assert "Bearer " not in output
    assert "https://" not in output


def test_ingest_sqlalchemy_failure_is_actionable_without_sql_or_secrets(monkeypatch):
    def fail_build(corpus, qa):
        raise DatabaseError(
            "SELECT * FROM private_table WHERE token = 'SQL_SENTINEL'",
            None,
            Exception(r"C:\private\SQL_PATH_SENTINEL\index.sqlite3"),
        )

    monkeypatch.setattr("app.cli._build", fail_build)

    result = runner.invoke(cli_app, ["ingest", "--corpus", "x", "--qa", "y"])

    assert result.exit_code == 1
    assert "Database operation failed" in result.output
    assert "check local index storage" in result.output
    assert_excludes_sensitive_text(
        result.output,
        "SQL_SENTINEL",
        "SQL_PATH_SENTINEL",
        "SELECT * FROM private_table",
        "private_table",
    )


def test_verification_sqlalchemy_failure_uses_same_sanitization(monkeypatch):
    def fail_repository():
        raise DatabaseError(
            "DELETE FROM secret_generation WHERE id = 'VERIFY_SQL_SENTINEL'",
            None,
            Exception("VERIFY_PATH_SENTINEL"),
        )

    monkeypatch.setattr("app.verification._repository", fail_repository)
    verifier = CoreVerifier(AppSettings(), Path("missing-corpus"), Path("missing-qa"))

    report = verifier.run().to_dict()

    message = next(
        check["message"]
        for check in report["checks"]
        if check["name"] == "active_generation"
    )
    assert "Database operation failed" in message
    assert_excludes_sensitive_text(
        message,
        "VERIFY_SQL_SENTINEL",
        "VERIFY_PATH_SENTINEL",
        "DELETE FROM secret_generation",
    )


def test_verification_qdrant_failure_hides_store_diagnostics():
    class BrokenClient:
        def get_collection(self, collection_name):
            raise ResponseHandlingException(
                "QDRANT_SENTINEL https://qdrant-secret.internal "
                "Authorization: Bearer QDRANT_SECRET"
            )

    store = SimpleNamespace(
        client=BrokenClient(), collection_name=lambda generation_id: generation_id
    )
    verifier = CoreVerifier(AppSettings(), Path("missing"), Path("missing"))

    check = verifier._check_qdrant(store, object(), "generation")

    assert check.ok is False
    assert "Vector store operation failed" in check.message
    assert "check Qdrant availability" in check.message
    assert_excludes_sensitive_text(
        check.message, "QDRANT_SENTINEL", "QDRANT_SECRET", "qdrant-secret.internal"
    )


def test_verify_malformed_yaml_json_hides_yaml_values_and_paths(tmp_path, monkeypatch):
    malformed = tmp_path / "YAML_PATH_SENTINEL.yaml"
    malformed.write_text(
        "embedding: [YAML_VALUE_SENTINEL, Authorization: Bearer YAML_SECRET",
        encoding="utf-8",
    )
    monkeypatch.setattr("app.cli.load_settings", lambda: load_settings(malformed))

    result = runner.invoke(cli_app, ["verify", "--corpus", "x", "--qa", "y"])

    assert result.exit_code == 1
    report = json.loads(result.stdout)
    message = report["checks"][0]["message"]
    assert report["checks"][0]["name"] == "configuration"
    assert "Invalid configuration" in message
    assert "YAML" in message
    assert_excludes_sensitive_text(
        result.stdout,
        "YAML_PATH_SENTINEL",
        "YAML_VALUE_SENTINEL",
        "YAML_SECRET",
    )


def test_ask_sanitizes_openai_sdk_failure_during_streaming(monkeypatch):
    class Repository:
        def active_generation(self):
            return "generation-test"

    class Client:
        def close(self):
            pass

    class Service:
        repository = Repository()
        retrieval_service = SimpleNamespace(qdrant=SimpleNamespace(client=Client()))

        async def prepare(self, question, history, generation_id):
            return SimpleNamespace(degraded=False, citations=[])

        async def stream_answer(self, prepared):
            raise OpenAIError(
                "OPENAI_SENTINEL https://openai-secret.internal "
                "Authorization: Bearer OPENAI_SECRET"
            )
            yield

    monkeypatch.setattr("app.cli.build_rag_service", lambda settings: Service())

    result = runner.invoke(cli_app, ["ask", "--question", "question"])

    assert result.exit_code == 1
    assert "Chat provider request failed" in result.output
    assert "check chat credentials" in result.output
    assert_excludes_sensitive_text(
        result.output, "OPENAI_SENTINEL", "OPENAI_SECRET", "openai-secret.internal"
    )


def test_cli_preserves_safe_bailian_request_error(monkeypatch):
    def fail_service(settings):
        raise BailianRequestError("Safe Bailian action; check DASHSCOPE_API_KEY")

    monkeypatch.setattr("app.cli.build_rag_service", fail_service)

    result = runner.invoke(cli_app, ["ask", "--question", "question"])

    assert result.exit_code == 1
    assert "Safe Bailian action; check DASHSCOPE_API_KEY" in result.output


def test_cli_does_not_swallow_programming_errors(monkeypatch):
    def fail_build(corpus, qa):
        raise TypeError("PROGRAMMING_ERROR_SENTINEL")

    monkeypatch.setattr("app.cli._build", fail_build)

    result = runner.invoke(cli_app, ["ingest", "--corpus", "x", "--qa", "y"])

    assert result.exit_code == 1
    assert isinstance(result.exception, TypeError)
    assert "PROGRAMMING_ERROR_SENTINEL" in str(result.exception)
