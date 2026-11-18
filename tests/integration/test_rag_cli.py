import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
from openpyxl import Workbook
from sqlalchemy import select
from typer.testing import CliRunner

from app.bootstrap import FakeChatModel, build_rag_service
from app.cli import cli_app
from app.config import QdrantSettings, load_settings
from app.db.models import ChildChunkRecord, IndexGenerationRecord
from app.db.session import create_session_factory
from app.indexing.repository import IndexRepository
from app.ingestion.text_office import QA_COLUMNS
from app.models.bailian import BailianConfigurationError
from app.retrieval.qdrant_store import QdrantGenerationStore

pytestmark = pytest.mark.integration
runner = CliRunner()


@dataclass(frozen=True)
class CLIEnvironment:
    corpus: Path
    qa: Path
    state: Path
    qdrant_url: str
    collection_prefix: str


def write_qa_workbook(path: Path, row_count: int) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(QA_COLUMNS)
    for row in range(row_count):
        worksheet.append(
            (
                f"QA-{row:04d}",
                f"差旅住宿标准是多少？{row}",
                "每天500元。",
                "DOC-000",
                "住宿标准为每天500元。",
                "行政制度",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
    workbook.close()


@pytest.fixture
def cli_environment(
    tmp_path: Path, qdrant_url: str, monkeypatch: pytest.MonkeyPatch
) -> CLIEnvironment:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "DOC-000.txt").write_text("差旅住宿标准为每天500元。", encoding="utf-8")
    (corpus / "DOC-001.txt").write_text("年假为10天。", encoding="utf-8")
    (corpus / "evaluation.jsonl").write_text(
        json.dumps({"question": "held out", "answerable": False}) + "\n",
        encoding="utf-8",
    )
    qa = tmp_path / "qa" / "knowledge_qa.xlsx"
    write_qa_workbook(qa, row_count=2)
    state = tmp_path / "state"
    collection_prefix = f"cli_test_{uuid4().hex}"
    monkeypatch.setenv("RAG_STATE_DIR", str(state))
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    original_load_settings = load_settings

    def isolated_settings() -> object:
        settings = original_load_settings()
        return settings.model_copy(
            update={
                "qdrant": QdrantSettings(
                    url=qdrant_url, collection_prefix=collection_prefix
                ),
                "chunk": settings.chunk.model_copy(
                    update={"parent_size": 200, "child_size": 80, "child_overlap": 10}
                ),
            }
        )

    monkeypatch.setattr("app.cli.load_settings", isolated_settings)
    return CLIEnvironment(corpus, qa, state, qdrant_url, collection_prefix)


def ingest(environment: CLIEnvironment, command: str = "ingest"):
    return runner.invoke(
        cli_app,
        [
            command,
            "--corpus",
            str(environment.corpus),
            "--qa",
            str(environment.qa),
        ],
        env={"MODEL_PROVIDER": "fake", "APP_ENV": "test"},
    )


def active_generation(environment: CLIEnvironment) -> str | None:
    repository = IndexRepository(
        create_session_factory(environment.state / "index.sqlite3")
    )
    return repository.active_generation()


def test_ingest_rebuild_and_ask_show_answer_status_and_citations(
    cli_environment: CLIEnvironment,
):
    ingested = ingest(cli_environment)
    assert ingested.exit_code == 0, ingested.output
    assert "ordinary documents: 2" in ingested.output
    assert "knowledge Q&A: 2" in ingested.output
    assert "evaluation.jsonl" not in ingested.output
    first = active_generation(cli_environment)
    assert first is not None

    rebuilt = ingest(cli_environment, command="rebuild")
    assert rebuilt.exit_code == 0, rebuilt.output
    second = active_generation(cli_environment)
    assert second is not None and second != first

    store = QdrantGenerationStore(
        QdrantSettings(
            url=cli_environment.qdrant_url,
            collection_prefix=cli_environment.collection_prefix,
        )
    )
    # The rebuild retains the old generation rather than deleting it first.
    assert store.collection_name(first) in {
        collection.name for collection in store.client.get_collections().collections
    }
    store.client.close()

    asked = runner.invoke(
        cli_app,
        ["ask", "--question", "差旅住宿标准是多少？"],
        env={"MODEL_PROVIDER": "fake", "APP_ENV": "test"},
    )
    assert asked.exit_code == 0, asked.output
    assert "Status:" in asked.output
    assert "500" in asked.output
    assert "[S1]" in asked.output
    assert "Citations:" in asked.output
    assert "DOC-000" in asked.output
    assert second in asked.output


def test_verify_rejects_short_qa_file(cli_environment, tmp_path):
    qa_path = tmp_path / "qa.xlsx"
    write_qa_workbook(qa_path, row_count=999)
    result = runner.invoke(
        cli_app,
        ["verify", "--corpus", str(cli_environment.corpus), "--qa", str(qa_path)],
        env={"MODEL_PROVIDER": "fake", "APP_ENV": "test"},
    )
    assert result.exit_code == 1
    assert "expected 1000" in result.stdout
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert {check["name"] for check in report["checks"]} == {
        "ordinary_documents",
        "knowledge_qa",
        "active_generation",
        "parent_child_integrity",
        "qdrant_dimension",
        "bm25_manifest",
    }


def test_verify_rejects_child_with_missing_parent(cli_environment):
    assert ingest(cli_environment).exit_code == 0
    database = cli_environment.state / "index.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(
            "DELETE FROM parent_chunks WHERE id = ("
            "SELECT parent_id FROM child_chunks LIMIT 1)"
        )

    result = runner.invoke(
        cli_app,
        [
            "verify",
            "--corpus",
            str(cli_environment.corpus),
            "--qa",
            str(cli_environment.qa),
        ],
        env={"MODEL_PROVIDER": "fake", "APP_ENV": "test"},
    )
    assert result.exit_code == 1
    assert "missing parent" in result.stdout
    integrity = next(
        check
        for check in json.loads(result.stdout)["checks"]
        if check["name"] == "parent_child_integrity"
    )
    assert integrity["ok"] is False
    assert integrity["actual"]["missing_parents"]


def test_fake_provider_requires_test_environment(cli_environment):
    result = ingest(cli_environment)
    assert result.exit_code == 0

    rejected = runner.invoke(
        cli_app,
        ["ask", "--question", "差旅住宿标准是多少？"],
        env={"MODEL_PROVIDER": "fake", "APP_ENV": "production"},
    )
    assert rejected.exit_code == 1
    assert "APP_ENV=test" in rejected.output


def test_fake_fixture_never_selects_real_mode_and_missing_keys_do_not_mutate(
    cli_environment: CLIEnvironment, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MODEL_PROVIDER", "fake")
    monkeypatch.setenv("APP_ENV", "test")
    settings = load_settings().model_copy(
        update={
            "qdrant": QdrantSettings(
                url="http://127.0.0.1:1",
                collection_prefix=cli_environment.collection_prefix,
            )
        }
    )
    service = build_rag_service(settings)
    assert isinstance(service.chat_model, FakeChatModel)
    service.retrieval_service.qdrant.client.close()

    assert ingest(cli_environment).exit_code == 0
    before = active_generation(cli_environment)
    with create_session_factory(cli_environment.state / "index.sqlite3")() as session:
        generation_count = len(session.scalars(select(IndexGenerationRecord)).all())

    missing_keys = runner.invoke(
        cli_app,
        [
            "rebuild",
            "--corpus",
            str(cli_environment.corpus),
            "--qa",
            str(cli_environment.qa),
        ],
        env={"MODEL_PROVIDER": "bailian", "APP_ENV": "production"},
    )
    assert missing_keys.exit_code == 1
    assert "DASHSCOPE_API_KEY is required" in missing_keys.output
    assert active_generation(cli_environment) == before
    with create_session_factory(cli_environment.state / "index.sqlite3")() as session:
        assert len(session.scalars(select(IndexGenerationRecord)).all()) == (
            generation_count
        )

    monkeypatch.setenv("MODEL_PROVIDER", "bailian")
    monkeypatch.setenv("APP_ENV", "test")
    bailian_settings = settings.model_copy(update={"model_provider": "bailian"})
    with pytest.raises(BailianConfigurationError):
        build_rag_service(bailian_settings)


def test_verify_reports_qdrant_dimension_and_bm25_manifest(cli_environment):
    # Verify source counts independently without building another 1000-row index.
    assert ingest(cli_environment).exit_code == 0
    for ordinal in range(2, 100):
        (cli_environment.corpus / f"DOC-{ordinal:03d}.txt").write_text(
            f"document {ordinal}", encoding="utf-8"
        )
    write_qa_workbook(cli_environment.qa, row_count=1000)

    result = runner.invoke(
        cli_app,
        [
            "verify",
            "--corpus",
            str(cli_environment.corpus),
            "--qa",
            str(cli_environment.qa),
        ],
        env={"MODEL_PROVIDER": "fake", "APP_ENV": "test"},
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["ok"] is True
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["ordinary_documents"]["actual"]["total"] == 100
    assert checks["ordinary_documents"]["actual"]["by_extension"][".txt"] == 100
    assert checks["knowledge_qa"]["actual"] == 1000
    assert checks["active_generation"]["actual"] == active_generation(cli_environment)
    assert checks["qdrant_dimension"]["actual"]["dimension"] == 1024
    generation = active_generation(cli_environment)
    assert checks["bm25_manifest"]["actual"]["generation_id"] == generation
    with create_session_factory(cli_environment.state / "index.sqlite3")() as session:
        child_count = len(
            session.scalars(
                select(ChildChunkRecord).where(
                    ChildChunkRecord.generation_id == generation
                )
            ).all()
        )
    assert checks["bm25_manifest"]["actual"]["chunk_count"] == child_count
