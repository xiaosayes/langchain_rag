import json
import pickle
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

import fitz
import pytest
from docx import Document as DocxDocument
from langchain_core.documents import Document
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
from app.rag.prompts import build_grounded_messages
from app.rag.service import INSUFFICIENT_CONTEXT_REFUSAL
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
    # Both ordinary sources carry the expected token so this composition test
    # does not make semantic retrieval claims about non-semantic fake embeddings.
    (corpus / "DOC-001.txt").write_text(
        "年假为10天；差旅住宿标准为每天500元。", encoding="utf-8"
    )
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


def test_fake_chat_answers_real_grounded_prompt():
    messages = build_grounded_messages(
        "差旅住宿标准是多少？",
        [Document(page_content="差旅住宿标准为每天500元。")],
    )

    answer = FakeChatModel().invoke(messages).content

    assert answer == "差旅住宿标准为每天500元。[S1]"
    assert INSUFFICIENT_CONTEXT_REFUSAL not in answer


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
    assert "Citations:" in asked.output
    answer_section = asked.output.split("Citations:", 1)[0]
    assert "500" in answer_section
    assert "[S1]" in answer_section
    assert INSUFFICIENT_CONTEXT_REFUSAL not in answer_section
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
        "vector_space_compatibility",
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


def write_exact_corpus_shape(corpus: Path) -> None:
    for ordinal in range(2, 25):
        (corpus / f"DOC-{ordinal:03d}.txt").write_text(
            f"document {ordinal}", encoding="utf-8"
        )
    for ordinal in range(25):
        document = DocxDocument()
        document.add_paragraph(f"DOCX document {ordinal}")
        document.save(corpus / f"DOCX-{ordinal:03d}.docx")
        workbook = Workbook()
        workbook.active.append(["value"])
        workbook.active.append([ordinal])
        workbook.save(corpus / f"XLSX-{ordinal:03d}.xlsx")
        workbook.close()
    for ordinal in range(25):
        with fitz.open() as document:
            page = document.new_page()
            if ordinal < 20:
                page.insert_text(
                    (72, 72), f"Text PDF {ordinal} has enough meaningful characters."
                )
            document.save(corpus / f"PDF-{ordinal:03d}.pdf")


def test_verify_reports_qdrant_dimension_and_bm25_manifest(cli_environment):
    # Verify source counts independently without building another 1000-row index.
    assert ingest(cli_environment).exit_code == 0
    write_exact_corpus_shape(cli_environment.corpus)
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
    assert checks["ordinary_documents"]["actual"]["by_extension"] == {
        ".docx": 25,
        ".pdf": 25,
        ".txt": 25,
        ".xlsx": 25,
    }
    assert checks["ordinary_documents"]["actual"]["pdf"] == {
        "scanned": 5,
        "text": 20,
    }
    assert checks["knowledge_qa"]["actual"] == 1000
    assert checks["active_generation"]["actual"] == active_generation(cli_environment)
    assert checks["qdrant_dimension"]["actual"]["dimension"] == 1024
    assert checks["vector_space_compatibility"]["ok"] is True
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


def verify(environment: CLIEnvironment):
    return runner.invoke(
        cli_app,
        [
            "verify",
            "--corpus",
            str(environment.corpus),
            "--qa",
            str(environment.qa),
        ],
        env={"MODEL_PROVIDER": "fake", "APP_ENV": "test"},
    )


def verification_check(result, name: str):
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    return next(check for check in report["checks"] if check["name"] == name)


def test_verify_rejects_same_count_wrong_qdrant_child_id(cli_environment):
    assert ingest(cli_environment).exit_code == 0
    generation = active_generation(cli_environment)
    store = QdrantGenerationStore(
        QdrantSettings(
            url=cli_environment.qdrant_url,
            collection_prefix=cli_environment.collection_prefix,
        )
    )
    points, _ = store.client.scroll(
        store.collection_name(generation), limit=1, with_payload=True
    )
    store.client.set_payload(
        store.collection_name(generation),
        payload={**points[0].payload, "child_id": "wrong-child-id"},
        points=[points[0].id],
        wait=True,
    )
    store.client.close()

    check = verification_check(verify(cli_environment), "qdrant_dimension")

    assert check["ok"] is False
    assert check["actual"]["child_ids_match"] is False


def test_verify_rejects_same_count_wrong_bm25_child_id(cli_environment):
    assert ingest(cli_environment).exit_code == 0
    generation = active_generation(cli_environment)
    state_path = cli_environment.state / "bm25" / generation / "bm25.pkl"
    with state_path.open("rb") as state_file:
        state = pickle.load(state_file)
    state["chunks"][0] = replace(state["chunks"][0], id="wrong-child-id")
    with state_path.open("wb") as state_file:
        pickle.dump(state, state_file, protocol=pickle.HIGHEST_PROTOCOL)

    check = verification_check(verify(cli_environment), "bm25_manifest")

    assert check["ok"] is False
    assert check["actual"]["child_ids_match"] is False


def test_verify_rejects_active_vector_space_mismatch(cli_environment):
    assert ingest(cli_environment).exit_code == 0
    generation = active_generation(cli_environment)
    with sqlite3.connect(cli_environment.state / "index.sqlite3") as connection:
        connection.execute(
            "UPDATE index_generations SET vector_space_fingerprint = 'wrong' "
            "WHERE id = ?",
            (generation,),
        )

    check = verification_check(
        verify(cli_environment), "vector_space_compatibility"
    )

    assert check["ok"] is False
    assert "rebuild" in check["message"]


def test_verify_malformed_yaml_emits_structured_json_failure(
    cli_environment, tmp_path, monkeypatch
):
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("embedding: [unterminated", encoding="utf-8")
    monkeypatch.setattr("app.cli.load_settings", lambda: load_settings(malformed))

    result = verify(cli_environment)

    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert report["checks"][0]["name"] == "configuration"
    assert report["checks"][0]["ok"] is False
    assert result.stderr == ""


def test_verify_reports_malformed_sqlite_as_json(cli_environment):
    cli_environment.state.mkdir()
    (cli_environment.state / "index.sqlite3").write_bytes(b"not a sqlite database")

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
    report = json.loads(result.stdout)
    assert report["ok"] is False
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["active_generation"]["ok"] is False
    assert "DatabaseError" in checks["active_generation"]["message"]
    for name in ("parent_child_integrity", "qdrant_dimension", "bm25_manifest"):
        assert checks[name]["ok"] is False
        assert "not checked" in checks[name]["message"]


def test_verify_reports_multiple_active_generations_as_json(cli_environment):
    cli_environment.state.mkdir()
    with sqlite3.connect(cli_environment.state / "index.sqlite3") as connection:
        connection.execute(
            "CREATE TABLE index_generations "
            "(id VARCHAR PRIMARY KEY, status VARCHAR(16) NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO index_generations (id, status) VALUES (?, 'active')",
            [("gen-a",), ("gen-b",)],
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
    report = json.loads(result.stdout)
    assert report["ok"] is False
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["active_generation"]["ok"] is False
    assert "MultipleResultsFound" in checks["active_generation"]["message"]
    for name in ("parent_child_integrity", "qdrant_dimension", "bm25_manifest"):
        assert checks[name]["ok"] is False
        assert "not checked" in checks[name]["message"]
