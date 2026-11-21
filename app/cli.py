"""Thin command-line interface for complete core indexing and grounded answers."""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, NoReturn

import httpx
import typer
from openai import OpenAIError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sqlalchemy.exc import SQLAlchemyError

from app.bootstrap import build_index_builder, build_rag_service, state_directory
from app.config import load_settings
from app.db.session import create_session_factory
from app.domain.models import ParsedBlock
from app.error_sanitization import sanitized_error_message
from app.ingestion.registry import ParserRegistry
from app.ingestion.text_office import QAParser
from app.monitoring.service import MonitoringService
from app.rag.service import RAGService
from app.verification import (
    CoreVerifier,
    VerificationCheck,
    VerificationReport,
    ordinary_document_paths,
)

cli_app = typer.Typer(
    name="rag-core",
    help="Ingest, rebuild, query, and verify the enterprise RAG core.",
    no_args_is_help=True,
)
CorpusOption = Annotated[
    Path, typer.Option("--corpus", help="Ordinary corpus directory")
]
QAOption = Annotated[Path, typer.Option("--qa", help="Knowledge Q&A workbook")]


def _fail(error: Exception) -> NoReturn:
    message = sanitized_error_message(error, preserve_application_message=True)
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1) from error


def _parse_documents(
    corpus: Path, qa: Path
) -> tuple[dict[str, list[ParsedBlock]], int]:
    paths = ordinary_document_paths(corpus)
    if not corpus.is_dir():
        raise ValueError(f"Corpus directory does not exist: {corpus}")
    registry = ParserRegistry.default()
    documents: dict[str, list[ParsedBlock]] = {}
    for path in paths:
        document_id = path.stem
        if document_id in documents or document_id == "knowledge_qa":
            raise ValueError(f"Duplicate document ID: {document_id}")
        documents[document_id] = registry.parse(path, document_id)
    documents["knowledge_qa"] = QAParser().parse(qa)
    return documents, len(paths)


def _build(corpus: Path, qa: Path) -> tuple[str, int, int]:
    # Parsing never writes index state. Adapter construction validates real API keys
    # before IndexBuilder.prepare_generation or any Qdrant/BM25 mutation.
    documents, ordinary_count = _parse_documents(corpus, qa)
    builder = build_index_builder(load_settings())
    try:
        generation_id = builder.build(documents)
    finally:
        builder.qdrant.client.close()
    return generation_id, ordinary_count, len(documents["knowledge_qa"])


def _build_command(corpus: Path, qa: Path, verb: str) -> None:
    try:
        generation_id, ordinary_count, qa_count = _build(corpus, qa)
    except (
        OSError,
        RuntimeError,
        ValueError,
        SQLAlchemyError,
        ResponseHandlingException,
        UnexpectedResponse,
        httpx.HTTPError,
    ) as error:
        _fail(error)
    typer.echo(f"{verb} generation: {generation_id}")
    typer.echo(f"ordinary documents: {ordinary_count}")
    typer.echo(f"knowledge Q&A: {qa_count}")


@cli_app.command()
def ingest(corpus: CorpusOption, qa: QAOption) -> None:
    """Parse ordinary documents plus knowledge Q&A and activate a new generation."""
    _build_command(corpus, qa, "Ingested")


@cli_app.command()
def rebuild(corpus: CorpusOption, qa: QAOption) -> None:
    """Build a complete replacement without deleting the active generation first."""
    _build_command(corpus, qa, "Rebuilt")


async def stream_cli_answer(service: RAGService, question: str) -> None:
    """Stream one answer and expose generation, degradation, and source lineage."""
    generation_id = service.repository.active_generation()
    if generation_id is None:
        raise RuntimeError("No active generation; run ingest first")
    prepared = await service.prepare(question, [], generation_id)
    typer.echo(f"Generation: {generation_id}")
    typer.echo(f"Status: {'degraded' if prepared.degraded else 'ok'}")
    async for text in service.stream_answer(prepared):
        typer.echo(text, nl=False)
    typer.echo()
    typer.echo("Citations:")
    if not prepared.citations:
        typer.echo("  none")
    for citation in prepared.citations:
        typer.echo(
            f"  [{citation.label}] {citation.document_id} "
            f"{citation.parent_id}: {citation.text}"
        )


@cli_app.command()
def ask(
    question: Annotated[str, typer.Option("--question", help="Question to answer")],
) -> None:
    """Answer from the active generation and print streamed text and citations."""
    service: RAGService | None = None
    try:
        service = build_rag_service(load_settings())
        asyncio.run(stream_cli_answer(service, question))
    except (
        OSError,
        RuntimeError,
        ValueError,
        SQLAlchemyError,
        ResponseHandlingException,
        UnexpectedResponse,
        httpx.HTTPError,
        OpenAIError,
    ) as error:
        _fail(error)
    finally:
        if service is not None:
            service.retrieval_service.qdrant.client.close()


@cli_app.command("monitoring-purge")
def monitoring_purge(
    days: Annotated[
        int | None,
        typer.Option(
            "--days",
            min=1,
            help="Delete traces older than this many days (default: configured value)",
        ),
    ] = None,
) -> None:
    """Explicitly delete old monitoring traces without touching business data."""
    try:
        settings = load_settings()
        retention_days = (
            settings.monitoring.retention_days if days is None else days
        )
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        root = state_directory()
        root.mkdir(parents=True, exist_ok=True)
        service = MonitoringService(
            create_session_factory(root / "index.sqlite3"), root
        )
        deleted = service.purge(cutoff)
    except (OSError, ValueError, SQLAlchemyError) as error:
        _fail(error)
    typer.echo(f"Deleted monitoring runs: {deleted}")
    typer.echo(f"Cutoff: {cutoff.isoformat()}")


@cli_app.command()
def verify(corpus: CorpusOption, qa: QAOption) -> None:
    """Emit structured source, generation, lineage, Qdrant, and BM25 checks."""
    try:
        settings = load_settings()
    except (OSError, ValueError) as error:
        report = VerificationReport(
            [
                VerificationCheck(
                    name="configuration",
                    ok=False,
                    expected="valid application configuration",
                    actual=None,
                    message=sanitized_error_message(error),
                )
            ]
        )
    else:
        report = CoreVerifier(settings, corpus, qa).run()
    typer.echo(report.to_json())
    if not report.ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    cli_app()
