import hashlib
import os
import socket
import subprocess
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from docker.errors import DockerException
from qdrant_client import QdrantClient
from testcontainers.core.container import DockerContainer

from app.config import ChunkSettings, QdrantSettings
from app.db.session import create_session_factory
from app.domain.models import ChildChunk, ParsedBlock, SourceLocation
from app.retrieval.qdrant_store import QdrantGenerationStore


def _free_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _standalone_qdrant_url(binary: Path) -> Iterator[str]:
    binary = binary.resolve()
    if not binary.is_file():
        pytest.fail(f"QDRANT_BINARY does not exist: {binary}")

    with tempfile.TemporaryDirectory(prefix="qdrant-1.15.4-") as temporary_dir:
        root = Path(temporary_dir)
        http_port = _free_local_port()
        grpc_port = _free_local_port()
        config_path = root / "config.yaml"
        config_path.write_text(
            "storage:\n"
            '  storage_path: "./storage"\n'
            '  snapshots_path: "./snapshots"\n'
            "service:\n"
            '  host: "127.0.0.1"\n'
            f"  http_port: {http_port}\n"
            f"  grpc_port: {grpc_port}\n"
            "telemetry_disabled: true\n",
            encoding="utf-8",
        )
        log_path = root / "qdrant.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            try:
                process = subprocess.Popen(
                    [
                        str(binary),
                        "--config-path",
                        str(config_path),
                        "--disable-telemetry",
                    ],
                    cwd=root,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            except OSError as exc:
                pytest.fail(f"Could not launch standalone Qdrant {binary}: {exc}")

            url = f"http://127.0.0.1:{http_port}"
            try:
                client = QdrantClient(url=url, timeout=2)
                deadline = time.monotonic() + 60
                while True:
                    return_code = process.poll()
                    if return_code is not None:
                        log_file.flush()
                        logs = log_path.read_text(encoding="utf-8", errors="replace")
                        pytest.fail(
                            f"Standalone Qdrant exited with code {return_code}. "
                            f"Logs from {log_path}:\n{logs}"
                        )
                    try:
                        client.get_collections()
                        break
                    except Exception as exc:
                        if time.monotonic() >= deadline:
                            log_file.flush()
                            logs = log_path.read_text(
                                encoding="utf-8", errors="replace"
                            )
                            raise RuntimeError(
                                f"Standalone Qdrant did not become ready at {url}. "
                                f"Logs from {log_path}:\n{logs}"
                            ) from exc
                        time.sleep(0.25)
                client.close()
                yield url
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                process.wait(timeout=5)


@pytest.fixture(scope="session")
def qdrant_url() -> Iterator[str]:
    if os.environ.get("RUN_INTEGRATION") != "1":
        pytest.skip("real Qdrant integration tests are disabled")

    binary = os.environ.get("QDRANT_BINARY")
    if binary:
        yield from _standalone_qdrant_url(Path(binary))
        return

    try:
        container = DockerContainer("qdrant/qdrant:1.15.4").with_exposed_ports(6333)
        container.start()
    except DockerException as exc:
        pytest.fail(
            "RUN_INTEGRATION=1 requires Docker for qdrant/qdrant:1.15.4; "
            f"container startup failed: {type(exc).__name__}: {exc}"
        )

    url = (
        f"http://{container.get_container_host_ip()}:{container.get_exposed_port(6333)}"
    )
    try:
        client = QdrantClient(url=url, timeout=5)
        deadline = time.monotonic() + 60
        while True:
            try:
                client.get_collections()
                break
            except Exception as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Qdrant did not become ready at {url}") from exc
                time.sleep(0.25)
        client.close()
        yield url
    finally:
        container.stop()


class CountingFakeEmbeddings:
    """Deterministic embeddings that record only document embedding calls."""

    dimension = 8

    def __init__(self) -> None:
        self.embedded_texts: list[str] = []

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 127.5 - 1 for byte in digest[: cls.dimension]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def fake_embeddings() -> CountingFakeEmbeddings:
    return CountingFakeEmbeddings()


def parsed_blocks(document_id: str, text: str) -> list[ParsedBlock]:
    return [
        ParsedBlock(
            document_id=document_id,
            sequence=0,
            text=text,
            location=SourceLocation(),
            metadata={"source": f"{document_id}.txt"},
        )
    ]


@pytest.fixture
def index_builder(tmp_path: Path, qdrant_url: str, fake_embeddings):
    from app.indexing.builder import IndexBuilder
    from app.indexing.repository import IndexRepository

    repository = IndexRepository(create_session_factory(tmp_path / "index.sqlite3"))
    qdrant = QdrantGenerationStore(QdrantSettings(url=qdrant_url))
    return IndexBuilder(
        repository=repository,
        qdrant=qdrant,
        bm25_root=tmp_path / "bm25",
        embeddings=fake_embeddings,
        chunk_settings=ChunkSettings(
            parent_size=80,
            child_size=40,
            child_overlap=5,
        ),
    )


def three_test_chunks() -> list[ChildChunk]:
    texts = ["alpha document", "beta document", "gamma document"]
    return [
        ChildChunk(
            id=child_id,
            parent_id=f"parent-{suffix}",
            document_id=f"document-{suffix}",
            ordinal=ordinal,
            text=text,
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            metadata={"ordinal": ordinal},
        )
        for ordinal, (child_id, suffix, text) in enumerate(
            zip(("child-a", "child-b", "child-c"), ("a", "b", "c"), texts)
        )
    ]
