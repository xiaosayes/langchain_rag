import hashlib
import os
import time
from collections.abc import Iterator

import pytest
from docker.errors import DockerException
from qdrant_client import QdrantClient
from testcontainers.core.container import DockerContainer

from app.domain.models import ChildChunk


@pytest.fixture(scope="session")
def qdrant_url() -> Iterator[str]:
    if os.environ.get("RUN_INTEGRATION") != "1":
        pytest.skip("real Qdrant integration tests are disabled")

    try:
        container = DockerContainer("qdrant/qdrant:1.15.4").with_exposed_ports(6333)
        container.start()
    except DockerException as exc:
        pytest.fail(
            "RUN_INTEGRATION=1 requires Docker for qdrant/qdrant:1.15.4; "
            f"container startup failed: {type(exc).__name__}: {exc}"
        )

    url = f"http://{container.get_container_host_ip()}:{container.get_exposed_port(6333)}"
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
