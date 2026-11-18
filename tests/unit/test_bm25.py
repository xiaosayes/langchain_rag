import hashlib
import json
import shutil
from pathlib import Path

import pytest

from app.domain.models import ChildChunk
from app.retrieval.bm25 import PersistentBM25


def _chunk(child_id: str, text: str) -> ChildChunk:
    return ChildChunk(
        id=child_id,
        parent_id=f"parent-{child_id}",
        document_id="document-1",
        ordinal=0,
        text=text,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        metadata={"child": child_id},
    )


def _chinese_chunks() -> list[ChildChunk]:
    return [
        _chunk("child-a", "差旅报销标准：住宿费、交通费按财务制度执行。"),
        _chunk("child-b", "员工请假流程与审批权限。"),
        _chunk("child-c", "Alpha BETA 123 产品使用说明。"),
    ]


def test_chinese_bm25_persists_and_reloads_without_original_chunks(tmp_path: Path):
    generation_dir = tmp_path / "generation-1"
    chunks = _chinese_chunks()
    index = PersistentBM25.build(chunks, generation_dir)
    chunks.clear()

    original_hit = index.search("差旅报销标准", top_k=1)[0]
    assert original_hit.child_id == "child-a"
    assert original_hit.source == "bm25"
    assert original_hit.parent_id == "parent-child-a"
    assert original_hit.metadata == {"child": "child-a"}

    reloaded = PersistentBM25.load(generation_dir)
    reloaded_hits = reloaded.search("差旅报销标准", top_k=3)

    assert reloaded_hits[0].child_id == original_hit.child_id
    assert reloaded_hits[0].score == pytest.approx(original_hit.score)
    assert reloaded_hits[0].text == chunks_text("child-a")

    lowercase_hits = reloaded.search("alpha beta 123", top_k=1)
    assert lowercase_hits[0].child_id == "child-c"

    shutil.rmtree(generation_dir)
    with pytest.raises(FileNotFoundError):
        PersistentBM25.load(generation_dir)
    with pytest.raises(FileNotFoundError):
        reloaded.search("差旅报销标准", top_k=1)


def chunks_text(child_id: str) -> str:
    return next(chunk.text for chunk in _chinese_chunks() if chunk.id == child_id)


@pytest.mark.parametrize(
    ("field", "incompatible_value"),
    [
        ("schema_version", 999),
        ("tokenizer", "incompatible-tokenizer"),
        ("tokenizer_version", "incompatible-version"),
    ],
)
def test_bm25_rejects_incompatible_manifests(
    tmp_path: Path, field: str, incompatible_value: object
):
    generation_dir = tmp_path / field
    PersistentBM25.build(_chinese_chunks(), generation_dir)
    manifest_path = generation_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = incompatible_value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible"):
        PersistentBM25.load(generation_dir)
