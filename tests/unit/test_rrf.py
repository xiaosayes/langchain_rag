import pytest

from app.domain.models import SearchHit
from app.retrieval.rrf import reciprocal_rank_fusion


def make_hit(child_id: str, source: str) -> SearchHit:
    return SearchHit(
        child_id=child_id,
        parent_id="parent-1",
        document_id="document-1",
        text=f"text for {child_id}",
        score=0.0,
        source=source,
        metadata={"original": child_id},
    )


def test_rrf_combines_one_based_ranks():
    dense = [make_hit("a", "dense"), make_hit("b", "dense")]
    bm25 = [make_hit("b", "bm25"), make_hit("c", "bm25")]

    fused = reciprocal_rank_fusion([dense, bm25], k=60)

    assert [hit.child_id for hit in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1].score == pytest.approx(1 / 61)
    assert fused[2].score == pytest.approx(1 / 62)
    assert fused[0].source == "fused"
    assert fused[0].metadata == {
        "original": "b",
        "ranks": {"dense": 2, "bm25": 1},
        "source_scores": {"dense": 0.0, "bm25": 0.0},
    }
    assert dense[1].source == "dense"
    assert dense[1].score == 0.0
    assert dense[1].metadata == {"original": "b"}


def test_rrf_uses_child_id_to_break_score_ties():
    fused = reciprocal_rank_fusion(
        [[make_hit("child-b", "dense")], [make_hit("child-a", "bm25")]],
        k=0,
    )

    assert [hit.child_id for hit in fused] == ["child-a", "child-b"]
    assert [hit.score for hit in fused] == [pytest.approx(1.0), pytest.approx(1.0)]
    assert fused[0].metadata["ranks"] == {"bm25": 1}
    assert fused[1].metadata["ranks"] == {"dense": 1}
