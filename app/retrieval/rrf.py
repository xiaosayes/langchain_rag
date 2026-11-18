"""Reciprocal-rank fusion for dense and lexical retrieval results."""

from dataclasses import replace
from typing import Any

from app.domain.models import SearchHit


def reciprocal_rank_fusion(
    result_sets: list[list[SearchHit]], k: int = 60
) -> list[SearchHit]:
    """Merge ranked results by child ID using one-based reciprocal ranks."""
    accumulated: dict[str, dict[str, Any]] = {}
    for results in result_sets:
        for rank, hit in enumerate(results, start=1):
            entry = accumulated.setdefault(
                hit.child_id,
                {"hit": hit, "score": 0.0, "ranks": {}, "source_scores": {}},
            )
            entry["score"] += 1.0 / (k + rank)
            entry["ranks"][hit.source] = rank
            entry["source_scores"][hit.source] = hit.score

    fused = [
        replace(
            entry["hit"],
            score=entry["score"],
            source="fused",
            metadata={
                **entry["hit"].metadata,
                "ranks": entry["ranks"],
                "source_scores": entry["source_scores"],
            },
        )
        for entry in accumulated.values()
    ]
    return sorted(fused, key=lambda hit: (-hit.score, hit.child_id))
