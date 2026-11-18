"""Compatibility identity for generation-scoped dense vector spaces."""

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from app.config import AppSettings


@dataclass(frozen=True)
class VectorSpace:
    """All properties that must match before dense vectors can be shared."""

    provider: str
    model: str
    dimension: int
    distance: Literal["cosine", "euclid", "dot"]

    @classmethod
    def from_settings(cls, settings: "AppSettings") -> "VectorSpace":
        return cls(
            provider=settings.model_provider,
            model=settings.embedding.model,
            dimension=settings.embedding.dimension,
            distance=settings.qdrant.distance,
        )

    @property
    def fingerprint(self) -> str:
        """Return a stable fingerprint over the canonical compatibility payload."""
        payload = json.dumps(
            {
                "provider": self.provider,
                "model": self.model,
                "dimension": self.dimension,
                "distance": self.distance,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
