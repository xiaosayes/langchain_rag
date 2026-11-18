"""Persistent Jieba-tokenized BM25 indexes scoped to one generation."""

import json
import pickle
from pathlib import Path
from typing import Any

import jieba  # type: ignore[import-untyped]
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from app.domain.models import ChildChunk, SearchHit

_SCHEMA_VERSION = 1
_TOKENIZER = "jieba"
_MANIFEST_NAME = "manifest.json"
_STATE_NAME = "bm25.pkl"


def _tokenize(text: str) -> list[str]:
    return [token for value in jieba.lcut(text.lower()) if (token := value.strip())]


class PersistentBM25:
    """A BM25 index whose manifest, state, and child payloads share a directory."""

    def __init__(
        self,
        generation_dir: Path,
        chunks: list[ChildChunk],
        index: BM25Okapi | None,
    ) -> None:
        self._generation_dir = generation_dir
        self._chunks = chunks
        self._index = index

    @classmethod
    def build(
        cls, chunks: list[ChildChunk], generation_dir: str | Path
    ) -> "PersistentBM25":
        directory = Path(generation_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stored_chunks = list(chunks)
        tokenized_corpus = [_tokenize(chunk.text) for chunk in stored_chunks]
        index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "tokenizer": _TOKENIZER,
            "tokenizer_version": jieba.__version__,
            "lowercase": True,
            "chunk_count": len(stored_chunks),
        }
        (directory / _MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (directory / _STATE_NAME).open("wb") as state_file:
            pickle.dump(
                {"index": index, "chunks": stored_chunks},
                state_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return cls(directory, stored_chunks, index)

    @classmethod
    def load(cls, generation_dir: str | Path) -> "PersistentBM25":
        directory = Path(generation_dir)
        manifest: dict[str, Any] = json.loads(
            (directory / _MANIFEST_NAME).read_text(encoding="utf-8")
        )
        expected = {
            "schema_version": _SCHEMA_VERSION,
            "tokenizer": _TOKENIZER,
            "tokenizer_version": jieba.__version__,
            "lowercase": True,
        }
        if any(manifest.get(field) != value for field, value in expected.items()):
            raise ValueError(
                f"Cannot load incompatible BM25 manifest from {directory}: "
                f"required {expected}, found {manifest}"
            )
        with (directory / _STATE_NAME).open("rb") as state_file:
            state = pickle.load(state_file)
        return cls(directory, state["chunks"], state["index"])

    @property
    def child_ids(self) -> frozenset[str]:
        return frozenset(chunk.id for chunk in self._chunks)

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        # A stale in-memory index must not make a removed generation searchable.
        if not (self._generation_dir / _MANIFEST_NAME).is_file():
            raise FileNotFoundError(
                f"BM25 generation directory is unavailable: {self._generation_dir}"
            )
        if self._index is None or top_k <= 0:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        scores = self._index.get_scores(query_tokens)
        ranked = sorted(
            zip(self._chunks, scores),
            key=lambda item: (-float(item[1]), item[0].id),
        )
        return [
            SearchHit(
                child_id=chunk.id,
                parent_id=chunk.parent_id,
                document_id=chunk.document_id,
                text=chunk.text,
                score=float(score),
                source="bm25",
                metadata=dict(chunk.metadata),
            )
            for chunk, score in ranked[:top_k]
        ]
