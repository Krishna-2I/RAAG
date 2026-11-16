"""Quest f2: the vector index. Exact cosine search with NumPy, saved together with the
name of the embedder that built it, so a tenant can never query with the wrong model.
Also supports metadata filters (`where`) and a parents table for parent-child retrieval."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .bm25 import BM25, top_indices
from .embeddings import Embedder, make_embedder
from .types import Chunk, Hit


class EmbedderMismatch(RuntimeError):
    pass


class Index:
    def __init__(self, embedder: Embedder, chunks: list[Chunk], vectors: np.ndarray, parents: dict | None = None):
        self.embedder, self.chunks, self.vectors = embedder, chunks, vectors
        self.parents: dict[str, str] = parents or {}
        self._bm25: BM25 | None = None

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "Index":
        return cls(embedder, chunks, embedder.embed([c.text for c in chunks]))

    def _mask(self, where) -> np.ndarray | None:
        """`where` is a dict of exact-match metadata filters, or a function metadata -> bool."""
        if not where:
            return None
        pred = where if callable(where) else (lambda md: all(md.get(k) == v for k, v in where.items()))
        return np.array([bool(pred(c.metadata)) for c in self.chunks])

    def metadata_values(self, key: str) -> list:
        return sorted({c.metadata[key] for c in self.chunks if key in c.metadata}, key=str)

    def search_dense(self, query: str, k: int, where=None) -> list[Hit]:
        scores = self.vectors @ self.embedder.embed_query(query)
        mask = self._mask(where)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        return [Hit(self.chunks[i], float(scores[i]), "dense") for i in top_indices(scores, k) if np.isfinite(scores[i])]

    def search_bm25(self, query: str, k: int, where=None) -> list[Hit]:
        if self._bm25 is None:
            self._bm25 = BM25([c.text for c in self.chunks])
        return [Hit(self.chunks[i], s, "bm25") for i, s in self._bm25.top_k(query, k, self._mask(where))]

    def save(self, path: str | Path) -> None:
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        (d / "chunks.jsonl").write_text("\n".join(json.dumps(c.__dict__) for c in self.chunks), encoding="utf-8")
        np.save(d / "vectors.npy", self.vectors)
        if self.parents:
            (d / "parents.json").write_text(json.dumps(self.parents), encoding="utf-8")
        meta = {"embedder": self.embedder.name, "dim": int(self.vectors.shape[1]), "n_chunks": len(self.chunks)}
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None = None) -> "Index":
        d = Path(path)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if embedder is not None and embedder.name != meta["embedder"]:
            raise EmbedderMismatch(
                f"index was built with {meta['embedder']!r} but got {embedder.name!r}; re-index to switch models"
            )
        embedder = embedder or make_embedder(meta["embedder"])
        lines = (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        chunks = [Chunk(**json.loads(line)) for line in lines if line]
        parents = json.loads((d / "parents.json").read_text(encoding="utf-8")) if (d / "parents.json").exists() else {}
        return cls(embedder, chunks, np.load(d / "vectors.npy"), parents)
