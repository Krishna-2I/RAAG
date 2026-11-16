"""Quest w2 / hybrid: dense + BM25, merged with Reciprocal Rank Fusion."""
from __future__ import annotations

from ..types import Hit
from .base import Strategy


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """score(id) = sum over lists of 1 / (k + rank), rank starting at 1. Uses ranks only, no score scaling."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def fuse_hits(hit_lists: list[list[Hit]], k: int, source: str = "fused", rrf_k: int = 60) -> list[Hit]:
    """RRF over several retrievals (different retrievers, or different query variants)."""
    fused = rrf_fuse([[h.chunk.id for h in hl] for hl in hit_lists], rrf_k)
    by_id = {h.chunk.id: h.chunk for hl in hit_lists for h in hl}
    return [Hit(by_id[cid], score, source) for cid, score in fused[:k]]


class HybridRAG(Strategy):
    name = "hybrid"

    def __init__(self, index, llm=None, k=5, candidates=30, rrf_k=60):
        super().__init__(index, llm, k)
        self.candidates, self.rrf_k = candidates, rrf_k

    def retrieve(self, query: str, k: int | None = None, where=None) -> list[Hit]:
        k = k or self.k
        n = max(self.candidates, k)
        dense = self.index.search_dense(query, n, where)
        sparse = self.index.search_bm25(query, n, where)
        return fuse_hits([dense, sparse], k, "hybrid", self.rrf_k)
