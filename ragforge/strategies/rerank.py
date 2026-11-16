"""Quest w2 / rerank: retrieve ~30 candidates cheaply, rescore with a cross-encoder, keep top k."""
from __future__ import annotations

from functools import lru_cache

from ..bm25 import tokenize
from ..types import Hit
from .base import Strategy
from .hybrid import HybridRAG


class Reranker:
    name = "base"

    def score(self, query: str, texts: list[str]) -> list[float]:
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder

        self.model, self.name = CrossEncoder(model), f"ce:{model}"

    def score(self, query, texts):
        return [float(s) for s in self.model.predict([(query, t) for t in texts])]


class OverlapReranker(Reranker):
    """Offline stand-in for tests and demos: unigram + bigram overlap with the query."""

    name = "overlap"

    def score(self, query, texts):
        q = tokenize(query)
        qu, qb = set(q), set(zip(q, q[1:]))
        out = []
        for t in texts:
            tok = tokenize(t)
            uni = len(qu & set(tok)) / max(len(qu), 1)
            bi = len(qb & set(zip(tok, tok[1:]))) / max(len(qb), 1)
            out.append(uni + 0.5 * bi)
        return out


@lru_cache(maxsize=4)  # loading a cross-encoder is slow; a service must reuse it
def make_reranker(spec: str) -> Reranker:
    kind, _, rest = spec.partition(":")
    if kind == "ce":
        return CrossEncoderReranker(rest)
    if kind == "overlap":
        return OverlapReranker()
    raise ValueError(f"unknown reranker spec: {spec!r}")


class RerankRAG(Strategy):
    name = "rerank"

    def __init__(self, index, llm=None, k=5, reranker: Reranker | None = None, candidates=30):
        super().__init__(index, llm, k)
        self.reranker, self.candidates = reranker, candidates
        self.base = HybridRAG(index, llm, k, candidates)  # best cheap candidate generator

    def retrieve(self, query: str, k: int | None = None, where=None) -> list[Hit]:
        k = k or self.k
        cands = self.base.retrieve(query, self.candidates, where)
        scores = self.reranker.score(query, [h.chunk.text for h in cands])
        ranked = sorted(zip(cands, scores), key=lambda p: -p[1])[:k]
        return [Hit(h.chunk, s, "rerank") for h, s in ranked]
