"""Quest w3 / query transformation: rewrite, multi-query, HyDE, step-back, decomposition.
Each mode asks the LLM for extra queries, retrieves for all of them, and fuses with RRF."""
from __future__ import annotations

from ..types import Hit
from ..util import clean_lines
from .base import Strategy
from .hybrid import HybridRAG, fuse_hits

PROMPTS = {
    "rewrite": "Rewrite the user question as one clear, standalone search query. Output only the query.",
    "multi": "Write 3 different search queries that would help answer the question, one per line. Output only the queries.",
    "stepback": "Write one broader, more general question whose answer would help answer the question. Output only the question.",
    "decompose": "Break the question into 2-4 simple sub-questions, one per line. If it is already simple, repeat it. Output only the questions.",
    "hyde": "Write a short passage (3-4 sentences) that plausibly answers the question, as if from a reference document. Output only the passage.",
}
MODES = list(PROMPTS)


class QueryTransformRAG(Strategy):
    def __init__(self, index, llm, k=5, mode="multi", base: Strategy | None = None, candidates=30):
        super().__init__(index, llm, k)
        if mode not in PROMPTS:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode, self.name = mode, mode
        self.candidates = candidates
        self.base = base or HybridRAG(index, llm, k, candidates)

    def expand(self, query: str) -> list[str]:
        if self.llm is None:
            raise RuntimeError(f"{self.mode} needs an LLM")
        raw = self.llm.generate(PROMPTS[self.mode], query, max_tokens=300)
        extra = [raw.strip()] if self.mode == "hyde" else clean_lines(raw)
        return [q for q in extra if q and q != query][:4]

    def retrieve(self, query: str, k: int | None = None, where=None) -> list[Hit]:
        k = k or self.k
        queries = [query] + self.expand(query)
        return fuse_hits([self.base.retrieve(q, self.candidates, where) for q in queries], k, self.mode)
