"""Quest w4 / corrective + self-RAG: grade what was retrieved, rewrite and retry if it is poor,
optionally fall back to another source, and refuse to answer when the answer is not grounded."""
from __future__ import annotations

from ..evaluation.judge import faithfulness
from ..types import Chunk, Hit
from .base import Strategy
from .hybrid import HybridRAG

GRADE_SYS = "You grade whether a passage helps answer a question. Reply with exactly one word: yes or no."
REWRITE_SYS = "The first search for this question found little. Rewrite it as a different, more specific search query. Output only the query."
UNKNOWN = "I don't know based on the provided context."


class CorrectiveRAG(Strategy):
    name = "corrective"

    def __init__(self, index, llm, k=5, base: Strategy | None = None, min_relevant=2, max_rounds=2,
                 fallback=None, grounding_check=True, candidates=30):
        super().__init__(index, llm, k)
        self.base = base or HybridRAG(index, llm, k, candidates)
        self.min_relevant, self.max_rounds = min_relevant, max_rounds
        self.fallback, self.grounding_check = fallback, grounding_check  # fallback: query -> list[str] (e.g. web search)

    def grade(self, query: str, hits: list[Hit]) -> list[bool]:
        return [
            self.llm.generate(GRADE_SYS, f"Question: {query}\nPassage: {h.chunk.text}\nRelevant?", max_tokens=3).strip().lower().startswith("yes")
            for h in hits
        ]

    def retrieve(self, query: str, k: int | None = None, where=None) -> list[Hit]:
        k = k or self.k
        q, kept, seen = query, [], set()
        for round_ in range(self.max_rounds + 1):
            hits = self.base.retrieve(q, k * 2, where)
            for h, ok in zip(hits, self.grade(query, hits)):
                if ok and h.chunk.id not in seen:
                    seen.add(h.chunk.id)
                    kept.append(h)
            if len(kept) >= self.min_relevant or round_ == self.max_rounds:
                break
            new_q = self.llm.generate(REWRITE_SYS, query, max_tokens=60).strip()
            if not new_q or new_q == q:
                break
            q = new_q
        if len(kept) < self.min_relevant and self.fallback:
            kept += [Hit(Chunk(f"fallback::{i}", "fallback", t), 0.0, "fallback") for i, t in enumerate(self.fallback(query))]
        return kept[:k]

    def generate(self, question: str, hits: list[Hit]) -> str:
        text = super().generate(question, hits)
        if self.grounding_check and hits:
            score = faithfulness(self.llm, text, [h.chunk.text for h in hits])
            if score is not None and score < 0.5:
                return UNKNOWN
        return text
