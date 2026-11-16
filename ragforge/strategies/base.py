"""Every RAG type implements `retrieve`; generation is shared so strategies stay comparable."""
from __future__ import annotations

from time import perf_counter

from ..index import Index
from ..llm import LLM
from ..types import Answer, Hit

SYSTEM = (
    "Answer the question using only the numbered context. Cite sources like [1]. "
    "If the context does not contain the answer, say you don't know."
)


def build_prompt(question: str, hits: list[Hit]) -> str:
    ctx = "\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits, 1))
    return f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"


class Strategy:
    name = "base"

    def __init__(self, index: Index, llm: LLM | None = None, k: int = 5):
        self.index, self.llm, self.k = index, llm, k

    def retrieve(self, query: str, k: int | None = None, where=None) -> list[Hit]:
        raise NotImplementedError

    def generate(self, question: str, hits: list[Hit]) -> str:
        if self.llm is None:
            raise RuntimeError("this strategy has no LLM configured")
        return self.llm.generate(SYSTEM, build_prompt(question, hits))

    def run(self, question: str) -> Answer:
        t0 = perf_counter()
        hits = self.retrieve(question)
        t1 = perf_counter()
        text = self.generate(question, hits) if self.llm else ""
        t2 = perf_counter()
        return Answer(question, text, hits, self.name, {"retrieve_ms": (t1 - t0) * 1e3, "generate_ms": (t2 - t1) * 1e3})
