"""Quest b1: LLM-as-judge for generation quality (faithfulness, answer relevance), scored 0..1."""
from __future__ import annotations

import re

from ..llm import LLM

_FAITH = (
    "You grade whether an answer is fully supported by the context. Reply with one digit 1-5: "
    "5 = every claim supported, 1 = mostly unsupported."
)
_REL = "You grade whether an answer addresses the question. Reply with one digit 1-5: 5 = fully, 1 = not at all."


def _grade(llm: LLM, system: str, prompt: str) -> float | None:
    m = re.search(r"[1-5]", llm.generate(system, prompt, max_tokens=8))
    return (int(m.group()) - 1) / 4 if m else None


def faithfulness(llm: LLM, answer: str, contexts: list[str]) -> float | None:
    return _grade(llm, _FAITH, "Context:\n" + "\n---\n".join(contexts) + f"\n\nAnswer:\n{answer}\n\nScore:")


def answer_relevance(llm: LLM, question: str, answer: str) -> float | None:
    return _grade(llm, _REL, f"Question:\n{question}\n\nAnswer:\n{answer}\n\nScore:")
