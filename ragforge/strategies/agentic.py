"""Quest w4 / agentic RAG: an LLM router decides which tool to search, with what query, and when to stop.
Tools are any Strategy objects, so the same router can drive documents, graph and SQL (Quest b5)."""
from __future__ import annotations

from ..types import Hit
from ..util import parse_json
from .base import Strategy
from .hybrid import HybridRAG

ROUTER_SYS = (
    "You are a retrieval planner. Given a question, the available tools and what has been retrieved so far, "
    'reply with ONE JSON object: {"action":"search","tool":"<tool name>","query":"<search query>"} to retrieve more, '
    'or {"action":"answer"} when the retrieved passages are enough.'
)


class AgenticRAG(Strategy):
    name = "agentic"

    def __init__(self, index, llm, k=5, tools: dict[str, Strategy] | None = None,
                 descriptions: dict[str, str] | None = None, max_steps=3):
        super().__init__(index, llm, k)
        self.tools = tools or {"docs": HybridRAG(index, llm, k)}
        self.descriptions = descriptions or {n: "search documents" for n in self.tools}
        self.max_steps = max_steps
        self.default_tool = next(iter(self.tools))

    def _prompt(self, query: str, got: list[Hit]) -> str:
        tools = "\n".join(f"- {n}: {self.descriptions.get(n, '')}" for n in self.tools)
        seen = "\n".join(f"- {h.chunk.text[:100]}" for h in got) or "(nothing yet)"
        return f"Tools:\n{tools}\n\nQuestion: {query}\n\nRetrieved so far:\n{seen}"

    def retrieve(self, query: str, k: int | None = None, where=None) -> list[Hit]:
        k = k or self.k
        got: list[Hit] = []
        ids: set[str] = set()
        for _ in range(self.max_steps):
            decision = parse_json(self.llm.generate(ROUTER_SYS, self._prompt(query, got), max_tokens=120))
            if not isinstance(decision, dict):
                break
            if decision.get("action") != "search" or decision.get("tool") not in self.tools:
                if got:
                    break
                decision = {"tool": self.default_tool, "query": query}  # always search at least once
            tool = self.tools[decision["tool"]]
            for h in tool.retrieve(decision.get("query") or query, k):
                if h.chunk.id not in ids:
                    ids.add(h.chunk.id)
                    got.append(h)
        if not got:
            got = self.tools[self.default_tool].retrieve(query, k, where)
        return got[: k * 2]
