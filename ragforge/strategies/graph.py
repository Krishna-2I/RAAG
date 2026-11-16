"""Quest w5 / GraphRAG: an LLM extracts (subject, relation, object) triples from every chunk;
a query finds its entities, walks the graph a couple of hops, and the chunks on those edges are
fused with normal hybrid retrieval. The graph is a plain in-memory adjacency map saved as JSON
(zero setup). Swap in Neo4j when you need Cypher, scale, or graph algorithms."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from ..types import Chunk, Hit
from ..util import parse_json
from .base import Strategy
from .hybrid import HybridRAG, fuse_hits

EXTRACT_SYS = (
    "Extract factual relationships from the text as a JSON array of [subject, relation, object] triples. "
    "Use short, consistent entity names. Reply with only the JSON array, or [] if there are none."
)
ENTITY_SYS = "List the entities mentioned in the question as a JSON array of strings. Reply with only the array."


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).lower()).strip()


class GraphIndex:
    def __init__(self):
        self.edges: list[tuple[str, str, str, str]] = []  # subject, relation, object, chunk_id
        self.adj: dict[str, list[int]] = defaultdict(list)

    def add_edge(self, s: str, r: str, o: str, chunk_id: str) -> None:
        s, o = norm(s), norm(o)
        if not s or not o or s == o:
            return
        self.edges.append((s, str(r).strip(), o, chunk_id))
        self.adj[s].append(len(self.edges) - 1)
        self.adj[o].append(len(self.edges) - 1)

    @classmethod
    def build(cls, chunks: list[Chunk], llm) -> "GraphIndex":
        g = cls()
        for c in chunks:
            triples = parse_json(llm.generate(EXTRACT_SYS, c.text, max_tokens=400))
            for t in triples if isinstance(triples, list) else []:
                if isinstance(t, (list, tuple)) and len(t) == 3:
                    g.add_edge(t[0], t[1], t[2], c.id)
        return g

    def expand(self, seeds: set[str], depth: int = 2) -> dict[str, float]:
        """Breadth-first walk. A chunk on an edge touching a node at distance d scores 1/(d+1)."""
        dist = {s: 0 for s in seeds if s in self.adj}
        frontier, scores = list(dist), defaultdict(float)
        while frontier:
            nxt = []
            for node in frontier:
                d = dist[node]
                for ei in self.adj[node]:
                    s, _, o, cid = self.edges[ei]
                    scores[cid] += 1.0 / (d + 1)
                    other = o if node == s else s
                    if other not in dist and d + 1 <= depth:
                        dist[other] = d + 1
                        nxt.append(other)
            frontier = nxt
        return dict(scores)

    def find_entities(self, text: str) -> set[str]:
        t = norm(text)
        return {n for n in self.adj if len(n) >= 3 and re.search(rf"(?<!\w){re.escape(n)}(?!\w)", t)}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.edges), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GraphIndex":
        g = cls()
        for s, r, o, cid in json.loads(Path(path).read_text(encoding="utf-8")):
            g.add_edge(s, r, o, cid)
        return g


class GraphRAG(Strategy):
    name = "graph"

    def __init__(self, index, llm=None, k=5, graph: GraphIndex | None = None, base: Strategy | None = None, depth=2, candidates=30):
        super().__init__(index, llm, k)
        self.graph = graph
        self.depth, self.candidates = depth, candidates
        self.base = base or HybridRAG(index, llm, k, candidates)
        self._by_id = {c.id: c for c in index.chunks}

    def seeds(self, query: str) -> set[str]:
        seeds = self.graph.find_entities(query)
        if self.llm is not None:
            names = parse_json(self.llm.generate(ENTITY_SYS, query, max_tokens=100))
            if isinstance(names, list):
                seeds |= {norm(n) for n in names if norm(n) in self.graph.adj}
        return seeds

    def retrieve(self, query: str, k: int | None = None, where=None) -> list[Hit]:
        k = k or self.k
        scores = self.graph.expand(self.seeds(query), self.depth)
        graph_hits = [
            Hit(self._by_id[cid], s, "graph")
            for cid, s in sorted(scores.items(), key=lambda kv: -kv[1])[: self.candidates]
            if cid in self._by_id
        ]
        return fuse_hits([graph_hits, self.base.retrieve(query, self.candidates, where)], k, "graph")
