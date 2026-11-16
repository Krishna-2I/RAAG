"""Quest w3 / hierarchical chunking: search small pieces, return big context.
  parent-child:    index child chunks, return the parent chunk they came from
  sentence-window: index single sentences, return the sentence plus its neighbours"""
from __future__ import annotations

import re

from ..chunking import chunk_text
from ..embeddings import Embedder
from ..index import Index
from ..types import Chunk, Document, Hit
from .base import Strategy
from .hybrid import HybridRAG


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]


def build_parent_child_index(docs: list[Document], embedder: Embedder, parent_size=1500, child_size=300, child_overlap=40) -> Index:
    parents: dict[str, str] = {}
    children: list[Chunk] = []
    for doc in docs:
        for pi, ptext in enumerate(chunk_text(doc.text, parent_size, 0)):
            pid = f"{doc.id}::p{pi}"
            parents[pid] = ptext
            for ci, ctext in enumerate(chunk_text(ptext, child_size, min(child_overlap, child_size - 1))):
                children.append(Chunk(f"{pid}::c{ci}", doc.id, ctext, {**doc.metadata, "parent_id": pid}))
    idx = Index.build(children, embedder)
    idx.parents = parents
    return idx


def build_sentence_window_index(docs: list[Document], embedder: Embedder, window: int = 2) -> Index:
    chunks: list[Chunk] = []
    for doc in docs:
        sents = split_sentences(doc.text)
        for i, sent in enumerate(sents):
            lo, hi = max(0, i - window), min(len(sents), i + window + 1)
            chunks.append(Chunk(f"{doc.id}::s{i}", doc.id, sent, {**doc.metadata, "window": " ".join(sents[lo:hi])}))
    return Index.build(chunks, embedder)


class HierarchicalRAG(Strategy):
    name = "hierarchical"

    def __init__(self, index, llm=None, k=5, base: Strategy | None = None, candidates=30):
        super().__init__(index, llm, k)
        self.base = base or HybridRAG(index, llm, k, candidates)

    def retrieve(self, query: str, k: int | None = None, where=None) -> list[Hit]:
        k = k or self.k
        out, seen = [], set()
        for h in self.base.retrieve(query, k * 4, where):
            md = h.chunk.metadata
            if "parent_id" in md:
                text = self.index.parents[md["parent_id"]]
                key = md["parent_id"]
            elif "window" in md:
                text = key = md["window"]
            else:
                text, key = h.chunk.text, h.chunk.id
            if key in seen:
                continue
            seen.add(key)
            out.append(Hit(Chunk(key if "parent_id" in md else h.chunk.id, h.chunk.doc_id, text, md), h.score, "hierarchical"))
            if len(out) == k:
                break
        return out
