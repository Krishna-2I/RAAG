"""Quest w3 / contextual + metadata RAG: context headers on chunks, and LLM-generated metadata filters."""
from __future__ import annotations

import json

from ..types import Chunk, Document, Hit
from ..util import parse_json
from .base import Strategy
from .hybrid import HybridRAG

SELF_QUERY_SYS = (
    "You turn a question into metadata filters. You are given the available metadata fields and their allowed values. "
    "Reply with one JSON object using only those fields and exact allowed values, or {} if no filter applies."
)


def doc_title(doc: Document) -> str:
    for line in doc.text.splitlines():
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return doc.metadata.get("source", doc.id)


def add_headers(chunks: list[Chunk], docs: list[Document]) -> list[Chunk]:
    """Prepend 'Document: <title>' to each chunk so the embedding knows where the text came from."""
    titles = {d.id: doc_title(d) for d in docs}
    return [
        Chunk(c.id, c.doc_id, f"Document: {titles.get(c.doc_id, c.doc_id)}\n{c.text}", {**c.metadata, "title": titles.get(c.doc_id, "")})
        for c in chunks
    ]


class MetadataRAG(Strategy):
    name = "metadata"

    def __init__(self, index, llm, k=5, base: Strategy | None = None, fields=("source",), candidates=30):
        super().__init__(index, llm, k)
        self.fields = fields
        self.base = base or HybridRAG(index, llm, k, candidates)

    def self_query(self, query: str) -> dict | None:
        allowed = {f: self.index.metadata_values(f)[:50] for f in self.fields}
        raw = self.llm.generate(SELF_QUERY_SYS, f"Fields: {json.dumps(allowed)}\nQuestion: {query}", max_tokens=100)
        parsed = parse_json(raw)
        if not isinstance(parsed, dict):
            return None
        clean = {k: v for k, v in parsed.items() if k in allowed and v in allowed[k]}
        return clean or None

    def retrieve(self, query: str, k: int | None = None, where=None) -> list[Hit]:
        k = k or self.k
        filt = where or self.self_query(query)
        hits = self.base.retrieve(query, k, filt)
        return hits or (self.base.retrieve(query, k) if filt else hits)  # never return nothing because of a bad filter
