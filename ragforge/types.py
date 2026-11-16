from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    source: str = ""  # which retriever produced this score: dense / bm25 / hybrid / rerank


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    strategy: str
    timings: dict = field(default_factory=dict)
