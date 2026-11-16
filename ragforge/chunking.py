"""Quest f1: recursive chunking with overlap.

`size` is a soft limit in characters: a chunk can run up to size + overlap because the
overlap tail from the previous chunk is prepended. Characters (not tokens) keep this
dependency-free; swap in a tokenizer later if you need exact token budgets.
"""
from __future__ import annotations

import re

import numpy as np

from .types import Chunk, Document

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not seps:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    if len(parts) == 1:
        return _split(text, size, rest)
    out: list[str] = []
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")
        out.extend(_split(piece, size, rest) if len(piece) > size else [piece])
    return out


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks: list[str] = []
    cur = ""
    for piece in _split(text.strip(), size, SEPARATORS):
        if cur and len(cur) + len(piece) > size:
            chunks.append(cur.strip())
            tail = cur[-overlap:] if overlap else ""
            space = tail.find(" ")
            tail = tail[space + 1 :] if space != -1 else tail  # start the overlap on a word boundary
            cur = tail + piece
        else:
            cur += piece
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def chunk_documents(docs: list[Document], size: int = 800, overlap: int = 120) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        for n, text in enumerate(chunk_text(doc.text, size, overlap)):
            out.append(Chunk(f"{doc.id}::{n}", doc.id, text, dict(doc.metadata)))
    return out


def semantic_chunk_text(text: str, embed, percentile: int = 25, max_chars: int = 1200) -> list[str]:
    """Quest w3 / semantic chunking: start a new chunk where adjacent sentences are least similar.
    `embed` is any function list[str] -> L2-normalized array (e.g. `embedder.embed`)."""
    sents = [s for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip()) if s.strip()]
    if len(sents) < 3:
        return [" ".join(sents)] if sents else []
    vecs = embed(sents)
    sims = (vecs[:-1] * vecs[1:]).sum(axis=1)
    cut = np.percentile(sims, percentile)
    chunks, cur = [], [sents[0]]
    for sent, sim in zip(sents[1:], sims):
        if sim < cut or len(" ".join(cur)) + len(sent) > max_chars:
            chunks.append(" ".join(cur))
            cur = [sent]
        else:
            cur.append(sent)
    chunks.append(" ".join(cur))
    return chunks


def semantic_chunk_documents(docs: list[Document], embed, percentile: int = 25) -> list[Chunk]:
    return [
        Chunk(f"{d.id}::{n}", d.id, t, dict(d.metadata))
        for d in docs
        for n, t in enumerate(semantic_chunk_text(d.text, embed, percentile))
    ]
