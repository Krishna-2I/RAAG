"""Quest f5 / b1: retrieval metrics. A chunk is relevant if it contains an evidence string,
so the same eval set keeps working when you change chunk size or strategy."""
from __future__ import annotations

import re

from ..types import Hit


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def evaluate_hits(hits: list[Hit], evidence: list[str], k: int) -> dict:
    top = [norm(h.chunk.text) for h in hits[:k]]
    ev = [norm(e) for e in evidence]
    found = [any(e in t for t in top) for e in ev]
    first = next((i for i, t in enumerate(top, 1) if any(e in t for e in ev)), 0)
    return {
        "hit": float(any(found)),
        "recall": sum(found) / len(ev) if ev else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "precision": sum(any(e in t for e in ev) for t in top) / max(len(top), 1),
    }


def aggregate(rows: list[dict]) -> dict:
    return {key: sum(r[key] for r in rows) / len(rows) for key in rows[0]} if rows else {}
