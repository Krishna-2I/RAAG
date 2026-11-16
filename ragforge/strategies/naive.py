"""Quest w2 / naive: chunk -> embed -> top-k -> prompt. The baseline everything else must beat."""
from __future__ import annotations

from ..types import Hit
from .base import Strategy


class NaiveRAG(Strategy):
    name = "naive"

    def retrieve(self, query: str, k: int | None = None, where=None) -> list[Hit]:
        return self.index.search_dense(query, k or self.k, where)
