"""Quest w2: BM25 keyword search, written out so you can see every term of the formula."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.n = k1, b, len(texts)
        self.doc_len = np.zeros(self.n)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, text in enumerate(texts):
            tokens = tokenize(text)
            self.doc_len[i] = len(tokens)
            for term, tf in Counter(tokens).items():
                self.postings[term].append((i, tf))
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n)
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            df = len(plist)
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i, tf in plist:
                norm = 1 - self.b + self.b * self.doc_len[i] / self.avgdl
                s[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return s

    def top_k(self, query: str, k: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        s = self.scores(query)
        if mask is not None:
            s = np.where(mask, s, 0.0)
        return [(int(i), float(s[i])) for i in top_indices(s, k) if s[i] > 0]
