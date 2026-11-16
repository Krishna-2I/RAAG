"""Quest f2: embedders. One interface, several backends, so a tenant can switch local <-> API.

Spec strings (also stored inside every saved index):
  st:BAAI/bge-small-en-v1.5       local sentence-transformers model
  ollama:nomic-embed-text         local Ollama server (OLLAMA_HOST, default http://localhost:11434)
  openai:text-embedding-3-small   OpenAI-compatible API (OPENAI_API_KEY, OPENAI_BASE_URL)
  hash:512                        offline hashing embedder, only for tests and demos
"""
from __future__ import annotations

import json
import os
import urllib.request
import zlib

import numpy as np

from .bm25 import tokenize


def normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def post_json(url: str, payload: dict, headers: dict | None = None, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class Embedder:
    name = "base"

    def embed(self, texts: list[str]) -> np.ndarray:  # documents -> (n, dim), L2-normalized
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash:{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = tokenize(text)
            for feat in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]:
                h = zlib.crc32(feat.encode())
                out[row, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        out = np.sign(out) * np.sqrt(np.abs(out))
        return normalize(out)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: str, query_prefix: str = "", doc_prefix: str = ""):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)
        self.name = f"st:{model}"
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix  # e5 models need "query: " / "passage: "

    def embed(self, texts: list[str]) -> np.ndarray:
        return normalize(self.model.encode([self.doc_prefix + t for t in texts], batch_size=32))

    def embed_query(self, query: str) -> np.ndarray:
        return normalize(self.model.encode([self.query_prefix + query]))[0]


class OllamaEmbedder(Embedder):
    def __init__(self, model: str):
        self.model = model
        self.name = f"ollama:{model}"
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs: list = []
        for i in range(0, len(texts), 32):
            resp = post_json(f"{self.host}/api/embed", {"model": self.model, "input": texts[i : i + 32]})
            vecs.extend(resp["embeddings"])
        return normalize(np.array(vecs))


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str):
        self.model = model
        self.name = f"openai:{model}"
        self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.key = os.environ.get("OPENAI_API_KEY", "")

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs: list = []
        for i in range(0, len(texts), 64):
            resp = post_json(
                f"{self.base}/embeddings",
                {"model": self.model, "input": texts[i : i + 64]},
                {"Authorization": f"Bearer {self.key}"},
            )
            vecs.extend(d["embedding"] for d in sorted(resp["data"], key=lambda d: d["index"]))
        return normalize(np.array(vecs))


def make_embedder(spec: str) -> Embedder:
    kind, _, rest = spec.partition(":")
    if kind == "hash":
        return HashingEmbedder(int(rest or 512))
    if kind == "st":
        return SentenceTransformerEmbedder(rest)
    if kind == "ollama":
        return OllamaEmbedder(rest)
    if kind == "openai":
        return OpenAIEmbedder(rest)
    raise ValueError(f"unknown embedder spec: {spec!r}")
