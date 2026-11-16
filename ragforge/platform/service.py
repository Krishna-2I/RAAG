"""Quest p2-p5: the service layer. Per tenant: stored files, a rebuildable index, cached models,
a rate limiter, and a background worker pool for ingestion. Model choice comes from the tenant's config
on every call, so switching a client between local and API models is a config change, not a deploy."""
from __future__ import annotations

import json
import shutil
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..chunking import chunk_documents
from ..embeddings import make_embedder
from ..index import Index
from ..llm import make_llm
from ..loaders import TEXT_EXT, load_path
from ..strategies import GraphIndex, make_strategy
from .store import Store

SUPPORTED = TEXT_EXT | {".pdf"}


class NoIndex(RuntimeError):
    pass


class ReindexRequired(RuntimeError):
    pass


class RateLimited(RuntimeError):
    def __init__(self, retry_after: int):
        super().__init__(f"rate limit exceeded, retry in {retry_after}s")
        self.retry_after = retry_after


class RagService:
    def __init__(self, data_dir: str | Path, workers: int = 2):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.dir / "platform.db")
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self._lock = threading.Lock()
        self._indexes: dict[str, Index] = {}
        self._embedders: dict[str, object] = {}
        self._calls: dict[str, deque] = defaultdict(deque)

    # ---- paths
    def files_dir(self, tid):
        d = self.dir / "tenants" / tid / "files"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def index_dir(self, tid):
        return self.dir / "tenants" / tid / "index"

    # ---- documents
    def save_files(self, tid: str, files: list[tuple[str, bytes]]) -> list[str]:
        saved = []
        for name, data in files:
            clean = Path(name).name  # strips any path components an attacker may send
            if Path(clean).suffix.lower() not in SUPPORTED:
                raise ValueError(f"unsupported file type: {clean} (use .pdf, .md or .txt)")
            (self.files_dir(tid) / clean).write_bytes(data)
            saved.append(clean)
        return saved

    def list_files(self, tid: str) -> list[str]:
        return sorted(f.name for f in self.files_dir(tid).iterdir() if f.is_file())

    def delete_file(self, tid: str, name: str) -> bool:
        path = self.files_dir(tid) / Path(name).name
        if not path.is_file():
            return False
        path.unlink()
        return True

    # ---- ingestion (background)
    def submit_ingest(self, tid: str) -> str:
        job = self.store.create_job(tid)
        self.pool.submit(self._ingest, tid, job)
        return job

    def _embedder(self, spec: str):
        with self._lock:
            if spec not in self._embedders:
                self._embedders[spec] = make_embedder(spec)
            return self._embedders[spec]

    def _ingest(self, tid: str, job: str) -> None:
        try:
            self.store.update_job(job, "running")
            cfg = self.store.get_tenant(tid)["config"]
            docs = load_path(self.files_dir(tid))
            if not docs:
                shutil.rmtree(self.index_dir(tid), ignore_errors=True)
                with self._lock:
                    self._indexes.pop(tid, None)
                self.store.update_job(job, "done", "no documents; index cleared")
                return
            chunks = chunk_documents(docs, cfg["chunk_size"], cfg["overlap"])
            idx = Index.build(chunks, self._embedder(cfg["embedder"]))
            tmp = self.index_dir(tid).with_name("index.tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            idx.save(tmp)
            if cfg["build_graph"]:
                GraphIndex.build(chunks, make_llm(cfg["llm"])).save(tmp / "graph.json")
            with self._lock:  # swap in the new index in one step
                shutil.rmtree(self.index_dir(tid), ignore_errors=True)
                tmp.rename(self.index_dir(tid))
                self._indexes.pop(tid, None)
            self.store.update_job(job, "done", f"{len(docs)} documents, {len(chunks)} chunks, embedder {cfg['embedder']}")
        except Exception as e:  # report every failure on the job instead of losing it in the thread
            self.store.update_job(job, "failed", f"{type(e).__name__}: {e}")

    # ---- querying
    def _check_rate(self, tid: str, limit: int) -> None:
        now, q = time.time(), self._calls[tid]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= limit:
            raise RateLimited(int(60 - (now - q[0])) + 1)
        q.append(now)

    def _load_index(self, tid: str, cfg: dict) -> Index:
        with self._lock:
            cached = self._indexes.get(tid)
        if cached is not None and cached.embedder.name == cfg["embedder"]:
            return cached
        d = self.index_dir(tid)
        if not (d / "meta.json").exists():
            raise NoIndex("no documents indexed yet; upload documents first")
        built_with = json.loads((d / "meta.json").read_text())["embedder"]
        if built_with != cfg["embedder"]:
            raise ReindexRequired(f"index was built with {built_with!r} but config says {cfg['embedder']!r}; call POST /v1/reindex")
        idx = Index.load(d, self._embedder(built_with))
        with self._lock:
            self._indexes[tid] = idx
        return idx

    def index_status(self, tid: str) -> dict:
        meta = self.index_dir(tid) / "meta.json"
        cfg = self.store.get_tenant(tid)["config"]
        if not meta.exists():
            return {"indexed": False, "reindex_required": False}
        built = json.loads(meta.read_text())
        return {"indexed": True, "embedder": built["embedder"], "chunks": built["n_chunks"], "reindex_required": built["embedder"] != cfg["embedder"]}

    def query(self, tid: str, question: str, strategy: str | None = None, k: int | None = None):
        cfg = self.store.get_tenant(tid)["config"]
        self._check_rate(tid, cfg["rate_limit_per_min"])
        idx = self._load_index(tid, cfg)
        g = self.index_dir(tid) / "graph.json"
        strat = make_strategy(strategy or cfg["default_strategy"], idx, make_llm(cfg["llm"]), k or cfg["k"], cfg["reranker"],
                              graph=GraphIndex.load(g) if g.exists() else None)
        ans = strat.run(question)
        self.store.log_usage(tid, ans.strategy, ans.timings["retrieve_ms"], ans.timings["generate_ms"], len(ans.hits))
        return ans
