"""Quest p4 / p6: tenants, API keys, per-tenant config, usage and job records (SQLite, stdlib only).
API keys are shown once and stored only as SHA-256 hashes."""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from ..strategies import ALL_STRATEGIES, DEFAULT_RERANKER

SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants(id TEXT PRIMARY KEY, name TEXT NOT NULL, key_hash TEXT UNIQUE NOT NULL, config TEXT NOT NULL, created REAL);
CREATE TABLE IF NOT EXISTS usage(id INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT, ts REAL, strategy TEXT,
                                 retrieve_ms REAL, generate_ms REAL, n_hits INTEGER);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, tenant_id TEXT, status TEXT, detail TEXT, created REAL, updated REAL);
"""

DEFAULT_CONFIG = {
    "embedder": "st:BAAI/bge-small-en-v1.5",  # st: / ollama: / openai: (hash: for tests)
    "llm": "ollama:llama3.1",                  # ollama: / openai: / anthropic: (extractive for tests)
    "reranker": DEFAULT_RERANKER,
    "default_strategy": "hybrid",
    "chunk_size": 800,
    "overlap": 120,
    "k": 5,
    "rate_limit_per_min": 60,
    "build_graph": False,
}
KINDS = {"embedder": {"st", "ollama", "openai", "hash"}, "llm": {"ollama", "openai", "anthropic", "extractive"}, "reranker": {"ce", "overlap"}}
INT_FIELDS = {"chunk_size": (100, 4000), "overlap": (0, 1000), "k": (1, 20), "rate_limit_per_min": (1, 10000)}


def validate_config(patch: dict) -> dict:
    unknown = set(patch) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    for key, kinds in KINDS.items():
        if key in patch and str(patch[key]).partition(":")[0] not in kinds:
            raise ValueError(f"{key} must start with one of {sorted(kinds)} (e.g. 'ollama:llama3.1'), got {patch[key]!r}")
    for key, (lo, hi) in INT_FIELDS.items():
        if key in patch and not (isinstance(patch[key], int) and not isinstance(patch[key], bool) and lo <= patch[key] <= hi):
            raise ValueError(f"{key} must be an integer between {lo} and {hi}")
    if "default_strategy" in patch and (patch["default_strategy"] not in ALL_STRATEGIES or patch["default_strategy"] == "sql"):
        raise ValueError(f"default_strategy must be one of {[s for s in ALL_STRATEGIES if s != 'sql']}")
    if "build_graph" in patch and not isinstance(patch["build_graph"], bool):
        raise ValueError("build_graph must be true or false")
    if patch.get("overlap", 0) >= patch.get("chunk_size", DEFAULT_CONFIG["chunk_size"]):
        raise ValueError("overlap must be smaller than chunk_size")
    return patch


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class Store:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(SCHEMA)

    def _one(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchone()

    def _run(self, sql, args=()):
        with self.lock:
            self.db.execute(sql, args)
            self.db.commit()

    # tenants
    def create_tenant(self, name: str, config: dict | None = None) -> tuple[str, str, dict]:
        cfg = {**DEFAULT_CONFIG, **validate_config(config or {})}
        tid, key = uuid.uuid4().hex[:12], "rf_" + secrets.token_urlsafe(32)
        self._run("INSERT INTO tenants VALUES (?,?,?,?,?)", (tid, name, hash_key(key), json.dumps(cfg), time.time()))
        return tid, key, cfg

    def _tenant(self, row):
        return {"id": row[0], "name": row[1], "config": json.loads(row[2])} if row else None

    def tenant_by_key(self, key: str):
        return self._tenant(self._one("SELECT id, name, config FROM tenants WHERE key_hash=?", (hash_key(key),)))

    def get_tenant(self, tid: str):
        return self._tenant(self._one("SELECT id, name, config FROM tenants WHERE id=?", (tid,)))

    def update_config(self, tid: str, patch: dict) -> dict:
        cfg = {**self.get_tenant(tid)["config"], **validate_config(patch)}
        validate_config({"overlap": cfg["overlap"], "chunk_size": cfg["chunk_size"]})
        self._run("UPDATE tenants SET config=? WHERE id=?", (json.dumps(cfg), tid))
        return cfg

    # usage
    def log_usage(self, tid, strategy, retrieve_ms, generate_ms, n_hits):
        self._run("INSERT INTO usage(tenant_id, ts, strategy, retrieve_ms, generate_ms, n_hits) VALUES (?,?,?,?,?,?)",
                  (tid, time.time(), strategy, retrieve_ms, generate_ms, n_hits))

    def usage_summary(self, tid: str) -> dict:
        with self.lock:
            total, r, g = self.db.execute("SELECT COUNT(*), AVG(retrieve_ms), AVG(generate_ms) FROM usage WHERE tenant_id=?", (tid,)).fetchone()
            by = dict(self.db.execute("SELECT strategy, COUNT(*) FROM usage WHERE tenant_id=? GROUP BY strategy", (tid,)).fetchall())
            day = self.db.execute("SELECT COUNT(*) FROM usage WHERE tenant_id=? AND ts>?", (tid, time.time() - 86400)).fetchone()[0]
        return {"queries_total": total, "queries_last_24h": day, "avg_retrieve_ms": round(r or 0, 1),
                "avg_generate_ms": round(g or 0, 1), "by_strategy": by}

    # jobs
    def create_job(self, tid: str) -> str:
        jid = uuid.uuid4().hex[:12]
        self._run("INSERT INTO jobs VALUES (?,?,?,?,?,?)", (jid, tid, "queued", "", time.time(), time.time()))
        return jid

    def update_job(self, jid: str, status: str, detail: str = ""):
        self._run("UPDATE jobs SET status=?, detail=?, updated=? WHERE id=?", (status, detail, time.time(), jid))

    def get_job(self, jid: str, tid: str):
        row = self._one("SELECT id, status, detail FROM jobs WHERE id=? AND tenant_id=?", (jid, tid))
        return {"id": row[0], "status": row[1], "detail": row[2]} if row else None
