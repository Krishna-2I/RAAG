"""HTTP API. Tenants authenticate with bearer keys; only the admin creates tenants."""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
import urllib.error
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from ..strategies import ALL_STRATEGIES, NEEDS_LLM
from .service import NoIndex, RagService, RateLimited, ReindexRequired
from .store import validate_config

MAX_FILE_BYTES = 10 * 1024 * 1024
ADMIN_SESSION_COOKIE = "ragforge_admin"
ADMIN_SESSION_MAX_AGE = 12 * 60 * 60


class TenantCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    config: dict | None = None


class AdminLogin(BaseModel):
    key: str = Field(min_length=1, max_length=512)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    strategy: str | None = None
    k: int | None = Field(default=None, ge=1, le=20)


def _session_signature(admin_key: str, message: str) -> str:
    return hmac.new(admin_key.encode(), message.encode(), hashlib.sha256).hexdigest()


def _make_admin_session(admin_key: str) -> str:
    expires = str(int(time.time()) + ADMIN_SESSION_MAX_AGE)
    message = f"{expires}.{secrets.token_urlsafe(24)}"
    return f"{message}.{_session_signature(admin_key, message)}"


def _valid_admin_session(admin_key: str, token: str) -> bool:
    try:
        expires, nonce, signature = token.split(".", 2)
        if int(expires) < int(time.time()) or not nonce:
            return False
    except (ValueError, TypeError):
        return False
    message = f"{expires}.{nonce}"
    return secrets.compare_digest(signature, _session_signature(admin_key, message))


def create_app(data_dir: str | None = None, admin_key: str | None = None) -> FastAPI:
    svc = RagService(data_dir or os.environ.get("RAGFORGE_DATA", "./data"))
    admin_key = admin_key or os.environ.get("RAGFORGE_ADMIN_KEY", "")
    app = FastAPI(title="ragforge", version="0.1.0")
    app.state.service = svc

    def admin(request: Request, x_admin_key: str = Header(default="")):
        if not admin_key:
            raise HTTPException(503, "admin API disabled: set RAGFORGE_ADMIN_KEY")
        cookie = request.cookies.get(ADMIN_SESSION_COOKIE, "")
        if not secrets.compare_digest(x_admin_key, admin_key) and not _valid_admin_session(admin_key, cookie):
            raise HTTPException(401, "bad admin key")

    def tenant(authorization: str = Header(default="")) -> dict:
        key = authorization.removeprefix("Bearer ").strip()
        t = svc.store.tenant_by_key(key) if key else None
        if not t:
            raise HTTPException(401, "missing or invalid API key")
        return t

    @app.exception_handler(NoIndex)
    async def _no_index(_, e):
        return JSONResponse({"detail": str(e)}, status_code=409)

    @app.exception_handler(ReindexRequired)
    async def _reindex(_, e):
        return JSONResponse({"detail": str(e), "reindex_required": True}, status_code=409)

    @app.exception_handler(RateLimited)
    async def _rate(_, e):
        return JSONResponse({"detail": str(e)}, status_code=429, headers={"Retry-After": str(e.retry_after)})

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/admin/session")
    def create_admin_session(body: AdminLogin, request: Request, response: Response):
        if not admin_key:
            raise HTTPException(503, "admin API disabled: set RAGFORGE_ADMIN_KEY")
        if not secrets.compare_digest(body.key, admin_key):
            raise HTTPException(401, "bad admin key")
        response.set_cookie(
            ADMIN_SESSION_COOKIE,
            _make_admin_session(admin_key),
            max_age=ADMIN_SESSION_MAX_AGE,
            httponly=True,
            secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "").split(",")[0].strip() == "https",
            samesite="strict",
            path="/",
        )
        return {"authenticated": True}

    @app.get("/admin/session")
    def admin_session(request: Request):
        if not admin_key:
            raise HTTPException(503, "admin API disabled: set RAGFORGE_ADMIN_KEY")
        return {"authenticated": _valid_admin_session(admin_key, request.cookies.get(ADMIN_SESSION_COOKIE, ""))}

    @app.delete("/admin/session")
    def delete_admin_session(request: Request, response: Response):
        response.delete_cookie(
            ADMIN_SESSION_COOKIE,
            path="/",
            secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "").split(",")[0].strip() == "https",
            httponly=True,
            samesite="strict",
        )
        return {"authenticated": False}

    @app.post("/admin/tenants", dependencies=[Depends(admin)])
    def create_tenant(body: TenantCreate):
        try:
            tid, key, cfg = svc.store.create_tenant(body.name, body.config)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {"tenant_id": tid, "api_key": key, "config": cfg, "note": "store the api_key now; it cannot be shown again"}

    @app.get("/v1/config")
    def get_config(t: dict = Depends(tenant)):
        return {"config": t["config"], **svc.index_status(t["id"])}

    @app.put("/v1/config")
    def put_config(patch: dict, t: dict = Depends(tenant)):
        try:
            cfg = svc.store.update_config(t["id"], patch)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {"config": cfg, **svc.index_status(t["id"])}

    @app.get("/v1/strategies")
    def strategies(_: dict = Depends(tenant)):
        return {"strategies": [{"name": s, "needs_llm": s in NEEDS_LLM} for s in ALL_STRATEGIES if s != "sql"]}

    @app.post("/v1/documents", status_code=202)
    async def upload(files: list[UploadFile] = File(...), t: dict = Depends(tenant)):
        payload = []
        for f in files:
            data = await f.read()
            if len(data) > MAX_FILE_BYTES:
                raise HTTPException(413, f"{f.filename} is larger than {MAX_FILE_BYTES // 2**20} MB")
            payload.append((f.filename or "upload", data))
        try:
            saved = svc.save_files(t["id"], payload)
        except ValueError as e:
            raise HTTPException(415, str(e))
        return {"job_id": svc.submit_ingest(t["id"]), "files": saved}

    @app.get("/v1/documents")
    def documents(t: dict = Depends(tenant)):
        return {"documents": svc.list_files(t["id"]), **svc.index_status(t["id"])}

    @app.delete("/v1/documents/{name}", status_code=202)
    def delete_document(name: str, t: dict = Depends(tenant)):
        if not svc.delete_file(t["id"], name):
            raise HTTPException(404, "no such document")
        return {"job_id": svc.submit_ingest(t["id"])}

    @app.post("/v1/reindex", status_code=202)
    def reindex(t: dict = Depends(tenant)):
        return {"job_id": svc.submit_ingest(t["id"])}

    @app.get("/v1/jobs/{job_id}")
    def job(job_id: str, t: dict = Depends(tenant)):
        j = svc.store.get_job(job_id, t["id"])
        if not j:
            raise HTTPException(404, "no such job")
        return j

    @app.post("/v1/query")
    def query(body: QueryRequest, t: dict = Depends(tenant)):
        if body.strategy is not None:
            try:
                validate_config({"default_strategy": body.strategy})
            except ValueError as e:
                raise HTTPException(400, str(e))
        try:
            ans = svc.query(t["id"], body.question, body.strategy, body.k)
        except ValueError as e:  # e.g. the graph strategy on a tenant without a graph
            raise HTTPException(400, str(e))
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            raise HTTPException(502, f"model backend unreachable ({t['config']['llm']}): {e}")
        return {
            "answer": ans.text,
            "strategy": ans.strategy,
            "sources": [{"id": h.chunk.id, "document": h.chunk.metadata.get("source", h.chunk.doc_id), "score": round(h.score, 4),
                         "text": h.chunk.text} for h in ans.hits],
            "timings_ms": {k.removesuffix("_ms"): round(v, 1) for k, v in ans.timings.items()},
        }

    @app.get("/v1/usage")
    def usage(t: dict = Depends(tenant)):
        return svc.store.usage_summary(t["id"])

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard():
        return (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")

    return app
