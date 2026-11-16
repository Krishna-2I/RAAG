import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragforge.platform.app import create_app

SAMPLE = sorted(Path("data/sample").glob("*.md"))
CFG = {"embedder": "hash:512", "llm": "extractive", "reranker": "overlap", "chunk_size": 300, "overlap": 50, "k": 3}


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(str(tmp_path), admin_key="adm")) as c:
        yield c


def make_tenant(client, name="acme", cfg=None):
    r = client.post("/admin/tenants", json={"name": name, "config": cfg or CFG}, headers={"X-Admin-Key": "adm"})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["api_key"]}


def wait(client, h, job):
    for _ in range(100):
        j = client.get(f"/v1/jobs/{job}", headers=h).json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.05)
    raise AssertionError("job timed out")


def upload(client, h, paths=SAMPLE):
    files = [("files", (p.name, p.read_bytes(), "text/markdown")) for p in paths]
    r = client.post("/v1/documents", files=files, headers=h)
    assert r.status_code == 202, r.text
    return wait(client, h, r.json()["job_id"])


def test_auth_and_admin(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/v1/config").status_code == 401
    assert client.get("/v1/config", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.post("/admin/tenants", json={"name": "x"}, headers={"X-Admin-Key": "wrong"}).status_code == 401
    assert client.post("/admin/tenants", json={"name": "x", "config": {"llm": "bad:model"}}, headers={"X-Admin-Key": "adm"}).status_code == 422


def test_admin_dashboard_session_is_http_only_and_authorizes_tenant_creation(client):
    assert client.get("/admin/session").json() == {"authenticated": False}
    assert client.post("/admin/session", json={"key": "wrong"}).status_code == 401

    login = client.post("/admin/session", json={"key": "adm"})
    assert login.status_code == 200 and login.json() == {"authenticated": True}
    cookie = login.headers["set-cookie"]
    assert "httponly" in cookie.lower() and "samesite=strict" in cookie.lower()
    assert "adm" not in login.text
    assert client.get("/admin/session").json() == {"authenticated": True}

    created = client.post("/admin/tenants", json={"name": "dashboard"})
    assert created.status_code == 200
    assert client.get("/v1/config", headers={"Authorization": f"Bearer {created.json()['api_key']}"}).status_code == 200

    assert client.delete("/admin/session").status_code == 200
    assert client.get("/admin/session").json() == {"authenticated": False}
    assert client.post("/admin/tenants", json={"name": "no longer authorized"}).status_code == 401

    secure_login = client.post("/admin/session", json={"key": "adm"}, headers={"X-Forwarded-Proto": "https"})
    assert "secure" in secure_login.headers["set-cookie"].lower()


def test_full_flow_upload_query_usage(client):
    h = make_tenant(client)
    assert client.post("/v1/query", json={"question": "hi"}, headers=h).status_code == 409  # nothing indexed yet
    job = upload(client, h)
    assert job["status"] == "done" and "chunks" in job["detail"]
    r = client.post("/v1/query", json={"question": "What does error E-4012 mean?"}, headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["strategy"] == "hybrid" and len(body["sources"]) == 3 and "E-4012" in body["answer"]
    assert any(s["document"] == "troubleshooting.md" for s in body["sources"])
    for s in ["naive", "rerank", "multi", "corrective", "agentic"]:
        assert client.post("/v1/query", json={"question": "What is RRF?", "strategy": s}, headers=h).status_code == 200, s
    use = client.get("/v1/usage", headers=h).json()
    assert use["queries_total"] == 6 and use["by_strategy"]["naive"] == 1
    assert client.post("/v1/query", json={"question": "x", "strategy": "nonsense"}, headers=h).status_code == 400
    g = client.post("/v1/query", json={"question": "x", "strategy": "graph"}, headers=h)
    assert g.status_code == 400 and "graph" in g.json()["detail"]  # tenant has no graph (build_graph is off)


def test_tenant_isolation(client):
    a, b = make_tenant(client, "a"), make_tenant(client, "b")
    upload(client, a)
    assert client.get("/v1/documents", headers=b).json()["documents"] == []
    assert client.post("/v1/query", json={"question": "E-4012"}, headers=b).status_code == 409
    job = client.post("/v1/reindex", headers=a).json()["job_id"]
    assert client.get(f"/v1/jobs/{job}", headers=b).status_code == 404  # b cannot see a's jobs


def test_switching_models_per_tenant(client):
    h = make_tenant(client)
    upload(client, h)
    # changing the LLM takes effect immediately
    assert client.put("/v1/config", json={"llm": "extractive"}, headers=h).status_code == 200
    # changing the embedder invalidates the index until it is rebuilt
    r = client.put("/v1/config", json={"embedder": "hash:256"}, headers=h).json()
    assert r["reindex_required"] is True
    r = client.post("/v1/query", json={"question": "E-4012"}, headers=h)
    assert r.status_code == 409 and r.json()["reindex_required"] is True
    assert wait(client, h, client.post("/v1/reindex", headers=h).json()["job_id"])["status"] == "done"
    assert client.post("/v1/query", json={"question": "What does error E-4012 mean?"}, headers=h).status_code == 200
    assert client.get("/v1/config", headers=h).json()["embedder"] == "hash:256"


def test_config_validation(client):
    h = make_tenant(client)
    for bad in [{"bogus": 1}, {"llm": "nope"}, {"k": 0}, {"chunk_size": 300, "overlap": 400}, {"default_strategy": "sql"}]:
        assert client.put("/v1/config", json=bad, headers=h).status_code == 422, bad


def test_rate_limit(client):
    h = make_tenant(client, cfg={**CFG, "rate_limit_per_min": 2})
    upload(client, h)
    codes = [client.post("/v1/query", json={"question": "E-4012"}, headers=h).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_upload_rules_and_delete(client):
    h = make_tenant(client)
    bad = client.post("/v1/documents", files=[("files", ("evil.exe", b"x", "application/octet-stream"))], headers=h)
    assert bad.status_code == 415
    sneaky = client.post("/v1/documents", files=[("files", ("../../escape.md", b"# hi\nsome text here.", "text/markdown"))], headers=h)
    assert sneaky.status_code == 202 and sneaky.json()["files"] == ["escape.md"]  # path components stripped
    wait(client, h, sneaky.json()["job_id"])
    assert client.get("/v1/documents", headers=h).json()["documents"] == ["escape.md"]
    d = client.delete("/v1/documents/escape.md", headers=h)
    assert d.status_code == 202 and wait(client, h, d.json()["job_id"])["status"] == "done"
    assert client.get("/v1/documents", headers=h).json()["indexed"] is False
    assert client.delete("/v1/documents/escape.md", headers=h).status_code == 404


def test_unreachable_model_backend_is_502(client, monkeypatch):
    h = make_tenant(client, cfg={**CFG, "llm": "ollama:llama3.1"})
    upload(client, h)
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    assert client.post("/v1/query", json={"question": "E-4012"}, headers=h).status_code == 502


def test_dashboard_and_strategy_list(client):
    h = make_tenant(client)
    dashboard = client.get("/dashboard").text
    assert "Ragforge" in dashboard and "Admin sign-in" in dashboard
    assert "localStorage" not in dashboard and "OPENAI_API_KEY" not in dashboard
    names = [s["name"] for s in client.get("/v1/strategies", headers=h).json()["strategies"]]
    assert {"naive", "hybrid", "rerank", "agentic", "graph"} <= set(names) and "sql" not in names
