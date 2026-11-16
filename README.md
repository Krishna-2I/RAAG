# ragforge: RAG library + RAG-as-a-service

A tested Python project with two layers:

1. **Library** (`ragforge/`): ingestion, embeddings, indexes, 14 selectable RAG strategies behind one interface, and an eval harness.
2. **Service** (`ragforge/platform/`): a multi-tenant FastAPI API on top of the library, with per-tenant model switching (local or API).

The full explanation of every concept, file, workflow and technology is in the separate guide document that came with this project.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[local,server,dev]"     # local = sentence-transformers, server = FastAPI, dev = pytest
pytest                                   # 40 tests, all offline
```

## Library quick start

```bash
ragforge ingest --data ./my_docs --index .index/mine --embedder st:BAAI/bge-small-en-v1.5
ragforge ask "your question" --index .index/mine --strategy hybrid --llm ollama:llama3.1
ragforge eval --index .index/mine --eval eval/mine.jsonl -k 5 --show-misses --out docs/results.md
```

Other ingest modes: `--mode semantic | parent-child | sentence-window`, `--headers`, `--captioner ollama:llava`
(images and CSV tables), `--build-graph ollama:llama3.1` (GraphRAG). Strategies: `naive hybrid rerank rewrite multi
hyde stepback decompose hierarchical metadata corrective agentic graph sql`. Model specs: embedders `st:` `ollama:`
`openai:`; LLMs `ollama:` `openai:` `anthropic:`; keys from `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`.

## Service quick start

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
# Set the generated value as RAGFORGE_ADMIN_KEY in your shell, then:
uvicorn ragforge.platform.app:create_app --factory --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000/dashboard` to sign in with the admin key, create a workspace, upload files, monitor indexing, and ask questions. The admin key is exchanged for a short-lived HTTP-only cookie and is not stored in browser JavaScript. A workspace API key is held in page memory only and stays hidden unless you explicitly reveal or copy it. Save it if you want to reconnect later.

The dashboard creates workspaces with the offline `hash:512` embedder and `extractive` answerer by default, so the complete flow works without model downloads or provider credentials. These are demo settings. For real model use, set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or an Ollama endpoint in the server environment and select matching model specs in workspace settings. Never put provider credentials in dashboard fields, source control, or workspace configuration. Changing an embedder makes its index stale until it is rebuilt; changing an answer model takes effect on the next query.

### Deploy to Render

The repository includes a Render Blueprint in `render.yaml`. It defines a Docker web service, `/health` health check, generated admin secret, and persistent `/data` disk for tenant databases, uploads, and indexes. Persistent disks require a paid Render service; review current pricing before deployment.

1. Create a Git repository and push this project. The included `.gitignore` excludes virtual environments, generated indexes, tenant data, and `.env` secrets. Review `git status` before pushing; commit `.env.example`, never your real `.env`.
2. In Render, choose **New → Blueprint**, connect the repository, and apply the `render.yaml` configuration.
3. Wait for the service deploy to complete. In the Render service's Environment page, retrieve the generated `RAGFORGE_ADMIN_KEY` for administrator sign-in. Set model-provider credentials there as secrets if needed; do not put them in the repository or browser.
4. Open the Render service URL at `/dashboard`, create a workspace, save its one-time workspace key, upload documents, wait for the indexing job to finish, and ask questions. `/docs` provides the interactive API reference.

Render's generated URL uses HTTPS. The dashboard's admin session cookie is HTTP-only, secure on HTTPS, same-site, and expires after 12 hours. The Render disk is mounted at `/data`; do not remove it if you need to retain uploaded content and indexes.

For a local Docker run, copy `.env.example` to `.env`, generate a strong random `RAGFORGE_ADMIN_KEY`, and set it in `.env`, then run `docker compose up --build`. The Compose file maps the app to port 8000 and stores app data in a named volume.

## Verified vs not verified

- Verified by the automated tests plus a live server smoke test: chunking, BM25, RRF, index save/load and the
  embedder guard, all strategies (with a scripted fake LLM driving the LLM-based ones), SQL safety and retries,
  graph building and traversal, multimodal loading (with a fake captioner), and the full API flow including tenant
  isolation, rate limiting, upload rules, config validation, reindexing and error codes.
- **Not run** (no network or GPU where this was built): real embedding models (bge), the real cross-encoder, Ollama,
  OpenAI and Anthropic calls, real vision captioning, and the Docker build. They use standard APIs and are isolated
  behind small adapters, so the first run on your machine is where any bug would surface.
- Numbers in `docs/results.md` come from an offline stand-in embedder and reranker on an 8-document sample. They prove the
  harness works; they do not predict real-model scores. Re-run on your own documents.

## Layout

```
ragforge/
  loaders.py chunking.py embeddings.py llm.py index.py bm25.py multimodal.py util.py types.py cli.py
  strategies/   base naive hybrid rerank query_transform hierarchical contextual corrective agentic graph sql
  evaluation/   dataset metrics judge run
  platform/     store service app dashboard.html
tests/  data/sample/  eval/sample_eval.jsonl  docs/  Dockerfile  docker-compose.yml
```
