# ragforge: RAG library + RAG-as-a-service

A tested Python project with two layers:

1. **Library** (`ragforge/`): ingestion, embeddings, indexes, 14 selectable RAG strategies behind one interface, and an eval harness.
2. **Service** (`ragforge/platform/`): a multi-tenant FastAPI API on top of the library, with per-tenant model switching (local or API).

The project can be downloaded from Git and run locally. It includes a command-line RAG workflow and an optional local web service/dashboard.

## Download and install

Requires Python 3.10 or newer and Git. Clone the repository:

```bash
git clone https://github.com/Krishna-2I/RAAG.git
cd RAAG
python -m venv .venv
```

On Windows PowerShell, install the lightweight CLI, service, and test dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[server,dev]"
```

On macOS/Linux, activate the environment and install the same extras:

```bash
source .venv/bin/activate
python -m pip install -e ".[server,dev]"
```

Run the automated tests with `.\.venv\Scripts\python.exe -m pytest` on Windows or `python -m pytest` on macOS/Linux. They run offline.

## Try the offline CLI demo

This uses bundled sample docs, the built-in test embedder, and an extractive answerer. It needs no model download or API key.

Windows PowerShell:

```powershell
.\.venv\Scripts\ragforge.exe ingest --data data\sample --index .index\demo --embedder hash:512 --chunk-size 300 --overlap 50
.\.venv\Scripts\ragforge.exe ask "What does error E-4012 mean?" --index .index\demo --strategy hybrid --llm extractive --reranker overlap -k 3
.\.venv\Scripts\ragforge.exe eval --index .index\demo --eval eval\sample_eval.jsonl -k 3 --reranker overlap --show-misses
```

macOS/Linux (after activating `.venv`):

```bash
ragforge ingest --data data/sample --index .index/demo --embedder hash:512 --chunk-size 300 --overlap 50
ragforge ask "What does error E-4012 mean?" --index .index/demo --strategy hybrid --llm extractive --reranker overlap -k 3
ragforge eval --index .index/demo --eval eval/sample_eval.jsonl -k 3 --reranker overlap --show-misses
```

## Use your own documents

Put PDF, Markdown, or text files in a folder such as `my_docs`, then ingest them. The `hash:512` embedder is only for an offline demo; for semantic search, install the larger local-model extra:

```bash
python -m pip install -e ".[local]"
ragforge ingest --data ./my_docs --index .index/mine --embedder st:BAAI/bge-small-en-v1.5
ragforge ask "your question" --index .index/mine --strategy hybrid --llm ollama:llama3.1
```

The embedding model downloads on first use. Ollama must be installed and running for `ollama:llama3.1`. OpenAI and Anthropic models are also supported; provide `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in the environment of the process running Ragforge. Never put real keys in source files or workspace configuration.

## Run the local web service

Windows PowerShell:

```powershell
$env:RAGFORGE_ADMIN_KEY = (.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))")
.\.venv\Scripts\python.exe -m uvicorn ragforge.platform.app:create_app --factory --host 127.0.0.1 --port 8000
```

macOS/Linux:

```bash
export RAGFORGE_ADMIN_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m uvicorn ragforge.platform.app:create_app --factory --host 127.0.0.1 --port 8000
```

Keep the terminal open, then visit `http://localhost:8000/dashboard`. Sign in with the admin key from that terminal, create a workspace, upload PDF/Markdown/text documents, wait for indexing to finish, then ask questions. The admin key is exchanged for a short-lived HTTP-only cookie; a workspace API key stays in page memory and is hidden unless you explicitly reveal or copy it. The default workspace settings (`hash:512` and `extractive`) work without real models. Press **Ctrl+C** in the server terminal to stop it.

For local Docker use, copy `.env.example` to `.env`, set a strong `RAGFORGE_ADMIN_KEY`, and run `docker compose up --build`. Docker is optional; the Python setup above is the simplest way to get started.

Other ingest modes include `--mode semantic | parent-child | sentence-window`, `--headers`, `--captioner ollama:llava` (images and CSV tables), and `--build-graph ollama:llama3.1` (GraphRAG). Strategies include `naive`, `hybrid`, `rerank`, `rewrite`, `multi`, `hyde`, `stepback`, `decompose`, `hierarchical`, `metadata`, `corrective`, `agentic`, `graph`, and `sql`.

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
