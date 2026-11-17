# Ragforge: Learning Roadmap and Project Guide

This guide is for learning the project from the ground up. You do not need to
understand every retrieval strategy before you can run Ragforge. Start with the
prerequisites, run the offline demo, and then trace one question through the
code.

## 1. Topics to learn first

Learn these in order. For each topic, the goal is enough to follow this code,
not to become an expert before moving on.

### A. Python and command-line foundations

1. **Using a terminal** — current directory, paths, files, command arguments,
   environment variables, and starting/stopping a long-running process.
2. **Python 3 basics** — variables, functions, classes, imports, exceptions,
   lists, dictionaries, comprehensions, and type annotations.
3. **Virtual environments and packages** — what `venv` isolates, how `pip`
   installs packages, and how `pyproject.toml` describes a package.
4. **Reading a Python project** — package directories, `__init__.py`,
   `__main__.py`, tests, and following a function call across modules.
5. **Git basics** — clone, status, diff, commit, push, and why generated files
   and credentials should not go in source control.

You can run the demo after learning these. The rest of the topics explain why
the application behaves as it does.

### B. Data and search foundations

6. **Text documents and metadata** — a document's content versus metadata such
   as filename, page number, and modality.
7. **Chunking** — why long documents are split into smaller passages; chunk
   size and overlap tradeoffs; why this implementation measures characters,
   not model tokens.
8. **Tokenization and inverted indexes** — how words in passages are mapped to
   their locations, which powers keyword search.
9. **BM25** — a lexical ranking method that favors useful, relatively rare
   query terms and normalizes for document length.
10. **Vectors and embeddings** — representing text as a list of numbers and
    using learned representations to find semantically related text.
11. **Cosine similarity and normalization** — comparing vectors by direction;
    with normalized vectors, cosine similarity is a dot product.
12. **Top-k retrieval** — returning the best `k` passages and understanding
    that the right passage can still be missed or ranked too low.
13. **Hybrid search and rank fusion** — combining complementary retrievers
    without assuming their raw scores share a scale.
14. **Reranking** — using a more expensive query-plus-passage model to reorder
    a shortlist from a cheaper first-stage search.

### C. Language models and RAG

15. **HTTP and JSON APIs** — requests, responses, headers, status codes, and
    why remote model calls can fail.
16. **Prompt/context construction** — how retrieved passages are placed in an
    LLM prompt and why answers should cite those passages.
17. **RAG (retrieval-augmented generation)** — retrieval finds evidence;
    generation writes an answer from that evidence.
18. **Grounding and hallucinations** — a plausible LLM answer is not proof
    that the source documents support it.
19. **Model adapters** — a common interface can hide differences among local
    models, API providers, and the offline demo implementation.
20. **Secrets and environment variables** — API credentials belong in the
    process environment, not source files, commits, or user-editable settings.

### D. Service and quality foundations

21. **REST APIs and FastAPI** — routes, request models, dependency injection,
    validation, and response codes.
22. **Authentication and multi-tenancy** — admin credentials versus workspace
    API keys, and how each workspace's data is separated.
23. **SQLite and file storage** — relational records versus uploaded files and
    serialized vector indexes.
24. **Background work and threads** — why uploads return a job ID while
    indexing continues, and how to poll for completion.
25. **Testing with pytest** — fixtures, temporary directories, fake model
    responses, assertions, and offline integration tests.
26. **Evaluation metrics** — Hit@k, Recall@k, MRR, Precision@k, latency, and
    how to read retrieval misses.
27. **Optional advanced topics** — semantic chunking, parent-child retrieval,
    query transformations, agentic routing, knowledge graphs, text-to-SQL, and
    image/table captioning.

## 2. What the project does

Ragforge contains two related ways to use one RAG library:

- The **CLI** ingests documents into an index, answers questions against that
  index, and evaluates retrieval quality.
- The **local web service** exposes tenant/workspace management, document
  upload, background indexing, and question answering through HTTP and a
  browser dashboard.

Both use the same core concepts: documents become chunks; chunks are indexed;
retrieval returns ranked evidence; an answerer can use that evidence to answer.

## 3. The RAG flow, end to end

```text
PDF / Markdown / text
          |
          v
   load_path() ------> Document(text, id, metadata)
          |
          v
   chunk_documents() -> Chunk(text, id, document id, metadata)
          |
          +----------------------------+
          |                            |
          v                            v
   embed each chunk             build keyword index
          |                         (BM25, on use)
          v                            |
   vectors + chunk records             |
          |                            |
          +------------- save ----------+
                        |
                        v
Question -> embed question + keyword search -> strategy ranks/fuses hits
                        |
                        v
             numbered context prompt
                        |
                        v
            extractive answerer or LLM
                        |
                        v
              answer plus source hits
```

In the service, uploaded files are stored per workspace and indexing runs as a
background job. A later question loads that workspace's index and configuration,
checks its query limit, runs the chosen strategy, and logs usage.

## 4. Important concepts in this implementation

### Document, chunk, hit, answer

The shared data structures are in [`ragforge/types.py`](../ragforge/types.py):

- `Document` is loaded source content and metadata.
- `Chunk` is a searchable passage derived from a document.
- `Hit` is a chunk returned by retrieval, with a score and the retrieval
  source.
- `Answer` bundles the question, answer text, hits, chosen strategy, and
  timings.

The application passes these objects between loaders, chunkers, indexes,
strategies, and the CLI/API.

### Loading and chunking

[`ragforge/loaders.py`](../ragforge/loaders.py) reads `.txt`, `.md`,
`.markdown`, and text-based PDFs. A PDF page with extracted text becomes a
document with source and page metadata. Scanned PDFs do not automatically get
OCR in this loader.

[`ragforge/chunking.py`](../ragforge/chunking.py) splits text at paragraph,
line, sentence-ish, then word boundaries. Overlap repeats a tail from the
previous chunk so evidence near a boundary is less likely to get separated.
The configured size is a **soft character limit**, not a token budget.

Chunk size is a tradeoff:

- Smaller chunks tend to be more precise, but can lose surrounding context.
- Larger chunks keep context, but can match a query less precisely.
- Overlap can preserve boundary context, but also creates duplicate text and
  more chunks.

### Embeddings and dense search

[`ragforge/embeddings.py`](../ragforge/embeddings.py) defines a common
`Embedder` interface and a `make_embedder()` factory:

- `hash:512` is deterministic, offline, and for testing/demo use. It hashes
  tokens and token pairs into a fixed-length vector; it is **not** a learned
  semantic model.
- `st:MODEL` uses Sentence Transformers locally and downloads/loads a model.
- `ollama:MODEL` calls the configured Ollama server.
- `openai:MODEL` calls an OpenAI-compatible embeddings API.

[`ragforge/index.py`](../ragforge/index.py) embeds every chunk at index-build
time. Dense search embeds a question and scores each stored vector with a dot
product. The vectors are normalized, so this is cosine similarity. This
implementation searches exactly with NumPy; despite the project topic/file
history, it does not currently implement an HNSW index.

An index stores:

- `chunks.jsonl`: chunk text and metadata.
- `vectors.npy`: dense vectors.
- `meta.json`: embedder name, vector dimension, and chunk count.
- Optional `parents.json`: parent text for hierarchical retrieval.

The embedder name is saved because vectors created by different models or
dimensions cannot safely be mixed. If you change the embedder, rebuild the
index.

### Keyword search and hybrid retrieval

[`ragforge/bm25.py`](../ragforge/bm25.py) tokenizes text, builds term postings,
and computes BM25 keyword scores. This is especially helpful for exact names,
IDs, and error codes; unlike semantic embeddings, it does not understand
synonyms.

The `naive` strategy searches dense vectors only. `hybrid` runs dense and BM25
search and combines their ranked lists with **Reciprocal Rank Fusion (RRF)**.
For each ranking, a chunk receives `1 / (60 + rank)`; scores are added across
lists. RRF uses ranks instead of directly comparing the unrelated score scales
from dense search and BM25.

The `rerank` strategy first gets a candidate shortlist with hybrid retrieval,
then scores each query/passage pair:

- `overlap` is a lightweight lexical stand-in useful for offline demos/tests.
- `ce:MODEL` uses a Sentence Transformers cross-encoder and may require a model
  download and more compute.

### Answer generation

[`ragforge/strategies/base.py`](../ragforge/strategies/base.py) provides the
common strategy lifecycle:

1. `retrieve()` finds evidence chunks.
2. `build_prompt()` numbers the evidence and includes the question.
3. `generate()` sends the prompt to the selected answerer.
4. `run()` returns an `Answer` and measures retrieval and generation timings.

[`ragforge/llm.py`](../ragforge/llm.py) has a common `LLM` interface and
`make_llm()` factory:

- `extractive` selects useful sentences from retrieved text; it is an offline
  demo answerer, not a general-purpose language model.
- `ollama:MODEL`, `openai:MODEL`, and `anthropic:MODEL` call real model
  backends.

The shared prompt asks the model to answer only from numbered context and cite
it. This instruction helps, but does not guarantee correctness; inspect the
returned source passages.

### The other strategies

The factory and strategy names are in
[`ragforge/strategies/__init__.py`](../ragforge/strategies/__init__.py).
Strategies share the same retrieval interface, so callers can switch strategies
without changing the CLI/API flow.

| Strategy | What it changes |
|---|---|
| `naive` | Dense top-k retrieval; useful as the simple baseline. |
| `hybrid` | Dense plus BM25 search, combined with RRF. |
| `rerank` | Hybrid shortlist, then a reranker reorders candidates. |
| `rewrite` | Ask an LLM to rewrite the question before searching. |
| `multi` | Ask an LLM for several search queries, search all, and fuse results. |
| `stepback` | Search with a broader question as well as the original. |
| `decompose` | Split a complex question into simpler searches. |
| `hyde` | Generate a hypothetical answer passage and search with it. |
| `hierarchical` | Search small child chunks but return parent text, or search sentences and return neighboring context. |
| `metadata` | Ask an LLM to choose from allowed metadata filters, then search. |
| `corrective` | Grade retrieved chunks, rewrite and retry weak searches, and optionally check answer faithfulness. |
| `agentic` | Have an LLM choose a search tool/query for a bounded number of steps. |
| `graph` | Combine entity/relationship graph traversal with hybrid text search. Ingestion must build the graph first. |
| `sql` | Ask an LLM for a read-only SQL query over a supplied SQLite database. |

Most of the advanced strategies need an LLM and may issue multiple model calls.
They are not automatically better; compare them using the evaluation workflow.

## 5. Using the CLI

The CLI is defined in [`ragforge/cli.py`](../ragforge/cli.py), and the installed
`ragforge` command points to its `main()` function. The three commands are:

### `ingest`: build an index

```text
ragforge ingest --data FOLDER --index INDEX_PATH --embedder EMBEDDER
```

The CLI loads files, chooses a chunking mode, creates embeddings, builds an
`Index`, and saves it. Optional flags select semantic, parent-child, or
sentence-window chunking, document headers, image/table captioning, and graph
construction.

### `ask`: retrieve and answer

```text
ragforge ask "question" --index INDEX_PATH --strategy hybrid --llm extractive -k 3
```

The CLI loads the index, constructs the selected strategy and answerer, prints
the answer and source hits, and reports retrieval/generation time. The question
and selected strategy must match the data and capabilities of the index; for
example, graph search needs an index with a built graph.

### `eval`: measure retrieval

```text
ragforge eval --index INDEX_PATH --eval eval/sample_eval.jsonl -k 3 --show-misses
```

The JSONL evaluation file has one object per question, with a `question` and
`evidence` field; evidence is a phrase expected in a relevant retrieved chunk.
[`ragforge/evaluation/metrics.py`](../ragforge/evaluation/metrics.py) computes:

- **Hit@k**: did any top-k chunk contain any evidence?
- **Recall@k**: what share of evidence phrases appeared in the top-k?
- **MRR**: reciprocal rank of the first evidence-containing chunk.
- **Precision@k**: share of returned chunks containing any evidence.
- **Latency**: average retrieval time for the test questions.

The current evaluator uses evidence substring matching after basic whitespace
and case normalization. It is a useful, repeatable retrieval check, not a full
semantic grading system. Optional LLM-as-judge scores assess faithfulness and
answer relevance, but they are themselves model judgments.

## 6. Using the local web service

The service is optional. Its main components are:

- [`ragforge/platform/app.py`](../ragforge/platform/app.py) defines HTTP routes,
  request validation, authentication, and error responses.
- [`ragforge/platform/service.py`](../ragforge/platform/service.py) coordinates
  storage, background ingestion, index/model caches, query limits, and RAG.
- [`ragforge/platform/store.py`](../ragforge/platform/store.py) stores tenants,
  hashed workspace keys, config, jobs, and usage in SQLite.
- [`ragforge/platform/dashboard.html`](../ragforge/platform/dashboard.html)
  provides the browser workflow.

### Service flow

```text
Admin signs in
    -> server checks RAGFORGE_ADMIN_KEY
    -> server returns a signed, HTTP-only session cookie
    -> admin creates workspace
    -> one-time workspace key is returned; only its hash is stored
    -> user uploads files with workspace bearer key
    -> server saves files and queues an indexing job
    -> worker loads, chunks, embeds, and saves the workspace index
    -> user asks question with workspace bearer key
    -> service retrieves, answers, and records usage
```

Admin and workspace keys serve different roles. The admin key can create
workspaces. A workspace key authenticates document and query operations for
that workspace. The dashboard does not save these keys in browser storage;
the admin session cookie is HTTP-only, and the workspace key remains in page
memory unless deliberately revealed/copied.

Important endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check. |
| `POST /admin/session` | Exchange admin key for a browser session cookie. |
| `POST /admin/tenants` | Create a workspace; requires admin authorization. |
| `GET/PUT /v1/config` | Read or update workspace settings. |
| `POST/GET/DELETE /v1/documents` | Upload, list, or delete workspace files. |
| `POST /v1/reindex` | Queue an index rebuild. |
| `GET /v1/jobs/{job_id}` | Check background indexing status. |
| `POST /v1/query` | Ask a question and receive answer/source hits. |
| `GET /v1/strategies` | List available API strategies. |
| `GET /v1/usage` | Read workspace query and timing statistics. |
| `GET /dashboard` | Serve the browser interface. |

The default storage directory is `./data` and can be changed with
`RAGFORGE_DATA`. In this local project, the SQLite database and workspace
files/indexes are runtime data, not source code. Keep backups of documents and
important indexes.

## 7. A useful code-reading order

Follow one simple question through these files:

1. [`ragforge/types.py`](../ragforge/types.py) — the objects passed between layers.
2. [`ragforge/loaders.py`](../ragforge/loaders.py) — source files become `Document`s.
3. [`ragforge/chunking.py`](../ragforge/chunking.py) — documents become `Chunk`s.
4. [`ragforge/embeddings.py`](../ragforge/embeddings.py) — text becomes vectors.
5. [`ragforge/index.py`](../ragforge/index.py) — vectors/chunks are searched and persisted.
6. [`ragforge/bm25.py`](../ragforge/bm25.py) — keyword retrieval.
7. [`ragforge/strategies/base.py`](../ragforge/strategies/base.py) — the shared retrieve/generate/run lifecycle.
8. [`ragforge/strategies/hybrid.py`](../ragforge/strategies/hybrid.py) — a concrete strategy and RRF fusion.
9. [`ragforge/llm.py`](../ragforge/llm.py) — answer model adapters.
10. [`ragforge/cli.py`](../ragforge/cli.py) — how user commands connect the components.
11. [`ragforge/evaluation/`](../ragforge/evaluation/) — how retrieval is measured.
12. [`ragforge/platform/app.py`](../ragforge/platform/app.py), then `service.py`, then `store.py` — how the same library is exposed as a service.
13. [`tests/`](../tests/) — executable examples of expected behavior and edge cases.

When reading a function, ask: What does it receive? What does it return? Which
module calls it? Which test demonstrates it? This keeps the project from
feeling like a list of unrelated files.

## 8. Suggested hands-on learning plan

1. **Run the sample demo** using the commands in the README. Keep the output
   and note the answer, source chunks, and timings.
2. **Read one sample document**, then run ingest. Compare the original with
   the chunks in `.index/demo/chunks.jsonl`.
3. **Inspect an index**: open `meta.json`, inspect one chunk record, and learn
   what shape is stored in `vectors.npy`.
4. **Compare retrieval methods**: ask the same question with `naive`, `hybrid`,
   and `rerank`. Explain which source each method retrieved and why.
5. **Read an evaluation miss** from `--show-misses`. Decide whether the cause
   is missing evidence, chunking, keyword/dense retrieval, or ranking.
6. **Change chunk size/overlap**, rebuild to a different index path, then rerun
   the same evaluation. Compare measurements rather than relying on one answer.
7. **Run the API locally** and use the dashboard with `hash:512` and
   `extractive`. Upload one sample file, watch its job complete, and query it.
8. **Follow one HTTP request** from its route in `app.py` through `service.py`
   and into a strategy.
9. **Read the matching test** before modifying behavior. Run the smallest
   relevant test file, then the full test suite.
10. Only after the offline path makes sense, try real embeddings/LLMs. Check
    model downloads, API costs, and where document text is sent.

## 9. Testing and what the tests do not prove

Run the suite from the repository root:

```bash
python -m pytest
```

The tests cover chunking, BM25, index save/load, retrieval strategies,
evaluation, SQL safety properties, multimodal loading with a fake captioner,
and API tenant/auth/upload/config/job behavior. They use offline or scripted
models where possible, so the suite does not require model downloads or paid
API calls.

Passing tests do not prove that a particular real embedding model, LLM
provider, Ollama installation, or local hardware setup works. Try your chosen
backend separately and inspect logs/errors. Sample evaluation scores are not a
promise of quality on your own documents.

## 10. Limits and common misconceptions

- **The hash embedder is not semantic AI.** It is a deterministic offline
  stand-in that makes the pipeline runnable and testable.
- **Retrieval and generation are different failure points.** If the relevant
  passage is absent from the source hits, changing the answer model may not
  fix the root problem.
- **An index belongs to its embedder.** Rebuild after changing embedding model
  or embedding dimensions.
- **PDF support extracts text; it is not OCR.** Scanned image-only pages may
  yield no searchable text.
- **The default chunk size counts characters.** It is not guaranteed to match
  an LLM's token limit.
- **The default dense index is exact NumPy search.** The code does not currently
  provide HNSW/ANN search.
- **Not all strategies have the same prerequisites.** Query transforms,
  metadata, corrective, agentic, graph construction, and SQL need an LLM;
  graph also needs a graph built at ingest; SQL needs a user-supplied SQLite
  database.
- **The local API is a learning project, not automatically a production
  service.** Understand its authentication, storage, backup, model, and
  availability requirements before exposing it beyond your computer.
- **Model APIs receive document context.** Review provider privacy, credentials,
  and usage costs before sending sensitive content.

## 11. Glossary

| Term | Plain-language meaning |
|---|---|
| RAG | Retrieve source passages, then use them as context for an answer. |
| Corpus | The collection of documents being searched. |
| Chunk | A passage created from a larger document. |
| Metadata | Descriptive fields attached to a document or chunk. |
| Embedding | A numerical vector representing text. |
| Dense retrieval | Search by vector similarity. |
| Sparse/lexical retrieval | Search by matching token terms, such as BM25. |
| Candidate | A passage retrieved for possible use in the final answer. |
| Reranker | A model that re-scores a candidate list for one query. |
| Grounding | Support an answer with evidence from retrieved sources. |
| Tenant/workspace | An isolated collection of files, config, index, and usage in the API. |
| Ingestion | Loading, chunking, embedding, and indexing source documents. |
| Evaluation set | Questions paired with evidence phrases used to measure retrieval. |
| Factory | A function that chooses and constructs an implementation from a short spec/name. |
