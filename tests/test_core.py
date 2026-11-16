import numpy as np
import pytest

from ragforge.bm25 import BM25
from ragforge.chunking import chunk_documents, chunk_text
from ragforge.embeddings import HashingEmbedder, make_embedder
from ragforge.evaluation.dataset import load_eval
from ragforge.evaluation.metrics import evaluate_hits
from ragforge.evaluation.run import format_table, run_eval
from ragforge.index import EmbedderMismatch, Index
from ragforge.llm import ExtractiveLLM
from ragforge.loaders import load_path
from ragforge.strategies import make_strategy, rrf_fuse
from ragforge.types import Chunk, Hit

SAMPLE = "data/sample"


@pytest.fixture(scope="module")
def index():
    chunks = chunk_documents(load_path(SAMPLE), size=300, overlap=50)
    return Index.build(chunks, HashingEmbedder(512))


def test_chunk_sizes_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 5
    assert all(len(c) <= 200 + 40 for c in chunks)
    assert chunks[1][:10] in chunks[0]  # overlap tail repeats the end of the previous chunk
    assert "Sentence number 199 is here." in chunks[-1]


def test_chunk_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)


def test_chunk_documents_ids():
    chunks = chunk_documents(load_path(SAMPLE), 300, 50)
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.metadata["source"].endswith(".md") for c in chunks)


def test_bm25_prefers_rare_exact_term():
    bm = BM25(["the cat sat", "error e-4012 occurred here", "the dog ran and the cat sat"])
    assert bm.top_k("e-4012", 3)[0][0] == 1
    assert bm.top_k("zzz", 3) == []


def test_hash_embedder_normalized_and_deterministic():
    e = HashingEmbedder(128)
    a, b = e.embed(["hello world"]), e.embed(["hello world"])
    assert np.allclose(a, b) and abs(np.linalg.norm(a[0]) - 1) < 1e-5
    assert make_embedder("hash:128").name == "hash:128"


def test_rrf_rewards_agreement():
    fused = rrf_fuse([["a", "b", "c"], ["c", "b", "d"]])
    assert {cid for cid, _ in fused[:2]} == {"b", "c"}  # found by both lists, so they beat a and d
    assert dict(fused)["b"] == pytest.approx(1 / 62 + 1 / 62)
    assert dict(fused)["c"] == pytest.approx(1 / 63 + 1 / 61)


def test_metrics_math():
    hits = [Hit(Chunk(str(i), "d", t), 1.0) for i, t in enumerate(["nothing", "has the Evidence A here", "x"])]
    m = evaluate_hits(hits, ["evidence a", "evidence b"], k=3)
    assert m["hit"] == 1.0 and m["recall"] == 0.5 and m["mrr"] == 0.5
    assert evaluate_hits(hits, ["evidence a"], k=1)["hit"] == 0.0


def test_index_roundtrip_and_embedder_guard(tmp_path, index):
    index.save(tmp_path / "idx")
    loaded = Index.load(tmp_path / "idx")
    assert loaded.embedder.name == "hash:512" and len(loaded.chunks) == len(index.chunks)
    assert loaded.search_dense("embedding dimensions", 1)[0].chunk.id == index.search_dense("embedding dimensions", 1)[0].chunk.id
    with pytest.raises(EmbedderMismatch):
        Index.load(tmp_path / "idx", HashingEmbedder(64))


def test_all_strategies_return_k_hits_and_answer(index):
    llm = ExtractiveLLM()
    for name in ["naive", "hybrid", "rerank"]:
        s = make_strategy(name, index, llm, k=3, reranker="overlap")
        ans = s.run("What does error E-4012 mean?")
        assert len(ans.hits) == 3 and ans.strategy == name
        assert "[" in ans.text


def test_hybrid_finds_error_code(index):
    hits = make_strategy("hybrid", index, k=3).retrieve("E-4012")
    assert any("E-4012" in h.chunk.text for h in hits)


def test_eval_pipeline_end_to_end(index):
    items = load_eval("eval/sample_eval.jsonl")
    assert len(items) == 15
    res = run_eval(index, items, ["naive", "hybrid", "rerank"], k=3, reranker="overlap")
    table = format_table(res, 3, len(items))
    assert "| rerank |" in table and "baseline" in table
    for r in res.values():
        assert 0 <= r["metrics"]["recall"] <= 1
