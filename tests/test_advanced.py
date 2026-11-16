import sqlite3
import struct
import zlib

import pytest

from conftest import Scripted
from ragforge.chunking import semantic_chunk_text
from ragforge.embeddings import HashingEmbedder
from ragforge.multimodal import Captioner, load_multimodal
from ragforge.strategies import (AgenticRAG, CorrectiveRAG, GraphIndex, GraphRAG, HierarchicalRAG, HybridRAG,
                                 MetadataRAG, QueryTransformRAG, SQLRAG, make_strategy)
from ragforge.strategies.contextual import add_headers
from ragforge.strategies.hierarchical import build_parent_child_index, build_sentence_window_index
from ragforge.strategies.sql import validate_sql
from ragforge.chunking import chunk_documents


def has(hits, text):
    return any(text.lower() in h.chunk.text.lower() for h in hits)


# ---- World 3
def test_multi_query_adds_variants(small_index):
    llm = Scripted(lambda s, p: "1. E-4012 index schema\n2. embedding model mismatch\n3. re-index collection")
    s = QueryTransformRAG(small_index, llm, k=3, mode="multi")
    assert s.expand("what is wrong") == ["E-4012 index schema", "embedding model mismatch", "re-index collection"]
    assert has(s.retrieve("what is wrong"), "E-4012")


def test_hyde_uses_hypothetical_passage(small_index):
    llm = Scripted(lambda s, p: "Error E-4012 means the index schema does not match the embedding model.")
    hits = QueryTransformRAG(small_index, llm, k=3, mode="hyde").retrieve("my collection fails to load")
    assert has(hits, "E-4012")


@pytest.mark.parametrize("mode", ["rewrite", "stepback", "decompose"])
def test_other_query_modes_run(small_index, mode):
    llm = Scripted(lambda s, p: "reciprocal rank fusion\nhow rank fusion works")
    assert len(QueryTransformRAG(small_index, llm, k=3, mode=mode).retrieve("rrf?")) == 3


def test_parent_child_returns_parent_text(docs):
    idx = build_parent_child_index(docs, HashingEmbedder(512), parent_size=600, child_size=150, child_overlap=20)
    child = idx.search_dense("E-4012", 1)[0].chunk
    hits = HierarchicalRAG(idx, None, k=3).retrieve("What does error E-4012 mean?")
    assert any(len(h.chunk.text) > len(child.text) for h in hits)
    assert has(hits, "E-4012")


def test_sentence_window_returns_neighbours(docs):
    idx = build_sentence_window_index(docs, HashingEmbedder(512), window=1)
    hits = HierarchicalRAG(idx, None, k=2).retrieve("error E-5003 ran out of memory")
    assert has(hits, "E-5003") and has(hits, "batch size")


def test_semantic_chunking_covers_text(docs):
    text = next(d.text for d in docs if d.id == "embeddings.md")
    chunks = semantic_chunk_text(text, HashingEmbedder(512).embed, percentile=40)
    assert chunks and "dot product" in " ".join(chunks)


def test_headers_and_metadata_filter(docs):
    chunks = add_headers(chunk_documents(docs, 300, 50), docs)
    assert chunks[0].text.startswith("Document: ")
    from ragforge.index import Index
    idx = Index.build(chunks, HashingEmbedder(512))
    llm = Scripted(lambda s, p: '{"source": "troubleshooting.md", "bogus": 1}')
    s = MetadataRAG(idx, llm, k=3)
    assert s.self_query("anything about errors") == {"source": "troubleshooting.md"}
    assert all(h.chunk.metadata["source"] == "troubleshooting.md" for h in s.retrieve("anything about errors"))


def test_metadata_bad_filter_falls_back(small_index):
    llm = Scripted(lambda s, p: '{"source": "nope.md"}')
    assert len(MetadataRAG(small_index, llm, k=3).retrieve("rrf")) == 3


# ---- World 4
def test_corrective_keeps_only_relevant_and_rewrites(small_index):
    def fn(system, prompt):
        if "grade" in system:
            return "yes" if "E-4012" in prompt.split("Passage:")[1] else "no"
        return "E-4012 index schema"

    llm = Scripted(fn)
    s = CorrectiveRAG(small_index, llm, k=3, min_relevant=3, max_rounds=1, grounding_check=False)  # 3 relevant chunks never exist -> forces a rewrite
    hits = s.retrieve("my vectors will not load")
    assert hits and all("E-4012" in h.chunk.text for h in hits)
    assert any("Rewrite" in sys for sys, _ in llm.calls)


def test_corrective_fallback_and_abstain(small_index):
    llm = Scripted(lambda s, p: "1" if "digit" in s else ("no" if "grade" in s else "x"))
    s = CorrectiveRAG(small_index, llm, k=3, min_relevant=1, fallback=lambda q: ["web result"], max_rounds=0)
    hits = s.retrieve("zzz")
    assert hits[0].source == "fallback"
    assert s.generate("zzz", hits).startswith("I don't know")  # faithfulness judge returned 1 -> abstain


def test_agentic_routes_then_stops(small_index):
    replies = iter(['{"action":"search","tool":"docs","query":"E-4012"}', '{"action":"answer"}'])
    llm = Scripted(lambda s, p: next(replies))
    hits = AgenticRAG(small_index, llm, k=3).retrieve("what is E-4012")
    assert has(hits, "E-4012") and len(llm.calls) == 2


def test_agentic_falls_back_to_default_search(small_index):
    llm = Scripted(lambda s, p: "garbage")
    assert has(AgenticRAG(small_index, llm, k=3).retrieve("E-4012"), "E-4012")


# ---- World 5
def test_graph_build_walk_and_roundtrip(small_index, tmp_path):
    def fn(system, prompt):
        if "triples" in system:
            return '[["E-4012", "means", "schema mismatch"], ["schema mismatch", "fixed by", "re-indexing"]]' if "E-4012" in prompt else "[]"
        return "[]"

    g = GraphIndex.build(small_index.chunks, Scripted(fn))
    assert g.edges and "e-4012" in g.find_entities("What is E-4012?")
    g.save(tmp_path / "g.json")
    g2 = GraphIndex.load(tmp_path / "g.json")
    assert len(g2.edges) == len(g.edges)
    hits = GraphRAG(small_index, Scripted(lambda s, p: "[]"), k=3, graph=g2).retrieve("What is E-4012?")
    assert has(hits, "E-4012") and hits[0].source == "graph"


@pytest.fixture
def sales_db(tmp_path):
    path = tmp_path / "sales.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sales (region TEXT, amount INTEGER)")
    conn.executemany("INSERT INTO sales VALUES (?, ?)", [("north", 10), ("south", 32)])
    conn.commit()
    conn.close()
    return str(path)


def test_validate_sql():
    assert validate_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    for bad in ["DROP TABLE sales", "SELECT 1; DELETE FROM sales", "UPDATE sales SET amount=0", "SELECT * FROM t WHERE x IN (SELECT 1) ; --"]:
        with pytest.raises(ValueError):
            validate_sql(bad)


def test_sql_rag_retries_and_stays_read_only(sales_db):
    replies = iter(["DROP TABLE sales", "SELECT nope FROM sales", "SELECT SUM(amount) AS total FROM sales"])
    llm = Scripted(lambda s, p: next(replies))
    hits = SQLRAG(None, llm, db_path=sales_db).retrieve("total sales?")
    assert "42" in hits[0].chunk.text and len(llm.calls) == 3
    assert "Error" in llm.calls[1][1]  # error fed back for the retry
    assert sqlite3.connect(sales_db).execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 2
    with pytest.raises(sqlite3.OperationalError):  # the connection itself is read-only
        c = sqlite3.connect(f"file:{sales_db}?mode=ro", uri=True)
        c.execute("DELETE FROM sales")


def _png():
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    raw = zlib.compress(b"\x00\xff\x00\x00")
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)) + chunk(b"IDAT", raw) + chunk(b"IEND", b"")


def test_multimodal_loader(tmp_path):
    (tmp_path / "note.md").write_text("Plain text note about revenue.")
    (tmp_path / "chart.png").write_bytes(_png())
    (tmp_path / "t.csv").write_text("region,amount\n" + "\n".join(f"r{i},{i}" for i in range(25)))

    class Fake(Captioner):
        name = "fake"
        def caption(self, image, mime):
            return "a bar chart of quarterly revenue"

    docs = load_multimodal(tmp_path, Fake())
    mods = {d.metadata["modality"] for d in docs}
    assert mods == {"text", "image", "table"}
    assert sum(d.metadata["modality"] == "table" for d in docs) == 2  # 25 rows -> 2 groups of <=20
    assert all("| region | amount |" in d.text for d in docs if d.metadata["modality"] == "table")


# ---- Quest b5: one router over text + graph + SQL
def test_router_over_text_graph_sql(small_index, sales_db):
    g = GraphIndex()
    g.add_edge("E-4012", "means", "schema mismatch", small_index.chunks[0].id)
    tools = {
        "docs": HybridRAG(small_index, None, 3),
        "sql": SQLRAG(None, Scripted(lambda s, p: "SELECT SUM(amount) FROM sales"), db_path=sales_db),
    }
    replies = iter(['{"action":"search","tool":"sql","query":"total sales"}', '{"action":"search","tool":"docs","query":"E-4012"}', '{"action":"answer"}'])
    hits = AgenticRAG(small_index, Scripted(lambda s, p: next(replies)), k=3, tools=tools).retrieve("total sales and what is E-4012")
    assert any(h.source == "sql" for h in hits) and has(hits, "E-4012")


def test_make_strategy_guards(small_index):
    with pytest.raises(ValueError):
        make_strategy("multi", small_index, llm=None)
    with pytest.raises(ValueError):
        make_strategy("graph", small_index, llm=None)
