from __future__ import annotations

import argparse
from pathlib import Path

from .chunking import chunk_documents, semantic_chunk_documents
from .embeddings import make_embedder
from .evaluation.dataset import load_eval
from .evaluation.run import format_table, run_eval
from .index import Index
from .llm import make_llm
from .loaders import load_path
from .multimodal import load_multimodal, make_captioner
from .strategies import ALL_STRATEGIES, DEFAULT_RERANKER, STRATEGY_NAMES, GraphIndex, make_strategy
from .strategies.contextual import add_headers
from .strategies.hierarchical import build_parent_child_index, build_sentence_window_index


def _ingest(a):
    docs = load_multimodal(a.data, make_captioner(a.captioner)) if a.captioner else load_path(a.data)
    emb = make_embedder(a.embedder)
    size = a.chunk_size or (300 if a.mode == "parent-child" else 800)
    if a.mode == "parent-child":
        idx = build_parent_child_index(docs, emb, a.parent_size, size, a.overlap)
    elif a.mode == "sentence-window":
        idx = build_sentence_window_index(docs, emb, a.window)
    else:
        chunks = semantic_chunk_documents(docs, emb.embed) if a.mode == "semantic" else chunk_documents(docs, size, a.overlap)
        if a.headers:
            chunks = add_headers(chunks, docs)
        idx = Index.build(chunks, emb)
    idx.save(a.index)
    if a.build_graph:
        GraphIndex.build(idx.chunks, make_llm(a.build_graph)).save(Path(a.index) / "graph.json")
    print(f"{len(docs)} documents -> {len(idx.chunks)} chunks -> {a.index} (mode {a.mode}, embedder {emb.name}, dim {idx.vectors.shape[1]})")


def _strategy(a, idx, name, llm):
    g = Path(a.index) / "graph.json"
    graph = GraphIndex.load(g) if g.exists() else None
    return make_strategy(name, idx, llm, a.k, a.reranker, a.candidates, graph, a.db)


def _ask(a):
    idx = Index.load(a.index)
    ans = _strategy(a, idx, a.strategy, make_llm(a.llm)).run(a.question)
    print(ans.text, "\n")
    for i, h in enumerate(ans.hits, 1):
        print(f"[{i}] {h.chunk.id}  score={h.score:.3f}  ({h.source})")
    print(f"\nretrieve {ans.timings['retrieve_ms']:.0f} ms, generate {ans.timings['generate_ms']:.0f} ms")


def _eval(a):
    idx = Index.load(a.index)
    items = load_eval(a.eval)
    llm = make_llm(a.llm) if a.llm else None
    names = [s.strip() for s in a.strategies.split(",")]
    g = Path(a.index) / "graph.json"
    results = run_eval(idx, items, names, a.k, llm, a.judge, a.reranker, a.candidates, GraphIndex.load(g) if g.exists() else None, a.db)
    table = format_table(results, a.k, len(items))
    print(table)
    if a.show_misses:
        for name, r in results.items():
            for q in r["misses"]:
                print(f"MISS [{name}] {q}")
    if a.out:
        Path(a.out).write_text(f"# Results\n\nIndex embedder: `{idx.embedder.name}`, reranker: `{a.reranker}`\n\n{table}\n", encoding="utf-8")
        print(f"\nwrote {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="ragforge")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("ingest", help="load, chunk, embed and save an index")
    i.add_argument("--data", required=True)
    i.add_argument("--index", required=True)
    i.add_argument("--embedder", default="st:BAAI/bge-small-en-v1.5")
    i.add_argument("--mode", default="flat", choices=["flat", "semantic", "parent-child", "sentence-window"])
    i.add_argument("--chunk-size", type=int, default=None)
    i.add_argument("--overlap", type=int, default=120)
    i.add_argument("--parent-size", type=int, default=1500)
    i.add_argument("--window", type=int, default=2)
    i.add_argument("--headers", action="store_true", help="prepend the document title to every chunk")
    i.add_argument("--captioner", default=None, help="vision model spec (ollama:llava, anthropic:<model>) to index images and CSV tables")
    i.add_argument("--build-graph", default=None, metavar="LLM_SPEC", help="extract a knowledge graph with this LLM")
    i.set_defaults(fn=_ingest)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--index", required=True)
    common.add_argument("-k", type=int, default=5)
    common.add_argument("--reranker", default=DEFAULT_RERANKER)
    common.add_argument("--candidates", type=int, default=30)
    common.add_argument("--db", default=None, help="sqlite file for the sql strategy")

    q = sub.add_parser("ask", parents=[common], help="ask one question")
    q.add_argument("question")
    q.add_argument("--strategy", default="hybrid", choices=ALL_STRATEGIES)
    q.add_argument("--llm", default="ollama:llama3.1")
    q.set_defaults(fn=_ask)

    e = sub.add_parser("eval", parents=[common], help="compare strategies on an eval set")
    e.add_argument("--eval", required=True)
    e.add_argument("--strategies", default=",".join(STRATEGY_NAMES))
    e.add_argument("--llm", default=None, help="needed for LLM-based strategies and --judge")
    e.add_argument("--judge", action="store_true", help="also score faithfulness and answer relevance")
    e.add_argument("--show-misses", action="store_true")
    e.add_argument("--out", default=None)
    e.set_defaults(fn=_eval)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
