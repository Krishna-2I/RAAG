"""Quest b1 / b2: run every strategy on the same eval set and print one comparison table."""
from __future__ import annotations

from time import perf_counter

from ..strategies import DEFAULT_RERANKER, make_strategy
from .judge import answer_relevance, faithfulness
from .metrics import aggregate, evaluate_hits


def run_eval(index, items, strategies, k=5, llm=None, judge=False, reranker=DEFAULT_RERANKER, candidates=30, graph=None, db_path=None) -> dict:
    results = {}
    for name in strategies:
        strat = make_strategy(name, index, llm, k, reranker, candidates, graph, db_path)
        rows, lat, misses, faith, rel = [], [], [], [], []
        for it in items:
            t0 = perf_counter()
            hits = strat.retrieve(it["question"], k)
            lat.append((perf_counter() - t0) * 1e3)
            row = evaluate_hits(hits, it["evidence"], k)
            rows.append(row)
            if not row["hit"]:
                misses.append(it["question"])
            if judge and llm is not None:
                ans = strat.generate(it["question"], hits)
                f = faithfulness(llm, ans, [h.chunk.text for h in hits])
                r = answer_relevance(llm, it["question"], ans)
                faith += [f] if f is not None else []
                rel += [r] if r is not None else []
        res = aggregate(rows)
        res["latency_ms"] = sum(lat) / len(lat)
        if faith:
            res["faithfulness"] = sum(faith) / len(faith)
        if rel:
            res["answer_relevance"] = sum(rel) / len(rel)
        results[name] = {"metrics": res, "misses": misses}
    return results


def _delta(value: float, base: float) -> str:
    return "n/a" if base == 0 else f"{(value - base) / base * 100:+.1f}%"


def format_table(results: dict, k: int, n_questions: int) -> str:
    base = results.get("naive", {}).get("metrics")
    judged = any("faithfulness" in r["metrics"] for r in results.values())
    head = f"| Strategy | Hit@{k} | Recall@{k} | MRR | Precision@{k} | Retrieval ms | Recall vs naive | MRR vs naive |"
    sep = "|---|---|---|---|---|---|---|---|"
    if judged:
        head += " Faithfulness | Relevance |"
        sep += "---|---|"
    lines = [f"Eval set: {n_questions} questions, k={k}", "", head, sep]
    for name, r in results.items():
        m = r["metrics"]
        is_base = name == "naive"
        d_rec = "baseline" if is_base else "n/a" if base is None else _delta(m["recall"], base["recall"])
        d_mrr = "baseline" if is_base else "n/a" if base is None else _delta(m["mrr"], base["mrr"])
        line = (
            f"| {name} | {m['hit']:.2f} | {m['recall']:.2f} | {m['mrr']:.2f} | {m['precision']:.2f} "
            f"| {m['latency_ms']:.1f} | {d_rec} | {d_mrr} |"
        )
        if judged:
            line += f" {m.get('faithfulness', float('nan')):.2f} | {m.get('answer_relevance', float('nan')):.2f} |"
        lines.append(line)
    return "\n".join(lines)
