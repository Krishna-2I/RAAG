from .agentic import AgenticRAG
from .base import Strategy
from .contextual import MetadataRAG
from .corrective import CorrectiveRAG
from .graph import GraphIndex, GraphRAG
from .hierarchical import HierarchicalRAG
from .hybrid import HybridRAG, fuse_hits, rrf_fuse
from .naive import NaiveRAG
from .query_transform import MODES as QUERY_MODES
from .query_transform import QueryTransformRAG
from .rerank import OverlapReranker, RerankRAG, make_reranker
from .sql import SQLRAG

DEFAULT_RERANKER = "ce:cross-encoder/ms-marco-MiniLM-L-6-v2"
STRATEGY_NAMES = ["naive", "hybrid", "rerank"]  # need no LLM: used as the default eval set
ALL_STRATEGIES = STRATEGY_NAMES + QUERY_MODES + ["hierarchical", "metadata", "corrective", "agentic", "graph", "sql"]
NEEDS_LLM = set(QUERY_MODES) | {"metadata", "corrective", "agentic"}


def make_strategy(name, index, llm=None, k=5, reranker=DEFAULT_RERANKER, candidates=30, graph=None, db_path=None) -> Strategy:
    if name in NEEDS_LLM | {"sql"} and llm is None:
        raise ValueError(f"strategy {name!r} needs an LLM")
    if name == "naive":
        return NaiveRAG(index, llm, k)
    if name == "hybrid":
        return HybridRAG(index, llm, k, candidates)
    if name == "rerank":
        return RerankRAG(index, llm, k, make_reranker(reranker), candidates)
    if name in QUERY_MODES:
        return QueryTransformRAG(index, llm, k, name, candidates=candidates)
    if name == "hierarchical":
        return HierarchicalRAG(index, llm, k, candidates=candidates)
    if name == "metadata":
        return MetadataRAG(index, llm, k, candidates=candidates)
    if name == "corrective":
        return CorrectiveRAG(index, llm, k, candidates=candidates)
    if name == "agentic":
        return AgenticRAG(index, llm, k)
    if name == "graph":
        if graph is None:
            raise ValueError("graph strategy needs a graph (ingest with --build-graph)")
        return GraphRAG(index, llm, k, graph, candidates=candidates)
    if name == "sql":
        if not db_path:
            raise ValueError("sql strategy needs db_path")
        return SQLRAG(index, llm, k, db_path)
    raise ValueError(f"unknown strategy: {name!r} (choose from {ALL_STRATEGIES})")


__all__ = [
    "Strategy", "NaiveRAG", "HybridRAG", "RerankRAG", "OverlapReranker", "QueryTransformRAG", "HierarchicalRAG",
    "MetadataRAG", "CorrectiveRAG", "AgenticRAG", "GraphRAG", "GraphIndex", "SQLRAG", "rrf_fuse", "fuse_hits",
    "make_strategy", "make_reranker", "ALL_STRATEGIES", "STRATEGY_NAMES", "NEEDS_LLM",
]
