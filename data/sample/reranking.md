# Re-ranking

A bi-encoder embeds the query and each chunk separately, which is fast but approximate. A cross-encoder reads the query and one chunk together and outputs a relevance score, which is slower but much more accurate.

The standard pattern is to retrieve about 30 candidates cheaply, rerank them with a cross-encoder, and keep the top 5 for the prompt. Reranking is often the single highest-impact upgrade after naive RAG because it fixes the order of results without changing the index. Typical choices are bge-reranker-base or ms-marco-MiniLM-L-6-v2.
